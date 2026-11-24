"""Train REPPO on the saved B′ and L PPO experiment settings and reference banks."""

import argparse
from dataclasses import asdict
from pathlib import Path
from shutil import copyfile
from time import perf_counter

from flax import serialization
import jax
import jax.numpy as jnp
import numpy as np

from envs.quadrotor_smooth import DT, REWARD_PROFILES, make_environment
from experiments.quadrotor_comparison import finite_parameters, read_csv, write_csv
from experiments.quadrotor_scratch import (
    VARIANTS, hash_file, load_bank, read_json, save_json, source_hashes as ppo_source_hashes, timestamp,
)
from experiments.quadrotor_thrust_preview import REWARD
from experiments.quadrotor_yaw_practice import PracticeEnvironment
from rl.quadrotor_reppo import QuadActor, ReppoConfig, make_learner


DEFAULT_PPO_RUN = Path("runs/se23_fresh_bprime_l_yaw_8m")
BANK_FILES = ("train_references.npz", "test_references.npz")
MATCHED_CONFIG = ("total_steps", "num_envs", "rollout_steps", "epochs", "minibatches",
                  "layers", "width", "learning_rate", "gamma", "max_grad_norm")
MATCHED_TASK = ("dt", "horizon", "reward_profile", "reward_settings", "variants", "seeds",
                "train_per_family", "test_per_family", "ref_seeds", "evaluation_seed", "checkpoint_every")


def source_hashes():
    root = Path(__file__).resolve().parents[1]
    sources = ("rl/quadrotor_reppo.py", "experiments/quadrotor_reppo.py")
    return {**ppo_source_hashes(), **{name: hash_file(root / name) for name in sources}}


def check_banks(directory, manifest):
    for name in BANK_FILES:
        if hash_file(directory / name) != manifest["reference_sha256"][name]:
            raise ValueError(f"Reference bank hash differs for {directory / name}")


def create_run(output, ppo_run, overrides):
    baseline = read_json(ppo_run / "manifest.json")
    if (tuple(baseline["variants"]) != VARIANTS or baseline["dt"] != DT
            or baseline["reward_profile"] != REWARD or baseline["reward_settings"] != REWARD_PROFILES[REWARD]._asdict()):
        raise ValueError("Use a fresh B′/L PPO experiment with the matching simulator and reward")
    sources = source_hashes()
    task_sources = ("envs/quadrotor_smooth.py", "envs/quadrotor_desired_thrust.py",
                    "envs/quadrotor_double_thrust.py", "envs/quadrotor_thrust_rate.py",
                    "experiments/quadrotor_yaw_practice.py", "experiments/quadrotor_thrust_preview.py",
                    "rl/quadrotor_ppo.py")
    if any(baseline["source_sha256"].get(name) != sources[name] for name in task_sources):
        raise ValueError("The environment or policy sources differ from the saved PPO run")
    settings = baseline["ppo"]
    config = ReppoConfig(**{key: settings[key] for key in MATCHED_CONFIG},
                         td_lambda=settings["gae_lambda"], **overrides)
    config.validate()
    interval = baseline["checkpoint_every"]
    if interval < config.batch_size or interval % config.batch_size:
        raise ValueError("The PPO checkpoint interval must divide into whole rollout batches")
    check_banks(ppo_run, baseline)
    output.mkdir(parents=True, exist_ok=False)
    for name in BANK_FILES:
        copyfile(ppo_run / name, output / name)
    manifest = dict(
        status="ready", algorithm="REPPO", created_utc=timestamp(), ppo_run=str(ppo_run),
        reppo=asdict(config), source_sha256=sources, reference_sha256=baseline["reference_sha256"],
        backend=jax.default_backend(), jax_version=jax.__version__,
        **{key: baseline[key] for key in MATCHED_TASK},
    )
    check_banks(output, manifest)
    save_json(output / "manifest.json", manifest)
    return manifest


def save_bytes(path, payload):
    temporary = path.with_suffix(".tmp")
    temporary.write_bytes(payload)
    temporary.replace(path)


def checkpoint(directory, variant, env, config, seeds, runners, update, *, export, elapsed, rate):
    steps = update * config.batch_size
    tag = f"step_{steps:09d}"
    state = directory / f"{tag}.msgpack"
    save_bytes(state, serialization.to_bytes(runners))
    if export:
        for lane, seed in enumerate(seeds):
            folder = directory / f"seed_{seed}" / tag
            folder.mkdir(parents=True, exist_ok=True)
            policy_params = jax.tree_util.tree_map(lambda x: x[lane], runners.actor.params["policy"])
            save_bytes(folder / "policy.msgpack", serialization.to_bytes({"params": policy_params}))
            save_json(folder / "config.json", dict(
                algorithm="REPPO", variant=variant, seed=seed, steps=steps, reppo=asdict(config),
                observation_size=env.obs_size, observation_scale=np.asarray(env.scale).tolist(),
                initial_action_mean=list(env.initial_action_mean), reward_profile=REWARD,
            ))
    latest = directory / "latest.json"
    old = read_json(latest)["file"] if latest.exists() else None
    save_json(latest, dict(update=update, steps=steps, file=state.name,
                           elapsed_seconds=elapsed, seconds_per_update=rate))
    if old and old != state.name:
        (directory / old).unlink(missing_ok=True)


def load_actor(directory):
    metadata = read_json(directory / "config.json")
    if metadata.get("algorithm") != "REPPO":
        raise ValueError(f"Expected a REPPO actor in {directory}")
    config = metadata["reppo"]
    model = QuadActor(config["layers"], config["width"], tuple(metadata["initial_action_mean"]))
    template = model.init(jax.random.PRNGKey(0), jnp.zeros(metadata["observation_size"]))
    params = serialization.from_bytes(template, (directory / "policy.msgpack").read_bytes())
    return model, params, metadata


def train_variant(output, variant, bank, families, manifest, progress_every):
    directory = output / variant
    directory.mkdir(exist_ok=True)
    config = ReppoConfig(**manifest["reppo"])
    seeds = manifest["seeds"]
    env = PracticeEnvironment(make_environment(bank, variant, reward_profile=REWARD), families)
    _, initialize, update = make_learner(env, config)
    runners = jax.jit(jax.vmap(initialize))(jnp.asarray(seeds, dtype=jnp.uint32))
    update = jax.jit(jax.vmap(update))
    metrics_file = directory / "training.csv"
    latest = directory / "latest.json"
    start, previous_elapsed, rate, rows = 0, 0., None, []
    if latest.is_file():
        saved = read_json(latest)
        start, previous_elapsed, rate = saved["update"], saved["elapsed_seconds"], saved["seconds_per_update"]
        if start == config.updates:
            return rate
        runners = jax.device_put(serialization.from_bytes(runners, (directory / saved["file"]).read_bytes()))
        rows = [row for row in read_csv(metrics_file) if int(row["steps"]) <= saved["steps"]]

    print(f"{variant}: starting at {start * config.batch_size:,} of {config.total_steps:,} steps per seed", flush=True)
    export_every = manifest["checkpoint_every"] // config.batch_size
    durations = []
    timer = perf_counter()
    for index in range(start + 1, config.updates + 1):
        began = perf_counter()
        runners, metrics = jax.block_until_ready(update(runners))
        duration = perf_counter() - began
        if index > start + 1:
            durations.append(duration)
        rate = float(np.mean(durations[-64:])) if durations else rate or duration
        steps = index * config.batch_size
        values = jax.device_get(metrics)
        if not all(np.isfinite(value).all() for value in values.values()):
            raise FloatingPointError(f"Nonfinite REPPO metrics for {variant} at {steps:,} steps")
        elapsed = previous_elapsed + perf_counter() - timer
        for lane, seed in enumerate(seeds):
            rows.append(dict(steps=steps, seed=seed, elapsed_seconds=elapsed,
                             **{key: float(value[lane]) for key, value in values.items()}))
        export = index % export_every == 0 or index == config.updates
        if export or index % progress_every == 0 or index == start + 1:
            if not finite_parameters((runners.actor.params, runners.critic.params)):
                raise FloatingPointError(f"Nonfinite REPPO parameters for {variant} at {steps:,} steps")
            temporary_csv = metrics_file.with_suffix(".tmp")
            write_csv(temporary_csv, rows)
            temporary_csv.replace(metrics_file)
            checkpoint(directory, variant, env, config, seeds, runners, index,
                       export=export, elapsed=elapsed, rate=rate)
            print(f"{variant}: {steps:,}/{config.total_steps:,} steps/seed | "
                  f"reward {values['step_reward'].mean():.3f} | KL {values['forward_kl'].mean():.4f} | "
                  f"{rate:.2f} s/update", flush=True)
    return rate


def train(output, *, ppo_run=None, resume=False, overrides=None, progress_every=64):
    output = output.resolve()
    manifest_path = output / "manifest.json"
    if resume:
        manifest = read_json(manifest_path)
        if manifest.get("algorithm") != "REPPO":
            raise ValueError("Resume requires a REPPO run directory")
        if manifest["source_sha256"] != source_hashes():
            raise ValueError("The training sources differ from the saved REPPO run")
        check_banks(output, manifest)
    else:
        manifest = create_run(output, (ppo_run or DEFAULT_PPO_RUN).resolve(), overrides or {})
    bank, families = load_bank(output / BANK_FILES[0])
    manifest.pop("error", None)
    timings = manifest.setdefault("post_compile_seconds_per_update", {})
    try:
        for variant in manifest["variants"]:
            manifest.update(status="training", current_variant=variant, updated_utc=timestamp())
            save_json(manifest_path, manifest)
            timings[variant] = train_variant(output, variant, bank, families, manifest, progress_every)
            jax.clear_caches()
        manifest.update(status="trained", updated_utc=timestamp())
        manifest.pop("current_variant", None)
        save_json(manifest_path, manifest)
    except BaseException as error:
        manifest.update(status="interrupted" if isinstance(error, KeyboardInterrupt) else "failed",
                        error=f"{type(error).__name__}: {error}", updated_utc=timestamp())
        save_json(manifest_path, manifest)
        raise


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    source = parser.add_mutually_exclusive_group()
    source.add_argument("--ppo-run", type=Path, help=f"PPO run with saved settings (default {DEFAULT_PPO_RUN})")
    source.add_argument("--resume", action="store_true")
    parser.add_argument("--progress-every", type=int, default=64, help="Updates between restart checkpoints")
    parser.add_argument("--q-min", type=float, help="Lowest HL-Gauss bin center in scaled return units")
    parser.add_argument("--q-max", type=float, help="Highest HL-Gauss bin center in scaled return units")
    parser.add_argument("--auxiliary-weight", type=float, help="Weight of the critic's latent prediction loss")
    args = parser.parse_args()
    overrides = {key: getattr(args, key) for key in ("q_min", "q_max", "auxiliary_weight")
                 if getattr(args, key) is not None}
    if args.progress_every < 1:
        parser.error("--progress-every must be positive")
    if args.resume and overrides:
        parser.error("A resumed run uses its saved critic settings")
    train(args.output, ppo_run=args.ppo_run, resume=args.resume, overrides=overrides,
          progress_every=args.progress_every)


if __name__ == "__main__":
    main()
