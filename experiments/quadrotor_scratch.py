"""Train independent B′ and L actors with PPO alone on a shared practice bank."""

import argparse
import csv
from dataclasses import asdict
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
from time import perf_counter

from flax import serialization
import jax
import numpy as np

from envs.quadrotor_smooth import DT, REWARD_PROFILES, Reference, make_environment
from experiments.quadrotor_comparison import finite_parameters, write_csv
from experiments.quadrotor_longer import independent_runners
from experiments.quadrotor_thrust_preview import B_PRIME, L, REWARD
from experiments.quadrotor_yaw_practice import (
    EVALUATION_SEED, TEST_BANK_SEED, TRAIN_BANK_SEED, PracticeEnvironment, build_bank,
)
from rl.quadrotor_ppo import PPOConfig


VARIANTS = (B_PRIME, L)
DEFAULT_STEPS = 8_388_608
CHECKPOINT_EVERY = 1_048_576
SOURCES = (
    "envs/quadrotor_smooth.py", "envs/quadrotor_desired_thrust.py", "envs/quadrotor_double_thrust.py",
    "envs/quadrotor_thrust_rate.py", "rl/quadrotor_ppo.py", "experiments/quadrotor_longer.py",
    "experiments/quadrotor_thrust_preview.py", "experiments/quadrotor_yaw_practice.py",
    "experiments/quadrotor_scratch.py", "experiments/quadrotor_comparison.py",
)


def timestamp():
    return datetime.now(timezone.utc).isoformat()


def hash_file(path):
    with path.open("rb") as stream:
        return hashlib.file_digest(stream, "sha256").hexdigest()


def save_json(path, value):
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(value, indent=2, allow_nan=False) + "\n", encoding="utf-8")
    temporary.replace(path)


def read_json(path):
    return json.loads(path.read_text(encoding="utf-8"))


def save_bank(path, bank, families, pairs):
    np.savez_compressed(path, **bank._asdict(), families=families,
                        **{f"vertical_{name}": value for name, value in pairs.items()})


def load_bank(path):
    with np.load(path, allow_pickle=False) as arrays:
        bank = Reference(*(jax.device_put(arrays[name]) for name in Reference._fields))
        families = arrays["families"]
    return bank, families


def source_hashes():
    repo = Path(__file__).resolve().parents[1]
    return {source: hash_file(repo / source) for source in SOURCES}


def create_run(output, config, seeds, horizon, per_family, test_per_family, checkpoint_every):
    train, families, pairs = build_bank(TRAIN_BANK_SEED, per_family, horizon)
    test, test_families, test_pairs = build_bank(TEST_BANK_SEED, test_per_family, horizon)
    output.mkdir(parents=True, exist_ok=False)
    save_bank(output / "train_references.npz", train, families, pairs)
    save_bank(output / "test_references.npz", test, test_families, test_pairs)
    manifest = dict(
        status="ready", created_utc=timestamp(), ppo=asdict(config), seeds=list(seeds), variants=VARIANTS,
        seed_batching="independent-vmap", initial_policy="fresh independent actor and critic",
        warmup_updates=0, auxiliary_loss_weight=0., dt=DT, horizon=horizon, backend=jax.default_backend(),
        jax_version=jax.__version__, reward_profile=REWARD, reward_settings=REWARD_PROFILES[REWARD]._asdict(),
        train_per_family=per_family, test_per_family=test_per_family, checkpoint_every=checkpoint_every,
        ref_seeds=dict(train=TRAIN_BANK_SEED, test=TEST_BANK_SEED), evaluation_seed=EVALUATION_SEED,
        normalized_observation="existing fixed physical scales", source_sha256=source_hashes(),
        reference_sha256={name: hash_file(output / name)
                          for name in ("train_references.npz", "test_references.npz")},
    )
    save_json(output / "manifest.json", manifest)
    return manifest, train, families


def checkpoint(directory, variant, env, config, seeds, runners, update, *, export, elapsed, rate):
    step = update * config.batch_size
    tag = f"step_{step:09d}"
    payload = serialization.to_bytes(runners)
    state_path = directory / f"{tag}.msgpack"
    temporary = state_path.with_suffix(".tmp")
    temporary.write_bytes(payload)
    temporary.replace(state_path)
    if export:
        for lane, seed in enumerate(seeds):
            folder = directory / f"seed_{seed}" / tag
            folder.mkdir(parents=True, exist_ok=True)
            params = jax.tree_util.tree_map(lambda value: value[lane], runners.train_state.params)
            (folder / "policy.msgpack").write_bytes(serialization.to_bytes(params))
            save_json(folder / "config.json", dict(
                variant=variant, seed=seed, steps=step, ppo=asdict(config),
                observation_size=env.obs_size, observation_scale=np.asarray(env.scale).tolist(),
                initial_action_mean=list(env.initial_action_mean), reward_profile=REWARD,
            ))
    current = directory / "latest.json"
    old = read_json(current)["file"] if current.exists() else None
    save_json(current, dict(update=update, steps=step, file=state_path.name,
                            elapsed_seconds=elapsed, seconds_per_update=rate))
    if old and old != state_path.name:
        (directory / old).unlink(missing_ok=True)


def train_variant(output, variant, bank, families, manifest, *, progress_every):
    directory = output / variant
    directory.mkdir(exist_ok=True)
    config = PPOConfig(**manifest["ppo"])
    seeds = manifest["seeds"]
    env = PracticeEnvironment(make_environment(bank, variant, reward_profile=REWARD), families)
    _, runners, update = independent_runners(env, config, seeds)
    latest = directory / "latest.json"
    start, prior_elapsed = 0, 0.
    if latest.is_file():
        saved = read_json(latest)
        start, prior_elapsed = saved["update"], saved["elapsed_seconds"]
        runners = jax.device_put(serialization.from_bytes(runners, (directory / saved["file"]).read_bytes()))
    print(f"{variant}: starting at {start * config.batch_size:,} of {config.total_steps:,} steps per seed",
          flush=True)
    if start == config.updates:
        return read_json(latest)["seconds_per_update"]
    exported_every = manifest["checkpoint_every"] // config.batch_size
    timer = perf_counter()
    durations, rows = [], []
    prior_rate = read_json(latest)["seconds_per_update"] if start else None
    metrics_file = directory / "training.csv"
    if start and metrics_file.is_file():
        with metrics_file.open(encoding="utf-8", newline="") as stream:
            rows.extend(row for row in csv.DictReader(stream) if int(row["steps"]) <= start * config.batch_size)
    for index in range(start + 1, config.updates + 1):
        began = perf_counter()
        runners, metrics = jax.block_until_ready(update(runners))
        elapsed_update = perf_counter() - began
        if index > start + 1:
            durations.append(elapsed_update)
        step = index * config.batch_size
        values = {key: np.asarray(value) for key, value in metrics.items()}
        if not all(np.isfinite(value).all() for value in values.values()):
            raise FloatingPointError(f"Nonfinite PPO metrics for {variant} at {step:,} steps")
        elapsed = prior_elapsed + perf_counter() - timer
        rate = float(np.mean(durations[-64:])) if durations else (prior_rate or elapsed_update)
        for lane, seed in enumerate(seeds):
            rows.append(dict(steps=step, seed=seed, elapsed_seconds=elapsed, **{
                key: float(value[lane]) for key, value in values.items()
            }))
        export = index % exported_every == 0 or index == config.updates
        if export or index % progress_every == 0 or index == start + 1:
            if not finite_parameters(runners.train_state.params):
                raise FloatingPointError(f"Nonfinite policy parameters for {variant} at {step:,} steps")
            write_csv(metrics_file, rows)
            checkpoint(directory, variant, env, config, seeds, runners, index,
                       export=export, elapsed=elapsed, rate=rate)
            print(f"{variant}: {step:,}/{config.total_steps:,} steps/seed | "
                  f"reward {values['step_reward'].mean():.3f} | {rate:.2f} s/update", flush=True)
    return rate


def train(args):
    output = args.output.resolve()
    if args.resume:
        manifest = read_json(output / "manifest.json")
        if manifest["source_sha256"] != source_hashes():
            raise ValueError("The training sources differ from the saved configuration")
        for name, expected in manifest["reference_sha256"].items():
            if hash_file(output / name) != expected:
                raise ValueError(f"Saved bank hash differs: {name}")
        bank, families = load_bank(output / "train_references.npz")
    else:
        config = PPOConfig(total_steps=args.steps, layers=6, width=args.width)
        config.validate()
        if (not args.seeds or len(set(args.seeds)) != len(args.seeds) or min(args.seeds) < 0
                or args.checkpoint_every < config.batch_size or args.checkpoint_every % config.batch_size):
            raise ValueError("Use distinct nonnegative seeds and a checkpoint interval divisible by the PPO batch")
        manifest, bank, families = create_run(output, config, args.seeds, args.horizon, args.per_family,
                                              args.test_per_family, args.checkpoint_every)
    train_begin = perf_counter()
    timings = dict(manifest.get("post_compile_seconds_per_update", {}))
    manifest.update(status="training", updated_utc=timestamp())
    manifest.pop("error", None)
    save_json(output / "manifest.json", manifest)
    try:
        for variant in VARIANTS:
            manifest.update(current_variant=variant, updated_utc=timestamp())
            save_json(output / "manifest.json", manifest)
            timings[variant] = train_variant(output, variant, bank, families, manifest,
                                             progress_every=args.progress_every)
            manifest["post_compile_seconds_per_update"] = timings
            jax.clear_caches()
        manifest.update(status="trained", updated_utc=timestamp(), training_completed_utc=timestamp(),
                        training_session_seconds=perf_counter() - train_begin)
        manifest.pop("current_variant", None)
        save_json(output / "manifest.json", manifest)
    except BaseException as error:
        manifest.update(status="interrupted" if isinstance(error, KeyboardInterrupt) else "failed",
                        error=f"{type(error).__name__}: {error}", updated_utc=timestamp())
        save_json(output / "manifest.json", manifest)
        raise


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--steps", type=int, default=DEFAULT_STEPS)
    parser.add_argument("--width", type=int, default=256)
    parser.add_argument("--seeds", type=int, nargs="+", default=[0, 1, 2])
    parser.add_argument("--horizon", type=int, default=240)
    parser.add_argument("--per-family", type=int, default=64)
    parser.add_argument("--test-per-family", type=int, default=32)
    parser.add_argument("--checkpoint-every", type=int, default=CHECKPOINT_EVERY)
    parser.add_argument("--progress-every", type=int, default=64, help="Updates between restart checkpoints")
    args = parser.parse_args()
    if args.progress_every < 1:
        parser.error("--progress-every must be positive")
    train(args)


if __name__ == "__main__":
    main()
