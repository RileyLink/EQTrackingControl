"""Generate split-by-trajectory data shared by all three policy experiments."""

import argparse
from pathlib import Path

import jax
import jax.numpy as jnp
import numpy as np

from envs.quadrotor_smooth import DT, at, make_bank
from experiments.policy_structure.common import (
    ACTION_SCALES, SCHEMA_VERSION, SPLITS, Policy, actor_requested_wrench, actor_wrench, checkpoint_hashes,
    concatenate, digest, feasible_wrench, take, wrench_controller,
)
from experiments.policy_structure.lee import DEFAULT_GAINS, Gains, lee_terms, lee_wrench
from experiments.policy_structure.rollouts import (
    STRESS_BOUNDS, perturb_states, reference_states, reset_states, simulate,
)
from experiments.quadrotor_comparison import write_csv, write_json
from experiments.quadrotor_thrust_preview import vertical_pairs


PROBE_BOUNDS = (0.45, 0.45, 0.26, 0.30)  # m, m/s, rad, rad/s per component


def reference_banks(seed, count, steps):
    banks, families, splits, seeds = [], [], [], {}
    for index, split in enumerate(SPLITS):
        base = seed + 100 * index
        seeds[split] = dict(nominal=base, vertical=base + 1)
        _, spatial = make_bank(base, count, steps, dt=DT)
        vertical, labels = vertical_pairs(base + 1, count // 2, steps)
        banks.extend((spatial, vertical))
        names = ["nominal_3d"] * count + np.where(labels["family"] == 0, "vertical_rate",
                                                 "vertical_curvature").tolist()
        if split == "test":
            seeds[split]["faster"] = base + 2
            _, faster = make_bank(base + 2, count, steps, dt=DT, frequency_scale=1.3)
            banks.append(faster)
            names += ["faster_3d"] * count
        families.extend(names)
        splits.extend([split] * len(names))
    return concatenate(banks), np.asarray(splits), np.asarray(families), seeds


def rollout_samples(env, controller, starts, stride):
    frames = simulate(env, controller, starts)
    active = np.asarray(frames["active"])[::stride]
    return jax.tree_util.tree_map(lambda values: values[::stride][active], frames["state"])


def run(checkpoint, output, *, seed=581700, count=8, steps=240, stride=4,
        probes_per_trajectory=20, gains=DEFAULT_GAINS):
    if count < 2 or count % 2 or min(steps, stride, probes_per_trajectory) < 1:
        raise ValueError("Use a positive even trajectory count per family and positive horizon, stride and probe count")
    if not 0 <= seed < 2**32 - 1000:
        raise ValueError("Seed must be in [0, 2**32 - 1000)")
    if not np.isfinite(gains).all() or min(gains) <= 0:
        raise ValueError("All four Lee baseline gains must be finite and positive")
    policy = Policy.load(checkpoint)
    checkpoint_sha256 = checkpoint_hashes(policy.directory)
    bank, splits, families, seeds = reference_banks(seed, count, steps)
    env = policy.environment(bank, DT)
    output = Path(output)
    output.mkdir(parents=True, exist_ok=False)

    trajectory = jnp.arange(len(bank.pos))
    starts = reset_states(env, trajectory, seed + 301)
    actor = policy.controller(env)

    def lee(state):
        ref = at(bank, state.trajectory, state.step)
        terms = lee_terms(state, ref, gains)
        return lee_wrench(terms, gains), terms.valid_commanded_attitude

    rl = rollout_samples(env, actor, starts, stride)
    analytical = rollout_samples(env, wrench_controller(env, lambda s: lee(s)[0]), starts, stride)
    rng = np.random.default_rng(seed + 302)
    trajectories = jnp.repeat(trajectory, probes_per_trajectory)
    probe_times = jnp.asarray(rng.integers(0, steps, len(trajectories)), dtype=jnp.int32)
    amplitudes = rng.choice([.35, .8, 1.5], len(trajectories))[:, None, None]
    probes = perturb_states(reference_states(env, trajectories, probe_times), rng, PROBE_BOUNDS, amplitudes)
    states = concatenate((rl, analytical, probes))
    source = np.repeat(["rl_rollout", "lee_rollout", "perturbed"],
                       [len(rl.pos), len(analytical.pos), len(probes.pos)])

    @jax.jit
    def query(states):
        observation = jax.vmap(env.observe)(states)
        native = policy.from_observation(observation)
        applied = jax.vmap(lambda u: actor_wrench(env, u))(native)
        requested = jax.vmap(lambda u: actor_requested_wrench(env, u))(native)
        lee_requested, valid = jax.vmap(lee)(states)
        return dict(observation=observation, rl_native=native, rl_requested=requested, rl_wrench=applied,
                    lee_requested=lee_requested, lee_wrench=jax.vmap(feasible_wrench)(lee_requested),
                    lee_attitude_valid=valid)

    queries = query(states)
    if not all(np.isfinite(np.asarray(value)).all() for value in queries.values()):
        raise ValueError("Non-finite commands or observations in the sampled states")
    np.savez_compressed(output / "samples.npz", **states._asdict(), **queries, source=source)
    np.savez_compressed(output / "references.npz", **bank._asdict(), split=splits, family=families)
    write_csv(output / "trajectories.csv", [
        dict(trajectory=i, split=split, family=family,
             rl_rollout=int(np.sum(np.asarray(rl.trajectory) == i)),
             lee_rollout=int(np.sum(np.asarray(analytical.trajectory) == i)),
             perturbed=probes_per_trajectory)
        for i, (split, family) in enumerate(zip(splits, families))
    ])
    if checkpoint_sha256 != checkpoint_hashes(policy.directory):
        raise RuntimeError("Checkpoint changed during collection")
    repo = Path(__file__).resolve().parents[2]
    sources = ["envs/quadrotor_smooth.py", "envs/quadrotor_desired_thrust.py",
               "envs/quadrotor_double_thrust.py", "envs/quadrotor_thrust_rate.py", "rl/quadrotor_ppo.py",
               "experiments/quadrotor_comparison.py", "experiments/quadrotor_thrust_preview.py"]
    sources += [str(path.relative_to(repo)) for path in Path(__file__).parent.glob("*.py")]
    write_json(output / "manifest.json", dict(
        schema_version=SCHEMA_VERSION, checkpoint=str(policy.directory), checkpoint_sha256=checkpoint_sha256,
        checkpoint_config=policy.config, seed=seed, reference_seeds=seeds,
        reset_seed=seed + 301, probe_seed=seed + 302, stress_seed=seed + 303,
        trajectories_per_family=count, horizon_steps=steps, dt=DT, rollout_stride=stride,
        probes_per_trajectory=probes_per_trajectory, probe_bounds=PROBE_BOUNDS, stress_bounds=STRESS_BOUNDS,
        lee_gains=gains._asdict(), action_scales=ACTION_SCALES.tolist(), observation_scale=np.asarray(env.scale).tolist(),
        observation_definition=getattr(env, "feature_definition", None),
        action_definition=getattr(env, "action_definition", "four-normalized-rotor-commands"),
        sample_count=len(source), invalid_lee_attitude=int(np.sum(~np.asarray(queries["lee_attitude_valid"]))),
        artifacts_sha256={name: digest(output / name) for name in ("references.npz", "samples.npz")},
        source_sha256={name.replace("\\", "/"): digest(repo / name) for name in sources},
        jax_version=jax.__version__, numpy_version=np.__version__, device=str(jax.devices()[0]),
    ))
    print(f"Saved {len(source):,} observations on {len(trajectory)} trajectories to {output}", flush=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", type=Path, required=True, help="Directory with config.json and policy.msgpack")
    parser.add_argument("--output", type=Path, required=True, help="New dataset directory")
    parser.add_argument("--seed", type=int, default=581700)
    parser.add_argument("--count", type=int, default=8, help="Even number of trajectories per split and family")
    parser.add_argument("--steps", type=int, default=240)
    parser.add_argument("--stride", type=int, default=4, help="Subsample each tracking rollout")
    parser.add_argument("--probes-per-trajectory", type=int, default=20)
    parser.add_argument("--lee-gains", type=float, nargs=4, metavar=("KP", "KV", "KR", "KW"), default=DEFAULT_GAINS)
    args = parser.parse_args()
    run(args.checkpoint, args.output, seed=args.seed, count=args.count, steps=args.steps,
        stride=args.stride, probes_per_trajectory=args.probes_per_trajectory, gains=Gains(*args.lee_gains))


if __name__ == "__main__":
    main()
