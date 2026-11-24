"""Longer training-reward comparison, batching independent seeds with JAX vmap.

Reuses the existing task and PPO update unchanged. Each seed has independent
parameters, optimizer state, environment state and random-number streams.
"""

import argparse
from dataclasses import asdict, replace
from datetime import datetime, timezone
import hashlib
import importlib.metadata
import json
from pathlib import Path
import platform
import time

from flax import serialization
import jax
import jax.numpy as jnp
import numpy as np

from envs.quadrotor_smooth import (
    DEFAULT_VARIANTS, DT, VARIANTS, REWARD_PROFILES, POSITION_FOCUS_VARIANTS, make_bank, make_environment,
)
from experiments.quadrotor_comparison import (
    PRESETS, finite_parameters, report, save_bank, write_csv, write_json,
)
from rl.quadrotor_ppo import make_learner


def independent_runners(env, config, seeds, *, actor_auxiliary_loss=None):
    """Share only static functions, never parameters or data, between seeds."""
    model, base, update = make_learner(env, config, seeds[0], actor_auxiliary_loss=actor_auxiliary_loss)
    runners = [base]
    for seed in seeds[1:]:
        _, runner, _ = make_learner(env, config, seed, actor_auxiliary_loss=actor_auxiliary_loss)
        # TrainState marks functions/optimizer definitions as static PyTree
        # metadata. Reuse these identities to stack independent numeric state.
        runner = runner._replace(train_state=runner.train_state.replace(
            apply_fn=base.train_state.apply_fn, tx=base.train_state.tx))
        runners.append(runner)
    batch = jax.tree_util.tree_map(lambda *values: jnp.stack(values), *runners)
    return model, batch, jax.jit(jax.vmap(update))


def run(args):
    config = replace(PRESETS["pilot"]["config"], total_steps=args.steps)
    config.validate()
    seeds = args.seeds
    if not seeds or len(set(seeds)) != len(seeds) or any(seed < 0 for seed in seeds):
        raise ValueError("Use distinct nonnegative seeds")
    if args.checkpoint_every < 1:
        raise ValueError("Checkpoint interval must be positive")
    variants = args.variants
    if not variants or len(set(variants)) != len(variants):
        raise ValueError("Choose distinct variants")
    if args.reward_profile != "legacy" and not set(variants) <= set(POSITION_FOCUS_VARIANTS):
        raise ValueError("Position-focused training is restricted to A, B, K and L variants")
    output = Path(args.output).resolve()
    output.mkdir(parents=True, exist_ok=False)
    manifest = dict(status="running", created_utc=datetime.now(timezone.utc).isoformat(),
                    preset="pilot", pilot_only=True, config=asdict(config), seeds=seeds,
                    reward_profile=args.reward_profile,
                    reward_settings=REWARD_PROFILES[args.reward_profile]._asdict(),
                    variants=variants, horizon=400, dt=DT,
                    eval_every=args.checkpoint_every, evaluation_enabled=False,
                    reference_seeds={"train": 12345, "held_out": 54321, "faster": 65432},
                    initial_condition_seed=900000, python=platform.python_version(),
                    backend=jax.default_backend(), seed_batching="independent-vmap",
                    case_settings={},
                    dependencies={name: importlib.metadata.version(name) for name in
                                  ["jax", "jaxlib", "flax", "optax", "numpy", "matplotlib"]},
                    actual_derivative_mode="model-left-limit-zero-order-held-previous-command",
                    notes=["Same pilot task, networks and PPO; longer fresh learning-rate schedule.",
                           "Seeds are trained independently in a batch; no cross-seed statistics or gradients.",
                           "Training reward only. No held-out evaluations performed."])
    sources = [Path(__file__), Path("envs/quadrotor_smooth.py"), Path("envs/quadrotor_thrust_rate.py"),
               Path("envs/quadrotor_double_thrust.py"),
               Path("envs/quadrotor_desired_thrust.py"),
               Path("rl/quadrotor_ppo.py"), Path("experiments/quadrotor_comparison.py"),
               Path("experiments/plot_reward_trends.py"), Path("requirements-quadrotor-lock.txt")]
    (output / "source").mkdir()
    manifest["source_sha256"] = {}
    for source in sources:
        content = source.read_bytes()
        (output / "source" / source.name).write_bytes(content)
        manifest["source_sha256"][source.name] = hashlib.sha256(content).hexdigest()
    write_json(output / "manifest.json", manifest)
    started_all = time.perf_counter()
    try:
        trajectories, bank = make_bank(12345, 64, 400)
        manifest["bank_sha256"] = {"train": save_bank(output / "train_references.npz", trajectories, bank)}
        write_json(output / "manifest.json", manifest)
        print(f"Output: {output}\nDevices: {jax.devices()}\nSteps per policy/seed: {args.steps:,}; seeds: {seeds}", flush=True)
        for variant in variants:
            flags = VARIANTS[variant]
            env = make_environment(bank, variant, reward_profile=args.reward_profile)
            manifest["case_settings"][variant] = dict(
                observation_size=env.obs_size,
                feature_definition=getattr(env, "feature_definition", "standard"),
                action_definition=getattr(env, "action_definition", "four-normalized-rotor-commands"),
                reward_profile=env.reward_profile, reward_settings=env.reward_settings._asdict(),
                settings=getattr(env, "experiment_metadata", {}))
            write_json(output / "manifest.json", manifest)
            _, runners, update = independent_runners(env, config, seeds)
            histories = [[] for _ in seeds]
            paths = [output / variant / f"seed_{seed}" for seed in seeds]
            for path in paths:
                path.mkdir(parents=True)
            started = time.perf_counter()
            for index in range(1, config.updates + 1):
                runners, metrics = jax.block_until_ready(update(runners))
                metrics = {name: np.asarray(value) for name, value in metrics.items()}
                if not all(np.isfinite(value).all() for value in metrics.values()):
                    raise FloatingPointError(f"Non-finite metrics at {variant} update {index}")
                steps = index * config.batch_size
                elapsed = time.perf_counter() - started
                for lane in range(len(seeds)):
                    histories[lane].append(dict(steps=steps, wall_seconds=elapsed,
                        **{name: float(value[lane]) for name, value in metrics.items()}))
                if index % args.checkpoint_every == 0 or index == config.updates:
                    if not finite_parameters(runners.train_state.params):
                        raise FloatingPointError(f"Non-finite parameters at {variant} update {index}")
                    for lane, (seed, path) in enumerate(zip(seeds, paths)):
                        runner = jax.tree_util.tree_map(lambda value: value[lane], runners)
                        write_csv(path / "training.csv", histories[lane])
                        checkpoint = serialization.to_bytes(runner.train_state.params)
                        (path / "policy.msgpack").write_bytes(checkpoint)
                        (path / f"policy_step_{steps:012d}.msgpack").write_bytes(checkpoint)
                        write_json(path / "config.json", dict(variant=variant, seed=seed, steps=steps,
                                   observation_size=env.obs_size, observation_features=flags._asdict(),
                                   initial_action_mean=getattr(env, "initial_action_mean", None),
                                   reward_profile=env.reward_profile, reward_settings=env.reward_settings._asdict(),
                                   environment_settings=getattr(env, "experiment_metadata", {}),
                                   feature_definition=getattr(env, "feature_definition", "standard"),
                                   action_definition=getattr(env, "action_definition", "four-normalized-rotor-commands"),
                                   actual_derivative_mode=manifest["actual_derivative_mode"], ppo=asdict(config)))
                        # Save full state at checkpoints so a failed/interrupted
                        # long run retains optimizer, environment and RNG state.
                        (path / "training_state.msgpack").write_bytes(serialization.to_bytes(runner))
                    mean_reward = float(metrics["step_reward"].mean())
                    print(f"{variant}: {steps:,}/{args.steps:,} per seed, reward={mean_reward:.4f}, "
                          f"seed rewards={metrics['step_reward'].round(4).tolist()}, elapsed={elapsed:.1f}s", flush=True)
                    manifest["progress"] = dict(variant=variant, steps_per_seed=steps,
                                                total_wall_seconds=time.perf_counter() - started_all)
                    write_json(output / "manifest.json", manifest)
            jax.clear_caches()
            # Keep the reward graph reviewable as each policy finishes.
            report(output)
        manifest.update(status="complete", completed_utc=datetime.now(timezone.utc).isoformat(),
                        wall_seconds=time.perf_counter() - started_all)
        write_json(output / "manifest.json", manifest)
        report(output)
    except BaseException as error:
        manifest.update(status="interrupted" if isinstance(error, KeyboardInterrupt) else "failed",
                        error=f"{type(error).__name__}: {error}")
        write_json(output / "manifest.json", manifest)
        raise


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--steps", type=int, default=2097152)
    parser.add_argument("--seeds", nargs="+", type=int, default=[0, 1, 2])
    parser.add_argument("--variants", nargs="+", choices=VARIANTS, default=list(DEFAULT_VARIANTS))
    parser.add_argument("--reward-profile", choices=REWARD_PROFILES, default="legacy")
    parser.add_argument("--checkpoint-every", type=int, default=64)
    parser.add_argument("--output", default="runs/longer_training_reward")
    run(parser.parse_args())
