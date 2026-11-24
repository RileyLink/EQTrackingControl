"""Train and plot smooth-reference PPO policies; evaluation is optional.

Run from the repository root: python -m experiments.quadrotor_comparison --help
"""

import argparse
import csv
from dataclasses import asdict, replace
from datetime import datetime, timezone
import hashlib
import importlib.metadata
import json
from pathlib import Path
import platform
import subprocess
import time

from flax import serialization
import jax
import jax.numpy as jnp
import numpy as np

from envs.quadrotor_smooth import (
    DEFAULT_VARIANTS, DT, VARIANTS, REWARD_PROFILES, POSITION_FOCUS_VARIANTS, WRENCH_VARIANTS,
    SmoothQuadrotor, make_bank, make_environment, transform_bank,
)
from rl.quadrotor_ppo import (
    EVALUATION_COLUMNS, PPOConfig, QuadActorCritic, make_evaluator, make_learner,
)


PRESETS = {
    "smoke": dict(config=PPOConfig(total_steps=2048, num_envs=4, rollout_steps=64,
                                   epochs=2, layers=2, width=32),
                  train_count=8, eval_count=4, horizon=80, eval_every=4, seeds=[0]),
    "pilot": dict(config=PPOConfig(), train_count=64, eval_count=12,
                  horizon=400, eval_every=16, seeds=[0]),
    "research": dict(config=PPOConfig(total_steps=999981056, num_envs=64,
                                      rollout_steps=512, epochs=5, layers=6, width=128),
                     train_count=512, eval_count=20, horizon=1000,
                     eval_every=500, seeds=list(range(10))),
}
POLICY_NAMES = {
    "A_full": "A: full state",
    "B_symmetry": "B: symmetry",
    "C_full_derivatives": "C: full + derivatives",
    "D_symmetry_derivatives": "D: symmetry + derivatives",
    "E_full_actual_derivatives": "E: full + desired/actual derivatives",
    "F_symmetry_actual_derivatives": "F: symmetry + desired/actual derivatives",
    "G_symmetry_thrust_rates": "G: symmetry + Fdot inputs",
    "H_dynamic_thrust_extension": "H: dynamic thrust extension",
    "I_full_double_thrust_extension": "I: full + double thrust extension",
    "J_symmetry_double_thrust_extension": "J: symmetry + double thrust extension",
    "K_full_desired_thrust_extension": "K: full + desired thrust extension",
    "L_symmetry_desired_thrust_extension": "L: symmetry + desired thrust extension",
    "B_se23": "B: SE_2(3) errors",
    "L_se23_desired_thrust_extension": "L: SE_2(3) errors + desired thrust extension",
    "B_prime_se23_wrench": "B′: SE_2(3) errors + desired wrench",
}
OBSERVATION_SIZES = {
    "A_full": 40, "B_symmetry": 31,
    "C_full_derivatives": 54, "D_symmetry_derivatives": 45,
    "E_full_actual_derivatives": 69, "F_symmetry_actual_derivatives": 60,
    "G_symmetry_thrust_rates": 33, "H_dynamic_thrust_extension": 34,
    "I_full_double_thrust_extension": 44, "J_symmetry_double_thrust_extension": 35,
    "K_full_desired_thrust_extension": 42, "L_symmetry_desired_thrust_extension": 33,
    "B_se23": 25, "L_se23_desired_thrust_extension": 27,
    "B_prime_se23_wrench": 25,
}
ACTION_SIZE = 4
LABELS = {variant: f"{name} (obs {OBSERVATION_SIZES[variant]}, act {ACTION_SIZE})"
          for variant, name in POLICY_NAMES.items()}
COLORS = dict(zip(DEFAULT_VARIANTS, ["#64748b", "#2563eb", "#d97706", "#059669", "#9333ea", "#dc2626", "#8c564b", "#06b6d4", "#111827", "#ec4899", "#0e7490", "#65a30d"]))
COLORS.update(B_se23="#1e3a8a", L_se23_desired_thrust_extension="#365314", B_prime_se23_wrench="#2563eb")


def write_json(path, value):
    path.write_text(json.dumps(value, indent=2, allow_nan=False) + "\n", encoding="utf-8")


def write_csv(path, rows):
    if not rows:
        return
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def write_policy_dimensions(output, variants):
    rows = []
    for variant in variants:
        action = ("Fdot(1) + body torque(3)" if variant.startswith("H_") else
                  "Fddot(1) + body torque(3)" if variant.startswith(("I_", "J_")) else
                  "f(1) + body torque(3)" if variant in WRENCH_VARIANTS else
                  "rotor commands(4)")
        rows.append(dict(variant=variant, policy=POLICY_NAMES[variant],
                         observation_scalars=OBSERVATION_SIZES[variant], action_scalars=ACTION_SIZE,
                         action_definition=action, rotation_matrix_entries=9))
    write_csv(Path(output) / "policy_dimensions.csv", rows)


def save_bank(path, trajectories, bank):
    arrays = {"trajectory_" + name: np.asarray(value)
              for name, value in zip(trajectories._fields, trajectories)}
    arrays.update({"reference_" + name: np.asarray(value)
                   for name, value in zip(bank._fields, bank)})
    np.savez_compressed(path, **arrays)
    return hashlib.sha256(path.read_bytes()).hexdigest()


def eval_rows(evaluator, params, variant, seed, steps, case):
    per_trajectory, frames = jax.block_until_ready(evaluator(params))
    data = np.asarray(per_trajectory)
    if not np.isfinite(data).all():
        raise FloatingPointError(f"Non-finite evaluation for {variant}, seed {seed}")
    common = dict(variant=variant, seed=seed, steps=steps, case=case)
    aggregate = {**common, **dict(zip(EVALUATION_COLUMNS, data.mean(axis=0).tolist()))}
    individual = [{**common, "trajectory": i, **dict(zip(EVALUATION_COLUMNS, row.tolist()))}
                  for i, row in enumerate(data)]
    return aggregate, individual, frames


def finite_parameters(params):
    return all(np.isfinite(np.asarray(x)).all() for x in jax.tree_util.tree_leaves(params))


def run(args):
    preset = PRESETS[args.preset]
    config = preset["config"]
    overrides = {name: getattr(args, name) for name in
                 ("total_steps", "num_envs", "rollout_steps", "layers", "width")
                 if getattr(args, name) is not None}
    config = replace(config, **overrides)
    config.validate()
    seeds = args.seeds if args.seeds is not None else preset["seeds"]
    if len(seeds) != len(set(seeds)) or any(seed < 0 for seed in seeds):
        raise ValueError("Seeds must be distinct nonnegative integers")
    horizon = preset["horizon"] if args.horizon is None else args.horizon
    eval_every = preset["eval_every"] if args.eval_every is None else args.eval_every
    if horizon < 1 or eval_every < 1:
        raise ValueError("Horizon and evaluation interval must be positive")
    variants = args.variants
    if args.reward_profile != "legacy" and not set(variants) <= set(POSITION_FOCUS_VARIANTS):
        raise ValueError("Position-focused training is restricted to A, B, K and L variants")
    if len(variants) != len(set(variants)):
        raise ValueError("Variants must be distinct")
    output = Path(args.output or ("runs/quadrotor_" + datetime.now().strftime("%Y%m%d_%H%M%S"))).resolve()
    output.mkdir(parents=True, exist_ok=False)
    try:
        revision = subprocess.check_output(["git", "rev-parse", "HEAD"], text=True).strip()
        dirty = bool(subprocess.check_output(["git", "status", "--porcelain"], text=True).strip())
    except (OSError, subprocess.CalledProcessError):
        revision, dirty = None, True
    manifest = dict(status="running", created_utc=datetime.now(timezone.utc).isoformat(),
                    preset=args.preset, pilot_only=args.preset != "research",
                    config=asdict(config), seeds=seeds, variants=variants,
                    reward_profile=args.reward_profile, reward_settings=REWARD_PROFILES[args.reward_profile]._asdict(),
                    horizon=horizon, dt=DT, eval_every=eval_every, evaluation_enabled=args.evaluate,
                    reference_seeds={"train": 12345, "held_out": 54321, "faster": 65432},
                    initial_condition_seed=900000, git_revision=revision, git_dirty=dirty,
                    python=platform.python_version(), backend=jax.default_backend(),
                    dependencies={p: importlib.metadata.version(p) for p in
                                  ["jax", "jaxlib", "flax", "optax", "numpy", "matplotlib"]},
                    observation_sizes=OBSERVATION_SIZES,
                    actual_derivative_mode="model-left-limit-zero-order-held-previous-command",
                    notes=["Smooth-reference task; not a reproduction of random-control training.",
                           "Reference generator phase/parameters are hidden from all policies.",
                           "Finite derivative observations are not asserted to define a Markov quotient.",
                           "Fixed isotropic observation scales; separate actor and critic MLPs.",
                           "Actual features use only current state and previous command, with zero thrust derivatives.",
                           "Actual heading uses the flatness convention, masked outside the upright chart.",
                           "Time limits bootstrap terminal values; failures do not bootstrap.",
                           "Evaluation errors stop at first failure; interpret with survival and failure rate.",
                           "Research budget rounded down to complete PPO batches."])
    # Fingerprint the implementation as the base Git revision alone does not include local edits.
    sources = [Path(__file__), Path("envs/quadrotor_smooth.py"), Path("envs/quadrotor_thrust_rate.py"),
               Path("envs/quadrotor_double_thrust.py"), Path("envs/quadrotor_desired_thrust.py"),
               Path("rl/quadrotor_ppo.py")]
    manifest["source_sha256"] = {p.name: hashlib.sha256(p.read_bytes()).hexdigest() for p in sources}
    write_json(output / "manifest.json", manifest)
    try:
        print(f"Output: {output}\nBackend: {jax.devices()}\nSteps per policy/seed: {config.total_steps:,}", flush=True)
        train_tr, train_bank = make_bank(12345, preset["train_count"], horizon)
        manifest["bank_sha256"] = {
            "train": save_bank(output / "train_references.npz", train_tr, train_bank),
        }
        if args.evaluate:
            eval_tr, eval_bank = make_bank(54321, preset["eval_count"], horizon)
            faster_tr, faster_bank = make_bank(65432, preset["eval_count"], horizon, frequency_scale=1.3)
            manifest["bank_sha256"].update({
                "held_out": save_bank(output / "held_out_references.npz", eval_tr, eval_bank),
                "faster": save_bank(output / "faster_references.npz", faster_tr, faster_bank),
            })
            shifted_bank = transform_bank(eval_bank, 1.2, jnp.array([5., -4., 2.]))
        write_json(output / "manifest.json", manifest)
        evaluations, per_trajectory_rows = [], []
        start_all = time.perf_counter()
        for variant in variants:
            features = VARIANTS[variant]._asdict()
            env = make_environment(train_bank, variant, reward_profile=args.reward_profile)
            for seed in seeds:
                destination = output / variant / f"seed_{seed}"
                destination.mkdir(parents=True)
                model, runner, update = make_learner(env, config, seed)
                if args.evaluate:
                    evaluator = make_evaluator(make_environment(eval_bank, variant, reward_profile=args.reward_profile), model)
                    evaluations.append(eval_rows(evaluator, runner.train_state.params, variant, seed, 0, "held_out")[0])
                    write_csv(output / "evaluation.csv", evaluations)
                history = []
                started = time.perf_counter()
                for update_index in range(1, config.updates + 1):
                    runner, metrics = jax.block_until_ready(update(runner))
                    row = {"steps": update_index * config.batch_size,
                           "wall_seconds": time.perf_counter() - started,
                           **{k: float(np.asarray(v)) for k, v in metrics.items()}}
                    if not all(np.isfinite(v) for v in row.values()):
                        raise FloatingPointError(f"Non-finite training metrics: {variant}, seed {seed}")
                    history.append(row)
                    if update_index % eval_every == 0 or update_index == config.updates:
                        if not finite_parameters(runner.train_state.params):
                            raise FloatingPointError("Non-finite network parameters")
                        if args.evaluate:
                            aggregate, individuals, frames = eval_rows(
                                evaluator, runner.train_state.params, variant, seed, row["steps"], "held_out")
                            evaluations.append(aggregate)
                            write_csv(output / "evaluation.csv", evaluations)
                        write_csv(destination / "training.csv", history)
                        (destination / "policy.msgpack").write_bytes(serialization.to_bytes(runner.train_state.params))
                        write_json(destination / "config.json", {"variant": variant, "seed": seed,
                                   "steps": row["steps"], "observation_size": env.obs_size,
                                   "observation_features": features,
                                   "initial_action_mean": getattr(env, "initial_action_mean", None),
                                   "reward_profile": env.reward_profile, "reward_settings": env.reward_settings._asdict(),
                                   "environment_settings": getattr(env, "experiment_metadata", {}),
                                   "actual_derivative_mode": manifest["actual_derivative_mode"], "ppo": asdict(config)})
                        print(f"{variant} seed={seed} steps={row['steps']:,} "
                              f"reward/step={row['step_reward']:.3f} "
                              f"elapsed={row['wall_seconds']:.1f}s", flush=True)
                if args.evaluate:
                    per_trajectory_rows.extend(individuals)
                    np.savez_compressed(destination / "rollout.npz", **{k: np.asarray(v) for k, v in frames.items()})
                    for case, bank in [("yaw_translation", shifted_bank), ("faster", faster_bank)]:
                        case_evaluator = make_evaluator(make_environment(bank, variant, reward_profile=args.reward_profile), model)
                        aggregate, individuals, _ = eval_rows(case_evaluator, runner.train_state.params,
                                                              variant, seed, config.total_steps, case)
                        evaluations.append(aggregate)
                        per_trajectory_rows.extend(individuals)
                    write_csv(output / "evaluation.csv", evaluations)
                    write_csv(output / "per_trajectory.csv", per_trajectory_rows)
                # Complete optimizer/environment/RNG snapshot, plus convenient policy-only checkpoint.
                (destination / "training_state.msgpack").write_bytes(serialization.to_bytes(runner))
                jax.clear_caches()
        manifest.update(status="complete", wall_seconds=time.perf_counter() - start_all,
                        completed_utc=datetime.now(timezone.utc).isoformat())
        write_json(output / "manifest.json", manifest)
        report(output)
    except BaseException as exc:
        manifest.update(status="interrupted" if isinstance(exc, KeyboardInterrupt) else "failed",
                        error=f"{type(exc).__name__}: {exc}")
        write_json(output / "manifest.json", manifest)
        raise
    return output


def read_csv(path):
    with path.open(newline="", encoding="utf-8") as handle:
        return list(csv.DictReader(handle))


def evaluation_report(output):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    output = Path(output)
    manifest = json.loads((output / "manifest.json").read_text(encoding="utf-8"))
    evaluations = read_csv(output / "evaluation.csv")
    colors = COLORS
    fig, axes = plt.subplots(2, 2, figsize=(12, 8), constrained_layout=True)
    panels = [("step_reward", "Training reward per step"),
              ("position_rmse_m", "Held-out position RMSE [m]"),
              ("attitude_rmse_rad", "Held-out attitude RMSE [rad]"),
              ("failed", "Held-out failure fraction")]
    for variant in manifest["variants"]:
        for ax, (metric, title) in zip(axes.flat, panels):
            grouped = {}
            if metric == "step_reward":
                rows = []
                for seed in manifest["seeds"]:
                    path = output / variant / f"seed_{seed}" / "training.csv"
                    if path.exists():
                        rows.extend(read_csv(path))
            else:
                rows = [r for r in evaluations if r["variant"] == variant and r["case"] == "held_out"]
            for row in rows:
                grouped.setdefault(int(row["steps"]), []).append(float(row[metric]))
            steps = sorted(grouped)
            if not steps:
                continue
            means = np.array([np.mean(grouped[s]) for s in steps])
            std = np.array([np.std(grouped[s]) for s in steps])
            ax.plot(steps, means, label=LABELS[variant], color=colors[variant])
            if len(manifest["seeds"]) > 1:
                ax.fill_between(steps, means - std, means + std, color=colors[variant], alpha=.15)
            ax.set_title(title)
            ax.set_xlabel("Environment steps per seed")
            ax.grid(alpha=.2)
            ax.ticklabel_format(axis="x", style="sci", scilimits=(0, 0))
    axes[1, 1].set_ylim(-.02, 1.02)
    axes[0, 0].legend(fontsize=9)
    qualifier = "PILOT / pipeline validation - not converged results" if manifest["pilot_only"] else "Research run"
    fig.suptitle(f"Quadrotor: {len(manifest['variants'])} PPO policies\n{qualifier}; {len(manifest['seeds'])} seed(s)", fontsize=14)
    fig.savefig(output / "evaluation.png", dpi=160)
    plt.close(fig)

    final_steps = manifest["config"]["total_steps"]
    summary = []
    for case in ["held_out", "yaw_translation", "faster"]:
        for variant in manifest["variants"]:
            rows = [r for r in evaluations if r["variant"] == variant and r["case"] == case
                    and int(r["steps"]) == final_steps]
            if not rows:
                continue
            row = {"variant": variant, "case": case, "seeds": len(rows)}
            for metric in EVALUATION_COLUMNS:
                values = [float(r[metric]) for r in rows]
                row[metric + "_mean"], row[metric + "_std"] = float(np.mean(values)), float(np.std(values))
            summary.append(row)
    write_csv(output / "summary.csv", summary)
    fig, axes = plt.subplots(1, 2, figsize=(12, 4.5), constrained_layout=True)
    reference_drawn = False
    for variant in manifest["variants"]:
        path = output / variant / f"seed_{manifest['seeds'][0]}" / "rollout.npz"
        if not path.exists():
            continue
        with np.load(path) as data:
            position, reference, active = data["position"][:, 0], data["reference"][:, 0], data["active"][:, 0]
            if not reference_drawn:
                axes[0].plot(reference[:, 0], reference[:, 1], "k--", label="Desired")
                reference_drawn = True
            axes[0].plot(position[active, 0], position[active, 1], color=colors[variant], label=LABELS[variant])
            error = np.linalg.norm(position - reference, axis=-1)
            times = (np.arange(len(active)) + 1) * manifest["dt"]
            axes[1].plot(times[active], error[active], color=colors[variant], label=LABELS[variant])
    axes[0].set(title="Same held-out trajectory and initial state", xlabel="World x [m]", ylabel="World y [m]")
    axes[0].set_aspect("equal", adjustable="datalim")
    axes[0].legend(fontsize=8)
    axes[1].set(title="Position error (curves stop at failure)", xlabel="Time [s]", ylabel="Position error [m]")
    for ax in axes:
        ax.grid(alpha=.2)
    fig.suptitle(qualifier)
    fig.savefig(output / "rollouts.png", dpi=160)
    plt.close(fig)
    markdown = ["# Quadrotor evaluation", "", qualifier, "",
                f"Steps per policy/seed: {final_steps:,}. Seeds: {manifest['seeds']}.", "",
                "Errors are averaged over each trajectory up to first failure, then across trajectories and seeds.",
                "Always interpret RMSE together with failure fraction and survival time. Shading, when present, is seed standard deviation.", "",
                "| Policy | Position RMSE [m] | Attitude RMSE [rad] | Failure fraction | Survival [s] |",
                "|---|---:|---:|---:|---:|"]
    for row in summary:
        if row["case"] == "held_out":
            markdown.append(f"| {LABELS[row['variant']]} | {row['position_rmse_m_mean']:.3f} | "
                            f"{row['attitude_rmse_rad_mean']:.3f} | {row['failed_mean']:.1%} | "
                            f"{row['survival_seconds_mean']:.2f} |")
    markdown.extend(["", "See summary.csv for yaw/translation and faster-reference evaluations.",
                     "", "![Evaluation curves](evaluation.png)", "", "![Example rollouts](rollouts.png)"])
    (output / "EVALUATION.md").write_text("\n".join(markdown) + "\n", encoding="utf-8")


def report(output, include_evaluation=False):
    """Default display is exclusively training reward; evaluation stays opt-in."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    output = Path(output)
    manifest = json.loads((output / "manifest.json").read_text(encoding="utf-8"))
    write_policy_dimensions(output, manifest["variants"])
    fig, ax = plt.subplots(figsize=(11, 6), constrained_layout=True)
    summary = []
    for variant in manifest["variants"]:
        grouped = {}
        final_rewards = []
        for seed in manifest["seeds"]:
            path = output / variant / f"seed_{seed}" / "training.csv"
            if not path.exists():
                continue
            rows = read_csv(path)
            if rows:
                final_rewards.append(float(rows[-1]["step_reward"]))
            for row in rows:
                grouped.setdefault(int(row["steps"]), []).append(float(row["step_reward"]))
        steps = sorted(grouped)
        if not steps:
            continue
        means = np.array([np.mean(grouped[s]) for s in steps])
        std = np.array([np.std(grouped[s]) for s in steps])
        ax.plot(steps, means, label=LABELS[variant], color=COLORS[variant], linewidth=1.8)
        if len(manifest["seeds"]) > 1:
            ax.fill_between(steps, means - std, means + std, color=COLORS[variant], alpha=.13)
        summary.append({"variant": variant, "seeds": len(final_rewards), "steps": steps[-1],
                        "final_update_reward_mean": float(np.mean(final_rewards)),
                        "final_update_reward_std": float(np.std(final_rewards))})
    qualifier = "Pilot - not converged" if manifest["pilot_only"] else "Research run"
    ax.set(title=f"Quadrotor training reward | {len(summary)} policies | {len(manifest['seeds'])} seeds\n"
                 f"{manifest.get('reward_profile', 'legacy')} | {qualifier}",
           xlabel="Environment steps per seed", ylabel="Mean training reward per environment step (higher is better)")
    ax.ticklabel_format(axis="x", style="sci", scilimits=(0, 0))
    ax.grid(alpha=.2)
    if summary:
        ax.legend(fontsize=8.5, loc="lower right", ncol=2,
                  title="NN dimensions: obs = inputs, act = outputs", title_fontsize=9)
    fig.savefig(output / "training_reward.png", dpi=160)
    fig.savefig(output / "comparison.png", dpi=160)
    plt.close(fig)
    write_csv(output / "training_summary.csv", summary)
    markdown = ["# Quadrotor training reward", "", qualifier, "",
                f"Reward profile: {manifest.get('reward_profile', 'legacy')}. Raw reward units are plotted.", "",
                f"Budget per policy/seed: {manifest['config']['total_steps']:,}. Seeds: {manifest['seeds']}.", "",
                "The graph shows mean reward over each PPO rollout batch, averaged across seeds.",
                "Shading is seed standard deviation. No smoothing or held-out evaluation enters this graph.", "",
                "Legend parentheses give neural-network observation/action dimensions. Rotation matrices use nine entries. "
                "See policy_dimensions.csv for the complete dimension table.", "",
                "![Training reward](training_reward.png)", "",
                "Checkpoints and training CSVs are retained. Evaluation tools remain available for later use."]
    (output / "REPORT.md").write_text("\n".join(markdown) + "\n", encoding="utf-8")
    if include_evaluation:
        if not (output / "evaluation.csv").exists():
            raise ValueError("No evaluation data in this run; train with --evaluate to collect it")
        evaluation_report(output)
    print(f"Report: {output / 'REPORT.md'}", flush=True)


def load_policy(directory):
    """Load the policy-only checkpoint for external rollouts."""
    directory = Path(directory)
    config = json.loads((directory / "config.json").read_text(encoding="utf-8"))
    initial = config.get("initial_action_mean")
    model = QuadActorCritic(config["ppo"]["layers"], config["ppo"]["width"],
                            initial_action_mean=None if initial is None else tuple(initial))
    template = model.init(jax.random.PRNGKey(0), jnp.zeros(config["observation_size"]))
    params = serialization.from_bytes(template, (directory / "policy.msgpack").read_bytes())
    return model, params, config


def evaluate_saved(output):
    """Evaluate existing checkpoints later without running any additional PPO."""
    output = Path(output)
    manifest = json.loads((output / "manifest.json").read_text(encoding="utf-8"))
    if manifest["status"] != "complete":
        raise ValueError("Post-training evaluation requires a completed run")
    preset = PRESETS[manifest["preset"]]
    tr, held_out = make_bank(manifest["reference_seeds"]["held_out"], preset["eval_count"],
                             manifest["horizon"], dt=manifest["dt"])
    faster_tr, faster = make_bank(manifest["reference_seeds"]["faster"], preset["eval_count"],
                                  manifest["horizon"], dt=manifest["dt"], frequency_scale=1.3)
    hashes = {"held_out": save_bank(output / "held_out_references.npz", tr, held_out),
              "faster": save_bank(output / "faster_references.npz", faster_tr, faster)}
    shifted = transform_bank(held_out, 1.2, jnp.array([5., -4., 2.]))
    existing = read_csv(output / "evaluation.csv") if (output / "evaluation.csv").exists() else []
    # Retain historical curves and replace only the final checkpoint evaluations.
    final_steps = manifest["config"]["total_steps"]
    rows = [r for r in existing if int(r["steps"]) != final_steps]
    individual_rows = []
    for variant in manifest["variants"]:
        features = VARIANTS[variant]._asdict()
        for seed in manifest["seeds"]:
            path = output / variant / f"seed_{seed}"
            model, params, saved = load_policy(path)
            if saved["steps"] != final_steps:
                raise ValueError("Checkpoint step count does not match the completed run")
            for case, bank in [("held_out", held_out), ("yaw_translation", shifted), ("faster", faster)]:
                evaluator = make_evaluator(make_environment(bank, variant, dt=manifest["dt"],
                                           reward_profile=manifest.get("reward_profile", "legacy")), model,
                                           seed=manifest["initial_condition_seed"])
                aggregate, individuals, frames = eval_rows(evaluator, params, variant, seed, final_steps, case)
                rows.append(aggregate)
                individual_rows.extend(individuals)
                if case == "held_out":
                    np.savez_compressed(path / "rollout.npz", **{k: np.asarray(v) for k, v in frames.items()})
            write_csv(output / "evaluation.csv", rows)
            write_csv(output / "per_trajectory.csv", individual_rows)
            print(f"Evaluated saved {variant} seed={seed}", flush=True)
            jax.clear_caches()
    manifest.setdefault("bank_sha256", {}).update(hashes)
    manifest["post_training_evaluation_utc"] = datetime.now(timezone.utc).isoformat()
    write_json(output / "manifest.json", manifest)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--preset", choices=PRESETS, default="pilot")
    parser.add_argument("--output", help="New output directory; existing results are never overwritten")
    parser.add_argument("--seeds", nargs="+", type=int)
    parser.add_argument("--variants", nargs="+", choices=VARIANTS, default=list(DEFAULT_VARIANTS))
    parser.add_argument("--reward-profile", choices=REWARD_PROFILES, default="legacy")
    for name in ["total-steps", "num-envs", "rollout-steps", "layers", "width", "horizon", "eval-every"]:
        parser.add_argument("--" + name, type=int)
    parser.add_argument("--report-only", type=Path, help="Regenerate plots/tables from an existing run")
    parser.add_argument("--evaluate-only", type=Path, help="Evaluate completed checkpoints later without additional training")
    parser.add_argument("--evaluate", action="store_true", help="Collect held-out evaluation data; default is training only")
    parser.add_argument("--include-evaluation-plots", action="store_true", help="With --report-only, also render saved evaluation data")
    args = parser.parse_args()
    if args.evaluate_only:
        evaluate_saved(args.evaluate_only)
    elif args.report_only:
        report(args.report_only, include_evaluation=args.include_evaluation_plots)
    else:
        run(args)


if __name__ == "__main__":
    main()
