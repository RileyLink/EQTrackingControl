"""Evaluate PPO and REPPO on matched held-out B′ and L trials."""

import argparse
from collections import defaultdict
from pathlib import Path

import jax
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.ticker import EngFormatter
import numpy as np

from envs.quadrotor_smooth import make_environment
from experiments.policy_structure.rollouts import simulate
from experiments.quadrotor_comparison import load_policy, read_csv, write_csv
from experiments.quadrotor_reppo import MATCHED_CONFIG, MATCHED_TASK, check_banks, load_actor
from experiments.quadrotor_scratch import load_bank, read_json, save_json
from experiments.quadrotor_scratch_evaluate import COLORS, LABELS, evaluation_starts, trial_rows
from experiments.quadrotor_thrust_preview import L
from rl.quadrotor_ppo import command


TRIAL_METRICS = {
    "position_rmse_m": "position_rmse_m", "attitude_rmse_rad": "attitude_rmse_rad",
    "yaw_rmse_deg": "yaw_rmse_deg", "yaw_rate_rmse_radps": "yaw_rate_rmse_radps",
    "recovery_rate": "recovered", "yaw_recovery_rate": "yaw_recovered", "failure_rate": "failed",
    "survival_s": "survival_s", "rotor_saturation_fraction": "rotor_saturation_fraction",
}


def matched_manifests(ppo_run, reppo_run):
    ppo = read_json(ppo_run / "manifest.json")
    reppo = read_json(reppo_run / "manifest.json")
    if reppo.get("algorithm") != "REPPO" or reppo["status"] != "trained":
        raise ValueError("Finish the REPPO training run before comparing checkpoints")
    for key in (*MATCHED_TASK, "reference_sha256"):
        if ppo[key] != reppo[key]:
            raise ValueError(f"The runs do not match on {key}")
    for key in MATCHED_CONFIG:
        if ppo["ppo"][key] != reppo["reppo"][key]:
            raise ValueError(f"The runs do not match on {key}")
    if ppo["ppo"]["gae_lambda"] != reppo["reppo"]["td_lambda"]:
        raise ValueError("The runs have different lambda settings")
    check_banks(ppo_run, ppo)
    check_banks(reppo_run, reppo)
    return reppo


def summarize(trials):
    groups = defaultdict(list)
    for row in trials:
        key = tuple(row[name] for name in ("algorithm", "model", "seed", "steps"))
        selections = [("all", "all"), (row["family"], row["start"])]
        if row["start"] == "nominal":
            selections.append(("all", "nominal"))
        elif row["family"] == "yaw":
            selections.append(("yaw", "practice"))
        for selection in selections:
            groups[(*key, *selection)].append(row)
    summary = []
    for (algorithm, model, seed, steps, family, start), group in sorted(groups.items()):
        values = {name: [row[key] for row in group if row[key] is not None and np.isfinite(row[key])]
                  for name, key in TRIAL_METRICS.items()}
        summary.append(dict(algorithm=algorithm, model=model, seed=seed, steps=steps, family=family, start=start,
                            trials=len(group), **{key: float(np.mean(x)) if x else None for key, x in values.items()}))
    return summary


def paired_deltas(summary):
    keys = ("model", "seed", "steps", "family", "start")
    ppo = {tuple(row[key] for key in keys): row for row in summary if row["algorithm"] == "PPO"}
    for row in summary:
        if row["algorithm"] == "REPPO":
            match = {key: row[key] for key in keys}
            baseline = ppo[tuple(match.values())]
            yield dict(**match, trials=row["trials"], **{
                name: row[name] - baseline[name] if row[name] is not None and baseline[name] is not None else None
                for name in TRIAL_METRICS
            })


def training_summary(run, algorithm, manifest, steps):
    interval = manifest["checkpoint_every"]
    rows = []
    for variant in manifest["variants"]:
        source = read_csv(run / variant / "training.csv")
        for seed in manifest["seeds"]:
            lane = [row for row in source if int(row["seed"]) == seed]
            for step in steps:
                window = [row for row in lane if step - interval < int(row["steps"]) <= step]
                if not window or int(window[-1]["steps"]) != step:
                    raise ValueError(f"Missing training metrics for {algorithm} {variant} seed {seed}, step {step}")
                episodes = sum(float(row["completed_episodes"]) for row in window)
                rows.append(dict(
                    algorithm=algorithm, model=variant, seed=seed, steps=step,
                    mean_step_reward=float(np.mean([float(row["step_reward"]) for row in window])),
                    mean_episode_return=(sum(float(row["episode_return_sum"]) for row in window) / episodes
                                         if episodes else None),
                    elapsed_seconds=float(window[-1]["elapsed_seconds"]),
                ))
    return rows


def plot_comparison(path, summary, training, variants, steps):
    fig, grid = plt.subplots(2, 2, figsize=(11, 7.5), sharex=True, layout="constrained")
    panels = (
        (training, "mean_step_reward", "Training reward per step"),
        ([row for row in summary if (row["family"], row["start"]) == ("all", "nominal")],
         "position_rmse_m", "Held-out position RMSE (m)"),
        ([row for row in summary if (row["family"], row["start"]) == ("yaw", "practice")],
         "yaw_rmse_deg", "Held-out yaw RMSE (deg)"),
        ([row for row in summary if (row["family"], row["start"]) == ("all", "all")],
         "failure_rate", "Held-out failure rate (%)"),
    )
    for axis, (rows, metric, title) in zip(grid.flat, panels):
        for variant in variants:
            for algorithm, style in (("PPO", "--"), ("REPPO", "-")):
                groups = defaultdict(list)
                for row in rows:
                    if row["algorithm"] == algorithm and row["model"] == variant and row[metric] is not None:
                        groups[row["steps"]].append(row[metric])
                values = [groups[step] or [np.nan] for step in steps]
                scale = 100 if metric == "failure_rate" else 1
                mean = scale * np.asarray([np.mean(group) for group in values])
                std = scale * np.asarray([np.std(group) for group in values])
                axis.plot(steps, mean, style, marker="o", ms=3, color=COLORS[variant],
                          label=f"{algorithm} {LABELS[variant]}")
                axis.fill_between(steps, mean - std, mean + std, color=COLORS[variant], alpha=.09)
        axis.set_title(title)
        axis.grid(alpha=.2)
        if metric == "failure_rate":
            axis.set_ylim(0., min(100., max(1., axis.get_ylim()[1])))
    grid[1, 0].xaxis.set_major_formatter(EngFormatter(sep=""))
    grid[0, 0].legend(frameon=False, ncol=2)
    fig.supxlabel("Simulator steps per seed")
    fig.savefig(path, dpi=155)
    plt.close(fig)


def compare(reppo_run, output, *, ppo_run=None, selected_steps=None):
    reppo_run, output = reppo_run.resolve(), output.resolve()
    if output.exists():
        raise FileExistsError(f"Comparison output already exists at {output}")
    ppo_run = (ppo_run or Path(read_json(reppo_run / "manifest.json")["ppo_run"])).resolve()
    manifest = matched_manifests(ppo_run, reppo_run)
    total = manifest["reppo"]["total_steps"]
    saved_steps = [*range(manifest["checkpoint_every"], total, manifest["checkpoint_every"]), total]
    steps = sorted(set(selected_steps)) if selected_steps else saved_steps
    if not set(steps).issubset(saved_steps):
        raise ValueError(f"Select saved checkpoint steps: {saved_steps}")
    bank, families = load_bank(reppo_run / "test_references.npz")
    environments = {variant: make_environment(bank, variant, dt=manifest["dt"],
                                              reward_profile=manifest["reward_profile"])
                    for variant in manifest["variants"]}
    starts, cases = evaluation_starts(environments[L], families, manifest["evaluation_seed"])
    trials, training = [], []
    for algorithm, run, load in (("PPO", ppo_run, load_policy), ("REPPO", reppo_run, load_actor)):
        training.extend(training_summary(run, algorithm, manifest, steps))
        for variant, env in environments.items():
            rollout = None
            for seed in manifest["seeds"]:
                for step in steps:
                    folder = run / variant / f"seed_{seed}" / f"step_{step:09d}"
                    model, params, metadata = load(folder)
                    if (metadata["variant"] != variant or metadata["seed"] != seed or metadata["steps"] != step
                            or metadata["observation_size"] != env.obs_size
                            or not np.array_equal(metadata["observation_scale"], np.asarray(env.scale))
                            or tuple(metadata["initial_action_mean"]) != tuple(env.initial_action_mean)
                            or metadata["reward_profile"] != manifest["reward_profile"]):
                        raise ValueError(f"Checkpoint settings differ at {folder}")
                    if rollout is None:
                        def run_rollout(params, initial, model=model, env=env):
                            def controller(state):
                                return command(model.apply(params, env.observe(state))[0])
                            return simulate(env, controller, initial, jit=False)
                        rollout = jax.jit(run_rollout)
                    frames = rollout(params, starts)
                    trials.extend(dict(algorithm=algorithm, **row) for row in
                                  trial_rows(variant, seed, step, env, starts, cases, families, frames))
            print(f"Evaluated {algorithm} {variant} on {len(starts.pos)} held-out starts", flush=True)

    summary = summarize(trials)
    output.mkdir(parents=True, exist_ok=False)
    write_csv(output / "rollout_trials.csv", trials)
    write_csv(output / "rollout_summary.csv", summary)
    write_csv(output / "reppo_minus_ppo.csv", list(paired_deltas(summary)))
    write_csv(output / "training_summary.csv", training)
    plot_comparison(output / "comparison.png", summary, training, manifest["variants"], steps)
    save_json(output / "manifest.json", dict(
        ppo_run=str(ppo_run), reppo_run=str(reppo_run), steps=steps, seeds=manifest["seeds"],
        variants=manifest["variants"], evaluation_seed=manifest["evaluation_seed"],
        test_reference_sha256=manifest["reference_sha256"]["test_references.npz"],
    ))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("reppo_run", type=Path)
    parser.add_argument("--ppo-run", type=Path, help="Override the PPO run location saved by the REPPO trainer")
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--steps", type=int, nargs="+", help="Evaluate selected saved checkpoint steps")
    args = parser.parse_args()
    compare(args.reppo_run, args.output, ppo_run=args.ppo_run, selected_steps=args.steps)


if __name__ == "__main__":
    main()
