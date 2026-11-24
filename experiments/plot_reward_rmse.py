"""Compare reward and position RMSE from paired training logs, without new runs."""

import argparse
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import shutil

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
import numpy as np

from experiments.quadrotor_comparison import (
    COLORS, LABELS, read_csv, write_csv, write_json, write_policy_dimensions,
)


DEFAULT_VARIANTS = ["A_full", "B_symmetry", "K_full_desired_thrust_extension",
                    "L_symmetry_desired_thrust_extension"]


def digest(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def render(output, manifest, curves, final):
    variants = manifest["variants"]
    fig, axes = plt.subplots(1, 2, figsize=(14, 6.5))
    fig.subplots_adjust(left=.07, right=.98, bottom=.28, top=.82, wspace=.25)
    for ax in axes:
        for variant in variants:
            data, summary = curves[variant], final[variant]
            ax.plot(data["rmse"], data["reward"], color=COLORS[variant], linewidth=1.8, alpha=.8)
            ax.scatter(data["rmse"][0], data["reward"][0], facecolor="white", edgecolor=COLORS[variant],
                       s=40, linewidth=1.5, zorder=5)
            # Pooled RMSE and pooled mean reward both lie inside the seed range.
            x, y = summary["position_rmse_m"], summary["step_reward"]
            ax.errorbar(x, y, xerr=[[x - summary["seed_rmse_min_m"]], [summary["seed_rmse_max_m"] - x]],
                        yerr=[[y - summary["seed_reward_min"]], [summary["seed_reward_max"] - y]],
                        color=COLORS[variant], capsize=3, linewidth=1.3, alpha=.8, zorder=6)
            ax.scatter(x, y, color=COLORS[variant], edgecolor="white", linewidth=.7,
                       marker="*", s=210, zorder=7)
        ax.axhline(0, color="#64748b", linestyle=":", linewidth=1)
        ax.set(xlabel="Position tracking RMSE [m] (lower is better)",
               ylabel="Mean training reward per step (higher is better)")
        ax.grid(alpha=.2)
    axes[0].set_title("Full training range")
    axes[0].set_xlim(left=0)
    axes[0].set_ylim(top=.025)
    focused = [final[v] for v in variants if v.startswith(("B_", "L_"))]
    if focused:
        lo = min(row["seed_rmse_min_m"] for row in focused)
        hi = max(row["seed_rmse_max_m"] for row in focused)
        margin = max(.02, .3 * (hi - lo))
        bottom = min(row["seed_reward_min"] for row in focused) * 1.8
        axes[1].set(title="Zoom around final B/L results", xlim=(max(0., lo - margin), hi + margin),
                    ylim=(bottom, max(.001, -.05 * bottom)))
    for variant in variants:
        summary = final[variant]
        if variant.startswith(("A_", "K_")):
            axes[0].annotate(variant[0], (summary["position_rmse_m"], summary["step_reward"]),
                             xytext=(9, 3), textcoords="offset points", color=COLORS[variant], fontsize=11)
        if variant.startswith(("B_", "L_")):
            offset = (9, 9) if variant.startswith("B_") else (9, -20)
            axes[1].annotate(f"{variant[0]}: {summary['position_rmse_m'] * 100:.1f} cm",
                             (summary["position_rmse_m"], summary["step_reward"]),
                             xytext=offset, textcoords="offset points", color=COLORS[variant], fontsize=11,
                             bbox=dict(facecolor="white", edgecolor="none", alpha=.85, pad=2))
    handles = [Line2D([], [], color=COLORS[variant], linewidth=2, label=LABELS[variant]) for variant in variants]
    handles += [Line2D([], [], marker="o", markerfacecolor="white", color="#475569", linestyle="none",
                       label="Start of smoothed history"),
                Line2D([], [], marker="*", color="#475569", markersize=11, linestyle="none",
                       label=f"Final {manifest['tail_steps']:,} steps; whiskers = seed range")]
    fig.legend(handles=handles, loc="lower center", bbox_to_anchor=(.5, .005), ncol=2, fontsize=9,
               title="Legend dimensions: obs = neural-network inputs, act = action channels", title_fontsize=9)
    fig.suptitle(f"Training reward versus position tracking RMSE | {manifest['reward_profile']}\n"
                 f"Same training batches; {len(manifest['seeds'])} seeds; "
                 f"lines follow training with {manifest['window_updates']}-update pooling", fontsize=14)
    fig.savefig(output / "reward_vs_position_rmse.png", dpi=170)
    plt.close(fig)

    fig, axes = plt.subplots(2, 1, figsize=(12, 8), sharex=True)
    fig.subplots_adjust(left=.09, right=.98, bottom=.23, top=.87, hspace=.15)
    for variant in variants:
        data = curves[variant]
        x = data["steps"] / 1e6
        axes[0].plot(x, data["reward"], color=COLORS[variant], linewidth=2)
        axes[0].fill_between(x, data["seed_reward_min"], data["seed_reward_max"], color=COLORS[variant], alpha=.12)
        axes[1].plot(x, data["rmse"], color=COLORS[variant], linewidth=2)
        axes[1].fill_between(x, data["seed_rmse_min"], data["seed_rmse_max"], color=COLORS[variant], alpha=.12)
    axes[0].set_ylabel("Mean training reward per step")
    axes[0].axhline(0, color="#64748b", linestyle=":", linewidth=1)
    axes[1].set(ylabel="Position tracking RMSE [m]", xlabel="Environment steps per seed [millions]", ylim=(0, None))
    for ax in axes:
        ax.grid(alpha=.2)
    fig.legend(handles=handles[:len(variants)], loc="lower center", bbox_to_anchor=(.5, .01),
               ncol=2, fontsize=9)
    fig.suptitle(f"Reward and position RMSE over training | {manifest['reward_profile']}\n"
                 f"{manifest['window_updates']}-update pooling; shading = range across {len(manifest['seeds'])} seeds",
                 fontsize=14)
    fig.savefig(output / "reward_and_rmse_vs_steps.png", dpi=170)
    plt.close(fig)


def run(args):
    training = args.run.resolve()
    original = json.loads((training / "manifest.json").read_text(encoding="utf-8"))
    if original["status"] != "complete":
        raise ValueError("Choose a completed training run")
    variants = args.variants
    if len(set(variants)) != len(variants) or not set(variants) <= set(original["variants"]):
        raise ValueError("Choose distinct variants present in the saved training run")
    if args.window < 1 or args.tail < 1:
        raise ValueError("Pooling windows must be positive")
    output = args.output.resolve()
    output.mkdir(parents=True, exist_ok=False)
    batch = original["config"]["num_envs"] * original["config"]["rollout_steps"]
    manifest = dict(status="running", created_utc=datetime.now(timezone.utc).isoformat(),
        training_run=str(training), training_manifest_sha256=digest(training / "manifest.json"),
        variants=variants, seeds=original["seeds"], batch_steps=batch, window_updates=args.window,
        reward_profile=original.get("reward_profile", "legacy"),
        notes=["Both metrics use the same training batches, not held-out trajectories.",
               "RMSE is sqrt(mean of squared per-batch RMSE), pooling equal-sized batches and seeds.",
               "Training includes failures and resets. No fixed-length test error is mixed into these graphs.",
               "Reward includes position, velocity, attitude, angular-rate, rotor-command and failure terms.",
               "No additional training or simulation; raw logs are copied and hashed."])
    shutil.copy2(training / "manifest.json", output / "training_manifest.json")
    (output / "source").mkdir()
    manifest["source_sha256"] = {}
    for source in [Path(__file__), Path("experiments/quadrotor_comparison.py"),
                   Path("envs/quadrotor_smooth.py"), Path("rl/quadrotor_ppo.py"),
                   Path("requirements-quadrotor-lock.txt")]:
        shutil.copy2(source, output / "source" / source.name)
        manifest["source_sha256"][source.name] = digest(source)
    manifest["training_csv_sha256"] = {}
    curves, final, summary_rows, per_seed_rows, curve_rows = {}, {}, [], [], []
    common_steps = None
    for variant in variants:
        rewards, rmses, failures, ended = [], [], [], []
        for seed in manifest["seeds"]:
            relative = Path(variant) / f"seed_{seed}" / "training.csv"
            rows = read_csv(training / relative)
            numeric = np.array([[float(value) for value in row.values()] for row in rows])
            if not np.isfinite(numeric).all():
                raise ValueError("Non-finite training metrics")
            steps = np.array([int(row["steps"]) for row in rows])
            if len(steps) < args.window or steps[-1] != original["config"]["total_steps"]:
                raise ValueError("Incomplete training history or pooling window too long")
            np.testing.assert_array_equal(np.diff(steps), np.full(len(steps) - 1, batch))
            if common_steps is None: common_steps = steps
            np.testing.assert_array_equal(steps, common_steps)
            rewards.append([float(row["step_reward"]) for row in rows])
            rmses.append([float(row["rollout_position_rmse"]) for row in rows])
            failures.append([float(row["failures"]) for row in rows])
            ended.append([float(row["completed_episodes"]) for row in rows])
            (output / relative).parent.mkdir(parents=True, exist_ok=True)
            shutil.copy2(training / relative, output / relative)
            manifest["training_csv_sha256"][str(relative)] = digest(training / relative)
            assert digest(output / relative) == digest(training / relative)
        rewards, mse = np.asarray(rewards), np.asarray(rmses) ** 2
        if np.any(np.asarray(rmses) < 0): raise ValueError("Negative RMSE")
        kernel = np.ones(args.window) / args.window
        reward_pool = np.stack([np.convolve(values, kernel, mode="valid") for values in rewards])
        mse_pool = np.stack([np.convolve(values, kernel, mode="valid") for values in mse])
        seed_rmse = np.sqrt(mse_pool)
        data = dict(steps=steps[args.window - 1:], reward=reward_pool.mean(axis=0),
                    rmse=np.sqrt(mse_pool.mean(axis=0)), seed_reward_min=reward_pool.min(axis=0),
                    seed_reward_max=reward_pool.max(axis=0), seed_rmse_min=seed_rmse.min(axis=0),
                    seed_rmse_max=seed_rmse.max(axis=0))
        curves[variant] = data
        tail = min(args.tail, len(steps))
        tail_reward = rewards[:, -tail:].mean(axis=1)
        tail_rmse = np.sqrt(mse[:, -tail:].mean(axis=1))
        summary = dict(variant=variant, averaging_steps=tail * batch,
            step_reward=float(tail_reward.mean()), position_rmse_m=float(np.sqrt(mse[:, -tail:].mean())),
            seed_reward_min=float(tail_reward.min()), seed_reward_max=float(tail_reward.max()),
            seed_rmse_min_m=float(tail_rmse.min()), seed_rmse_max_m=float(tail_rmse.max()),
            failure_fraction_of_ended_episodes=float(np.asarray(failures)[:, -tail:].sum()
                / max(1., np.asarray(ended)[:, -tail:].sum())))
        final[variant] = summary
        summary_rows.append(summary)
        per_seed_rows.extend(dict(variant=variant, seed=seed, step_reward=float(r), position_rmse_m=float(e))
                             for seed, r, e in zip(manifest["seeds"], tail_reward, tail_rmse))
        curve_rows.extend(dict(variant=variant, **{name: int(values[i]) if name == "steps" else float(values[i])
                                                  for name, values in data.items()})
                          for i in range(len(data["steps"])))
    manifest["tail_steps"] = tail * batch
    write_csv(output / "final_window_summary.csv", summary_rows)
    write_csv(output / "per_seed_final.csv", per_seed_rows)
    write_csv(output / "paired_training_metrics.csv", curve_rows)
    write_policy_dimensions(output, variants)
    render(output, manifest, curves, final)
    manifest["status"] = "complete"
    write_json(output / "manifest.json", manifest)
    report = ["# Training reward versus position RMSE: A, B, K, L", "",
              "Both metrics come from the same saved training transitions, including failures and resets.", "",
              "![Reward versus RMSE](reward_vs_position_rmse.png)", "",
              "![Learning curves](reward_and_rmse_vs_steps.png)", "",
              "Curves pool 16 updates (65,536 steps per seed). Stars pool the final 64 updates "
              "(262,144 steps per seed). RMSE pools squared errors before taking the square root; "
              "whiskers/shading show the range across seeds. These are training metrics, not full-length evaluation RMSE.", "",
              "| Policy | Final-window mean reward/step | Pooled position RMSE [cm] |",
              "|---|---:|---:|"]
    for row in summary_rows:
        report.append(f"| {LABELS[row['variant']]} | {row['step_reward']:.5f} | {row['position_rmse_m'] * 100:.2f} |")
    report.extend(["", "## Reproduce", "", "From the repository root, use a fresh output directory:", "",
                   "```powershell", "..\\.venv\\Scripts\\python.exe -m experiments.plot_reward_rmse "
                   f"--run '{training}' --output runs/reward_rmse_abkl_repeat "
                   f"--window {args.window} --tail {args.tail}", "```", "",
                   "Copied raw training CSVs, paired plotted data, summaries, source and hashes are retained here."])
    (output / "REPORT.md").write_text("\n".join(report) + "\n", encoding="utf-8")
    print(json.dumps(summary_rows, indent=2))
    print(f"Saved figures: {output}", flush=True)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run", type=Path, default=Path("runs/longer_training_reward_all_12"))
    parser.add_argument("--output", type=Path, default=Path("runs/reward_rmse_abkl"))
    parser.add_argument("--variants", nargs="+", default=DEFAULT_VARIANTS)
    parser.add_argument("--window", type=int, default=16)
    parser.add_argument("--tail", type=int, default=64)
    run(parser.parse_args())
