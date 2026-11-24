"""Compare physical tracking accuracy across reward profiles for A/B/K/L."""

import argparse
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import shutil

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from experiments.quadrotor_comparison import COLORS, LABELS, read_csv, write_csv, write_json


VARIANTS = ["A_full", "B_symmetry", "K_full_desired_thrust_extension", "L_symmetry_desired_thrust_extension"]
OLD_EVAL = {"A_full": Path("runs/path_tracking_2m"), "B_symmetry": Path("runs/path_tracking_2m"),
            "K_full_desired_thrust_extension": Path("runs/path_tracking_desired_thrust_2m"),
            "L_symmetry_desired_thrust_extension": Path("runs/path_tracking_desired_thrust_symmetry_2m")}


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def run(args):
    old_run, new_run, new_eval = args.old.resolve(), args.new.resolve(), args.evaluation.resolve()
    old = json.loads((old_run / "manifest.json").read_text())
    new = json.loads((new_run / "manifest.json").read_text())
    evaluation = json.loads((new_eval / "manifest.json").read_text())
    if any(m["status"] != "complete" for m in [old, new, evaluation]):
        raise ValueError("Wait for complete training and evaluation")
    if set(new["variants"]) != set(VARIANTS) or set(evaluation["variants"]) != set(VARIANTS):
        raise ValueError("Compare only A, B, K and L")
    for name in ["config", "seeds", "horizon", "dt", "bank_sha256"]:
        if old[name] != new[name]: raise ValueError(f"Training mismatch: {name}")
    output = args.output.resolve()
    output.mkdir(parents=True, exist_ok=False)
    (output / "source").mkdir()
    source_hashes = {}
    for source in [Path(__file__), Path("experiments/quadrotor_comparison.py"),
                   Path("envs/quadrotor_smooth.py"), Path("rl/quadrotor_ppo.py"), Path("requirements-quadrotor-lock.txt")]:
        shutil.copy2(source, output / "source" / source.name)
        source_hashes[source.name] = sha(source)
    window, tail = 16, 64
    batch = new["config"]["num_envs"] * new["config"]["rollout_steps"]
    curves, summaries, raw_hashes = {}, [], {}
    for variant in VARIANTS:
        row = dict(variant=variant)
        curves[variant] = {}
        for condition, directory in [("old", old_run), ("new", new_run)]:
            rewards, errors, failures, ended = [], [], [], []
            for seed in new["seeds"]:
                relative = Path(variant) / f"seed_{seed}" / "training.csv"
                values = read_csv(directory / relative)
                steps = np.array([int(r["steps"]) for r in values])
                np.testing.assert_array_equal(steps, np.arange(1, len(steps)+1) * batch)
                if steps[-1] != new["config"]["total_steps"]: raise ValueError("Incomplete history")
                errors.append([float(r["rollout_position_rmse"])**2 for r in values])
                rewards.append([float(r["step_reward"]) for r in values])
                failures.append([float(r["failures"]) for r in values])
                ended.append([float(r["completed_episodes"]) for r in values])
                target = output / condition / relative
                target.parent.mkdir(parents=True, exist_ok=True)
                shutil.copy2(directory / relative, target)
                raw_hashes[str(target.relative_to(output))] = sha(target)
            errors = np.asarray(errors)
            kernel = np.ones(window)/window
            pooled = np.stack([np.convolve(e, kernel, mode="valid") for e in errors])
            per_seed = np.sqrt(pooled)
            curves[variant][condition] = dict(steps=steps[window-1:], rmse=np.sqrt(pooled.mean(axis=0)),
                                              lo=per_seed.min(axis=0), hi=per_seed.max(axis=0))
            row[condition + "_training_position_rmse_m"] = float(np.sqrt(errors[:, -tail:].mean()))
            row[condition + "_training_reward_per_step"] = float(np.asarray(rewards)[:, -tail:].mean())
            row[condition + "_training_failure_fraction"] = float(np.asarray(failures)[:, -tail:].sum()
                / max(1., np.asarray(ended)[:, -tail:].sum()))
        before_dir = OLD_EVAL[variant]
        before_manifest = json.loads((before_dir / "manifest.json").read_text())
        for field in ["seeds", "checkpoint_steps", "reference_seed", "trajectory_count", "horizon", "dt",
                      "initial_condition_seed", "policy_mode", "reference_bank_sha256"]:
            if before_manifest[field] != evaluation[field]: raise ValueError(f"Evaluation mismatch: {field}")
        for condition, directory in [("old", before_dir), ("new", new_eval)]:
            item = [r for r in read_csv(directory / "summary.csv") if r["variant"] == variant][0]
            row[condition + "_test_position_rmse_m"] = float(item["position_rmse_m"])
            row[condition + "_test_completed"] = int(item["completed"])
            row[condition + "_test_evaluations"] = int(item["evaluations"])
            row[condition + "_test_attitude_rmse_rad"] = float(item["attitude_rmse_rad"])
        summaries.append(row)
    write_csv(output / "before_after.csv", summaries)
    fig, axes = plt.subplots(2, 2, figsize=(12, 8), sharex=True, sharey=True)
    fig.subplots_adjust(left=.08, right=.98, top=.87, bottom=.15, hspace=.35, wspace=.15)
    for ax, variant, summary in zip(axes.flat, VARIANTS, summaries):
        for condition, color, style, label in [("old", "#94a3b8", "--", "Original reward"),
                                              ("new", COLORS[variant], "-", "Position-focused reward")]:
            data = curves[variant][condition]
            x = data["steps"] / 1e6
            ax.plot(x, data["rmse"], color=color, linestyle=style, linewidth=2, label=label)
            ax.fill_between(x, data["lo"], data["hi"], color=color, alpha=.12)
        ax.set(yscale="log", title=LABELS[variant], xlabel="Environment steps per seed [millions]",
               ylabel="Training position RMSE [m], log scale")
        ax.grid(alpha=.2, which="both")
        ax.text(.98,.06, f"Final window: {summary['old_training_position_rmse_m']*100:.1f} → "
                f"{summary['new_training_position_rmse_m']*100:.1f} cm", transform=ax.transAxes,
                ha="right", fontsize=10, bbox=dict(facecolor="white", alpha=.8, edgecolor="none"))
    axes.flat[0].legend(fontsize=9, loc="lower left")
    fig.suptitle("Does the stronger reward improve position tracking?\n"
                 "Same tasks, seeds, network and budget; 16-update pooling; shading = seed range", fontsize=14)
    fig.savefig(output / "position_rmse_before_after.png", dpi=170)
    plt.close(fig)

    fig, ax = plt.subplots(figsize=(11, 5.5), constrained_layout=True)
    x = np.arange(len(VARIANTS))
    for offset, condition, color, label in [(-.19,"old","#94a3b8","Original reward"),
                                          (.19,"new","#2563eb","Position-focused reward")]:
        values = [row[condition + "_test_position_rmse_m"] * 100 for row in summaries]
        bars = ax.bar(x+offset, values, width=.36, color=color, label=label)
        for bar, row in zip(bars, summaries):
            ax.text(bar.get_x()+bar.get_width()/2, bar.get_height()+3,
                    f"{row[condition+'_test_completed']}/36", ha="center", fontsize=10)
    ax.set_xticks(x, [variant[0] for variant in VARIANTS])
    ax.set(ylabel="Mean test position RMSE [cm]", xlabel="Policy; labels above bars = completed 20-second rollouts",
           title="Tracking on the same 12 new references × 3 trained seeds\n"
                 "RMSE ends at failure; read it together with completion counts")
    ax.set_ylim(top=max(ax.get_ylim()[1], max(row['old_test_position_rmse_m']*100 for row in summaries)*1.18))
    ax.legend(); ax.grid(axis="y",alpha=.2)
    fig.savefig(output / "test_rmse_before_after.png", dpi=170)
    plt.close(fig)
    manifest=dict(status="complete",created_utc=datetime.now(timezone.utc).isoformat(), variants=VARIANTS,
        old_training=str(old_run),new_training=str(new_run),new_evaluation=str(new_eval),
        old_reward_profile=old.get("reward_profile","legacy"),new_reward_profile=new.get("reward_profile","legacy"),
        new_reward_settings=new.get("reward_settings"),
        source_sha256=source_hashes,training_csv_sha256=raw_hashes,
        original_training_manifest_sha256=sha(old_run/'manifest.json'),new_training_manifest_sha256=sha(new_run/'manifest.json'),
        notes=["Raw reward values differ between objectives; compare position RMSE/completion, not reward magnitudes.",
               "All evaluations use identical references, physical starts and deterministic actor means.",
               "Old failed runs have shorter RMSE horizons; completion counts are reported alongside errors."])
    write_json(output / "manifest.json",manifest)
    report=["# Stronger position penalty: A/B/K/L", "",
        "Only these four policies were retrained. Three seeds, 2,097,152 steps per seed, same task/network/PPO budget.", "",
        "Position weight: 0.02 → 1.0. Velocity weight: 0.001 → 0.1. Attitude weight: 0.6. "
        "Failure cost: 25 → 1500. Critic reward units are divided by 50; logs retain raw rewards.", "",
        "Raw reward magnitudes are not comparable between profiles. Tracking accuracy and completion are.", "",
        "![Training position RMSE](position_rmse_before_after.png)","",
        "![Test position RMSE](test_rmse_before_after.png)","",
        "| Policy | Training RMSE old → new [cm] | Test RMSE old → new [cm] | Completed old → new |",
        "|---|---:|---:|---:|"]
    for r in summaries:
        report.append(f"| {LABELS[r['variant']]} | {r['old_training_position_rmse_m']*100:.1f} → {r['new_training_position_rmse_m']*100:.1f} | "
          f"{r['old_test_position_rmse_m']*100:.1f} → {r['new_test_position_rmse_m']*100:.1f} | {r['old_test_completed']}/36 → {r['new_test_completed']}/36 |")
    report += ["", "## Attitude tradeoff", "",
               "Position is weighted more strongly relative to attitude. The measured attitude errors are:", "",
               "| Policy | Attitude RMSE old → new [rad] |", "|---|---:|"]
    for r in summaries:
        report.append(f"| {r['variant'][0]} | {r['old_test_attitude_rmse_rad']:.4f} → {r['new_test_attitude_rmse_rad']:.4f} |")
    report += ["", "Training RMSE pools the final 262,144 steps per seed. Test RMSE averages per-rollout RMSE up to failure. "
               "Raw histories and code are saved here; all checkpoints/state traces remain in the linked source run directories."]
    (output/'REPORT.md').write_text('\n'.join(report)+'\n',encoding='utf-8')
    print(json.dumps(summaries,indent=2))


if __name__ == "__main__":
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--old',type=Path,default=Path('runs/longer_training_reward_all_12'))
    parser.add_argument('--new',type=Path,default=Path('runs/position_focus_abkl_2m'))
    parser.add_argument('--evaluation',type=Path,default=Path('runs/path_tracking_position_focus_abkl_2m'))
    parser.add_argument('--output',type=Path,default=Path('runs/position_focus_abkl_comparison'))
    run(parser.parse_args())
