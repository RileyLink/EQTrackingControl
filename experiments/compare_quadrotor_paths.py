"""Compare A/B/K/L using saved, paired rollouts; no training or simulation."""

import argparse
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import shutil
import textwrap

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from envs.quadrotor_smooth import Reference
from experiments.quadrotor_comparison import (
    COLORS, LABELS, OBSERVATION_SIZES, read_csv, write_csv, write_json, write_policy_dimensions,
)
from experiments.quadrotor_paths import selected_path, set_3d_limits


SOURCES = {
    "A_full": Path("runs/path_tracking_2m"),
    "B_symmetry": Path("runs/path_tracking_2m"),
    "K_full_desired_thrust_extension": Path("runs/path_tracking_desired_thrust_2m"),
    "L_symmetry_desired_thrust_extension": Path("runs/path_tracking_desired_thrust_symmetry_2m"),
}
SHORT_NAMES = {"A_full": "A: full state", "B_symmetry": "B: symmetry",
               "K_full_desired_thrust_extension": "K: desired extension, full",
               "L_symmetry_desired_thrust_extension": "L: desired extension, symmetry"}


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def render(output, variants, paths, desired, selected, seed, trajectory, dt, reward_profile="legacy"):
    fig = plt.figure(figsize=(12, 11))
    grid = fig.add_gridspec(2, 2, left=.045, right=.94, bottom=.07, top=.84,
                           wspace=.20, hspace=.43)
    for index, variant in enumerate(variants):
        ax = fig.add_subplot(grid[index // 2, index % 2], projection="3d")
        path = paths[variant]
        ax.plot(*desired.T, "--", color="#111827", linewidth=1.8, label="Desired")
        ax.plot(*path["position"].T, color=COLORS[variant], linewidth=2.2,
                label=f"Actual (obs {OBSERVATION_SIZES[variant]}, act 4)")
        ax.scatter(*path["position"][0], color="#16a34a", s=24)
        if path["failed"]:
            ax.scatter(*path["position"][-1], color="#dc2626", marker="x", s=60, linewidth=2)
        set_3d_limits(ax, [desired, path["position"]])
        title = (f"{SHORT_NAMES[variant]}\n(obs {OBSERVATION_SIZES[variant]}, act 4)\n"
                 f"Position RMSE {selected[variant]['position_rmse_m'] * 100:.1f} cm | "
                 + (f"failed at {path['time'][-1]:.2f} s" if path["failed"] else f"completed {path['time'][-1]:g} s"))
        ax.set_title(title, fontsize=11, pad=12)
        ax.legend(fontsize=8, loc="upper left")
    fig.suptitle(f"Actual paths: A, B, K and L | {reward_profile}\n"
                 f"Same new reference {trajectory}, training seed {seed}, and physical start\n"
                 "Dashed black = desired; colored = actual; red cross = failure; panel axes scale individually",
                 fontsize=13, y=.985)
    fig.savefig(output / "paths_abkl.png", dpi=170, bbox_inches="tight", pad_inches=.25)
    plt.close(fig)

    fig, ax = plt.subplots(figsize=(12, 5.5), constrained_layout=True)
    for variant in variants:
        path = paths[variant]
        ax.plot(path["time"], np.maximum(path["error"], 1e-8), color=COLORS[variant],
                label=LABELS[variant], linewidth=2.2)
        if path["failed"]:
            ax.scatter(path["time"][-1], path["error"][-1], marker="x", color=COLORS[variant],
                       s=60, linewidth=2, zorder=5)
    ax.set(yscale="log", xlabel="Time [s]", ylabel="3D position error [m], logarithmic scale",
           xlim=(0, (len(desired) - 1) * dt), title="Position error on the same reference and initial condition\n"
           "Each line is one actual rollout; curves stop at the first failure")
    ax.grid(alpha=.2, which="both")
    ax.legend(fontsize=9, ncol=2, loc="upper right")
    fig.savefig(output / "position_error_abkl.png", dpi=170)
    plt.close(fig)

    fig, axes = plt.subplots(3, 1, figsize=(12, 9), sharex=True, constrained_layout=True)
    times = np.arange(len(desired)) * dt
    for coordinate, ax in enumerate(axes):
        ax.plot(times, desired[:, coordinate], "--", color="#111827", label="Desired", linewidth=2)
        for variant in variants:
            path = paths[variant]
            ax.plot(path["time"], path["position"][:, coordinate], color=COLORS[variant],
                    label=f"{variant[0]} (obs {OBSERVATION_SIZES[variant]}, act 4)", linewidth=1.8)
        ax.set_ylabel("xyz"[coordinate] + " [m]")
        ax.grid(alpha=.2)
    axes[0].legend(fontsize=9, ncol=3)
    axes[-1].set_xlabel("Time [s]")
    fig.suptitle("Desired and actual x/y/z coordinates | same trajectory and start", fontsize=14)
    fig.savefig(output / "coordinates_abkl.png", dpi=170)
    plt.close(fig)

    # A common-scale overlay complements the individually scaled small multiples.
    fig = plt.figure(figsize=(10, 8))
    ax = fig.add_subplot(projection="3d")
    ax.plot(*desired.T, "--", color="#111827", label="Desired", linewidth=2)
    for variant in variants:
        path = paths[variant]
        ax.plot(*path["position"].T, color=COLORS[variant], label=LABELS[variant], linewidth=2)
        if path["failed"]:
            ax.scatter(*path["position"][-1], color=COLORS[variant], marker="x", s=60)
    set_3d_limits(ax, [desired] + [paths[name]["position"] for name in variants])
    ax.set_title("All four actual paths on the same spatial scale", pad=20)
    ax.legend(fontsize=8, loc="upper left")
    fig.savefig(output / "common_scale_overlay.png", dpi=170, bbox_inches="tight", pad_inches=.3)
    plt.close(fig)


def run(args):
    variants = list(SOURCES)
    sources = {variant: args.evaluation_run or source for variant, source in SOURCES.items()}
    manifests = {variant: json.loads((source / "manifest.json").read_text(encoding="utf-8"))
                 for variant, source in sources.items()}
    base = manifests[variants[0]]
    shared = ["checkpoint_steps", "seeds", "reference_seed", "trajectory_count", "horizon", "dt",
              "initial_condition_seed", "policy_mode", "reference_bank_sha256"]
    for variant, manifest in manifests.items():
        if manifest["status"] != "complete" or variant not in manifest["variants"]:
            raise ValueError(f"Missing completed rollout: {variant}")
        if manifest.get("reward_profile", "legacy") != base.get("reward_profile", "legacy"):
            raise ValueError("Choose rollouts with the same reward profile")
        for field in shared:
            if manifest[field] != base[field]:
                raise ValueError(f"Rollouts differ in {field}: {variant}")
        if sha(sources[variant] / "references.npz") != base["reference_bank_sha256"]:
            raise ValueError("Reference archive hash mismatch")
    if args.seed not in base["seeds"] or not 0 <= args.trajectory < base["trajectory_count"]:
        raise ValueError("Choose a saved seed and trajectory index")
    output = args.output.resolve()
    output.mkdir(parents=True, exist_ok=False)
    (output / "source").mkdir()
    source_hashes = {}
    for source in [Path(__file__), Path("experiments/quadrotor_paths.py"),
                   Path("experiments/quadrotor_comparison.py"), Path("envs/quadrotor_smooth.py"),
                   Path("requirements-quadrotor-lock.txt")]:
        shutil.copy2(source, output / "source" / source.name)
        source_hashes[source.name] = sha(source)
    shutil.copy2(sources[variants[0]] / "references.npz", output / "references.npz")
    with np.load(output / "references.npz") as archive:
        bank = Reference(*(archive["reference_" + name] for name in Reference._fields))
    desired = np.asarray(bank.pos[args.trajectory])
    paths, selected, summaries, provenance = {}, {}, [], {}
    common_start = None
    for variant in variants:
        source = sources[variant]
        original_file = source / variant / f"seed_{args.seed}" / "rollout.npz"
        copied = output / variant
        copied.mkdir()
        shutil.copy2(original_file, copied / "rollout.npz")
        with np.load(copied / "rollout.npz") as archive:
            frames = dict(archive)
        initial = {key: frames[key][args.trajectory] for key in
                   ["initial_position", "initial_velocity", "initial_rotation", "initial_omega"]}
        if common_start is None:
            common_start = initial
        for key, value in initial.items(): np.testing.assert_array_equal(value, common_start[key])
        path = selected_path(frames, bank, args.trajectory, base["dt"])
        paths[variant] = path
        candidates = [row for row in read_csv(source / "per_trajectory.csv") if row["variant"] == variant
                      and int(row["seed"]) == args.seed and int(row["trajectory"]) == args.trajectory]
        if len(candidates) != 1:
            raise ValueError("Missing or duplicate selected rollout metrics")
        row = {name: value if name == "variant" else float(value) for name, value in candidates[0].items()}
        recomputed_rmse = float(np.sqrt(np.mean(path["error"][1:] ** 2)))
        np.testing.assert_allclose(recomputed_rmse, row["position_rmse_m"], rtol=2e-5, atol=2e-5)
        row.update(max_position_error_m=float(path["error"][1:].max()),
                   final_position_error_m=float(path["error"][-1]))
        selected[variant] = row
        summaries.extend(row for row in read_csv(source / "summary.csv") if row["variant"] == variant)
        trace = [dict(time_s=float(t), x=float(p[0]), y=float(p[1]), z=float(p[2]),
                      desired_x=float(d[0]), desired_y=float(d[1]), desired_z=float(d[2]),
                      position_error_m=float(error)) for t, p, d, error in
                 zip(path["time"], path["position"], path["reference"], path["error"])]
        write_csv(copied / "display_path.csv", trace)
        shutil.copy2(source / "manifest.json", copied / "source_manifest.json")
        provenance[variant] = dict(source=str(source.resolve()),
            source_manifest_sha256=sha(source / "manifest.json"), rollout_sha256=sha(original_file),
            checkpoint_sha256=manifests[variant]["checkpoint_sha256"][f"{variant}/seed_{args.seed}"])
    write_csv(output / "display_trajectory.csv", list(selected.values()))
    write_csv(output / "summary_all_36.csv", summaries)
    write_policy_dimensions(output, variants)
    render(output, variants, paths, desired, selected, args.seed, args.trajectory, base["dt"], base.get("reward_profile", "legacy"))
    manifest = {field: base[field] for field in shared}
    manifest.update(status="complete", created_utc=datetime.now(timezone.utc).isoformat(), variants=variants,
                    display_seed=args.seed, display_trajectory=args.trajectory, provenance=provenance,
                    reward_profile=base.get("reward_profile", "legacy"),
                    source_sha256=source_hashes,
                    notes=["Reuses saved actual simulated rollouts; no training or simulation rerun.",
                           "Paths include the initial point and failing transition; no post-failure frozen samples.",
                           "Displayed RMSE excludes the initial point and ends at first failure.",
                           "All four physical starting states and references were verified equal.",
                           "A/B choose rotor commands; K/L choose direct f/body torque with desired thrust-reference features."])
    write_json(output / "manifest.json", manifest)
    report = ["# Actual paths: A, B, K and L", "",
              f"Same reference {args.trajectory}, training seed {args.seed}, physical initial state, "
              f"and {base['checkpoint_steps']:,}-step checkpoints. These are saved simulated rollouts.", "",
              "![Actual paths](paths_abkl.png)", "", "![Position error](position_error_abkl.png)", "",
              "| Policy | Observation/action scalars | Position RMSE [cm] | Duration [s] | Result |",
              "|---|---:|---:|---:|---|"]
    for variant, row in selected.items():
        report.append(f"| {SHORT_NAMES[variant]} | {OBSERVATION_SIZES[variant]} / 4 | "
                      f"{row['position_rmse_m'] * 100:.1f} | {row['survival_seconds']:.2f} | "
                      f"{'Failed' if row['failed'] else 'Completed'} |")
    report.extend(["", "RMSE includes only the trajectory up to failure, so failed cases have shorter durations. "
                   "Each small-multiple 3D panel scales individually; common_scale_overlay.png shows common spatial axes. "
                   "coordinates_abkl.png shows x/y/z over time. summary_all_36.csv retains the aggregate results "
                   "over all 12 references and three trained seeds.", "",
                   "## Reproduce", "", "From the repository root, choose a fresh output directory:", "",
                   "```powershell", "..\\.venv\\Scripts\\python.exe -m experiments.compare_quadrotor_paths "
                   f"--seed {args.seed} --trajectory {args.trajectory} --output runs/path_comparison_abkl_repeat", "```", "",
                   "Raw source rollout archives are copied here and verified by hash; source manifests and plotting code are retained."])
    (output / "REPORT.md").write_text("\n".join(report) + "\n", encoding="utf-8")
    print(json.dumps(list(selected.values()), indent=2))
    print(f"Saved comparison: {output / 'REPORT.md'}", flush=True)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--trajectory", type=int, default=0)
    parser.add_argument("--output", type=Path, default=Path("runs/path_comparison_abkl"))
    parser.add_argument("--evaluation-run", type=Path, help="Use one completed evaluation containing all four variants")
    run(parser.parse_args())
