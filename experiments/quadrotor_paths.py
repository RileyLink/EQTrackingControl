"""Plot actual/reference paths from saved policies, without additional training."""

import argparse
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import shutil
import textwrap

import jax
import numpy as np

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from envs.quadrotor_smooth import Reference, make_bank, make_environment
from experiments.quadrotor_comparison import (
    COLORS, LABELS, OBSERVATION_SIZES, load_policy, read_csv, save_bank,
    write_csv, write_json, write_policy_dimensions,
)
from rl.quadrotor_ppo import EVALUATION_COLUMNS, make_evaluator


def selected_path(frames, bank, trajectory, dt):
    """Include the initial point and failing transition, exclude frozen frames."""
    active = np.asarray(frames["active"][:, trajectory], dtype=bool)
    count = int(active.sum())
    if not np.array_equal(active, np.arange(len(active)) < count):
        raise ValueError("Rollout mask is not a contiguous prefix")
    position = np.concatenate([frames["initial_position"][trajectory][None],
                               frames["position"][:count, trajectory]], axis=0)
    reference = np.asarray(bank.pos[trajectory, :count + 1])
    failed = bool(np.any(frames["failed"][:count, trajectory]))
    error = np.linalg.norm(position - reference, axis=1)
    return dict(time=np.arange(count + 1) * dt, position=position,
                reference=reference, error=error, failed=failed)


def set_3d_limits(ax, arrays):
    points = np.concatenate(arrays, axis=0)
    center = (points.min(axis=0) + points.max(axis=0)) / 2
    span = max(float(np.ptp(points, axis=0).max()), .25) * 1.15
    for setter, coordinate in zip([ax.set_xlim, ax.set_ylim, ax.set_zlim], center):
        setter(coordinate - span / 2, coordinate + span / 2)
    ax.set_box_aspect((1, 1, 1))
    ax.view_init(elev=25, azim=-55)
    ax.set(xlabel="x [m]", ylabel="y [m]", zlabel="z [m]")
    ax.tick_params(labelsize=8)


def path_panel(ax, path, desired, variant, row):
    ax.plot(*desired.T, color="#111827", linestyle="--", linewidth=1.5, label="Desired")
    ax.plot(*path["position"].T, color=COLORS[variant], linewidth=1.8,
            label=f"Actual (obs {OBSERVATION_SIZES[variant]}, act 4)")
    ax.scatter(*path["position"][0], color="#16a34a", s=22, label="Start")
    if path["failed"]:
        ax.scatter(*path["position"][-1], color="#dc2626", marker="x", s=45, label="Failure")
    set_3d_limits(ax, [desired, path["position"]])
    status = f"failed at {path['time'][-1]:.2f}s" if path["failed"] else f"completed {path['time'][-1]:g}s"
    title = textwrap.fill(LABELS[variant], 31)
    ax.set_title(f"{title}\nRMSE {row['position_rmse_m'] * 100:.1f} cm | {status}", fontsize=10)


def plot_paths(output, manifest, bank, chosen, selected):
    desired = np.asarray(bank.pos[manifest["display_trajectory"]])
    variants = manifest["variants"]
    ncols = min(5, len(variants))
    nrows = int(np.ceil(len(variants) / ncols))
    # Matplotlib's constrained layout underestimates the extents of 3D labels.
    # Reserve explicit margins/row spacing so all ten panels remain readable.
    fig = plt.figure(figsize=(4.8 * ncols, 6.8 if len(variants) == 1 else 5.8 * nrows))
    grid = fig.add_gridspec(nrows, ncols, left=.025, right=.975, bottom=.085,
                           top=.73 if len(variants) == 1 else .84, wspace=.16, hspace=.42)
    for index, variant in enumerate(variants):
        ax = fig.add_subplot(grid[index // ncols, index % ncols], projection="3d")
        path_panel(ax, chosen[variant], desired, variant, selected[variant])
    if len(variants) == 1:
        ax.legend(fontsize=8, loc="upper left")
        fig.suptitle("Actual simulated path versus desired\n"
                     f"Training seed {manifest['display_seed']} | new reference {manifest['display_trajectory']}\n"
                     "Dashed = desired; colored = actual", fontsize=11)
    else:
        fig.suptitle("Actual simulated paths versus desired path\n"
                     f"Fixed training seed {manifest['display_seed']}, new reference {manifest['display_trajectory']}; "
                     "black dashed = desired, colored = actual; paths stop at first failure", fontsize=15)
    fig.savefig(output / "all_policy_paths.png", dpi=150, bbox_inches="tight", pad_inches=.25)
    plt.close(fig)

    fig, ax = plt.subplots(figsize=(12, 6), constrained_layout=True)
    for variant in variants:
        path = chosen[variant]
        ax.plot(path["time"], np.maximum(path["error"], 1e-8), color=COLORS[variant],
                label=LABELS[variant], linewidth=1.8,
                linestyle=":" if variant.startswith("I_") else "--" if variant.startswith("J_") else "-")
        if path["failed"]:
            ax.scatter(path["time"][-1], path["error"][-1], color=COLORS[variant], marker="x", s=45)
    ax.set(yscale="log", xlabel="Time [s]", ylabel="Position error ||p - p_d|| [m], logarithmic scale",
           title="Tracking error on the same new trajectory and initial condition\n"
                 f"Fixed seed {manifest['display_seed']}; crosses mark failures; curves stop at failure",
           xlim=(0, manifest["horizon"] * manifest["dt"]))
    ax.grid(alpha=.2, which="both")
    ax.legend(fontsize=9, ncol=2, loc="upper right")
    fig.savefig(output / "position_error.png", dpi=160)
    plt.close(fig)

    focus = [name for name in ["I_full_double_thrust_extension", "J_symmetry_double_thrust_extension"]
             if name in variants]
    if len(focus) != 2:
        return
    fig = plt.figure(figsize=(15, 9), constrained_layout=True)
    grid = fig.add_gridspec(2, 3)
    for index, variant in enumerate(focus):
        ax = fig.add_subplot(grid[0, index], projection="3d")
        path_panel(ax, chosen[variant], desired, variant, selected[variant])
        if index == 0:
            ax.legend(fontsize=8, loc="upper left")
    ax = fig.add_subplot(grid[0, 2])
    for variant in focus:
        path = chosen[variant]
        ax.plot(path["time"], path["error"] * 100, color=COLORS[variant],
                label=f"{variant[0]} (obs {OBSERVATION_SIZES[variant]}, act 4)", linewidth=2)
        if path["failed"]:
            ax.scatter(path["time"][-1], path["error"][-1] * 100, color=COLORS[variant], marker="x", s=45)
    ax.set(title="Position error", xlabel="Time [s]", ylabel="Euclidean error [cm], logarithmic scale", yscale="log")
    ax.grid(alpha=.2, which="both")
    ax.legend()
    times = np.arange(len(desired)) * manifest["dt"]
    for coordinate, name in enumerate("xyz"):
        ax = fig.add_subplot(grid[1, coordinate])
        ax.plot(times, desired[:, coordinate], color="#111827", linestyle="--", label="Desired", linewidth=1.8)
        for variant in focus:
            path = chosen[variant]
            ax.plot(path["time"], path["position"][:, coordinate], color=COLORS[variant],
                    label=f"{variant[0]} (obs {OBSERVATION_SIZES[variant]}, act 4)", linewidth=1.8)
        ax.set(title=f"{name} coordinate", xlabel="Time [s]", ylabel=f"{name} [m]")
        ax.grid(alpha=.2)
        if coordinate == 0:
            ax.legend(fontsize=9)
    fig.suptitle("Double thrust extension: actual versus desired motion\n"
                 f"Training seed {manifest['display_seed']}, new reference {manifest['display_trajectory']}; "
                 "same rigid-body starting condition; deterministic actor means", fontsize=15)
    fig.savefig(output / "double_extension_paths.png", dpi=160)
    plt.close(fig)


def run(args):
    training = args.run.resolve()
    original = json.loads((training / "manifest.json").read_text(encoding="utf-8"))
    if original["status"] != "complete":
        raise ValueError("Choose a completed training run")
    variants = original["variants"] if args.variants is None else args.variants
    if len(set(variants)) != len(variants) or not set(variants) <= set(original["variants"]):
        raise ValueError("Choose distinct policies present in the saved run")
    if args.seed not in original["seeds"] or not 0 <= args.trajectory < args.count:
        raise ValueError("Choose a saved display seed and valid trajectory index")
    output = args.output.resolve()
    output.mkdir(parents=True, exist_ok=False)
    manifest = dict(status="running", created_utc=datetime.now(timezone.utc).isoformat(),
        training_run=str(training), training_manifest_sha256=hashlib.sha256((training / "manifest.json").read_bytes()).hexdigest(),
        checkpoint_steps=original["config"]["total_steps"], variants=variants, seeds=original["seeds"],
        reward_profile=original.get("reward_profile", "legacy"), reward_settings=original.get("reward_settings"),
        reference_seed=original["reference_seeds"]["held_out"], trajectory_count=args.count,
        horizon=original["horizon"], dt=original["dt"], initial_condition_seed=original["initial_condition_seed"],
        display_seed=args.seed, display_trajectory=args.trajectory,
        policy_mode="deterministic actor mean; no action sampling", selection="fixed seed and reference index; no performance selection",
        metrics="RMSE excludes initial point; includes failing transition; no samples after first failure",
        notes=["No additional PPO training. Training histories and original manifests are untouched.",
               "References use a different generator seed from training.",
               "Physical starting perturbations are paired across all variants and policy seeds.",
               "F/Fdot initial conditions follow each variant's trained actuator model.",
               "Raw traces include frozen frames after failure; always apply the active mask."])
    (output / "source").mkdir()
    write_policy_dimensions(output, variants)
    manifest["source_sha256"] = {}
    for source in [Path(__file__), Path("rl/quadrotor_ppo.py"), Path("envs/quadrotor_smooth.py"),
                   Path("envs/quadrotor_thrust_rate.py"), Path("envs/quadrotor_double_thrust.py"),
                   Path("envs/quadrotor_desired_thrust.py"),
                   Path("experiments/quadrotor_comparison.py"), Path("requirements-quadrotor-lock.txt")]:
        shutil.copy2(source, output / "source" / source.name)
        manifest["source_sha256"][source.name] = hashlib.sha256(source.read_bytes()).hexdigest()
    write_json(output / "manifest.json", manifest)
    try:
        trajectories, bank = make_bank(manifest["reference_seed"], args.count, manifest["horizon"], dt=manifest["dt"])
        manifest["reference_bank_sha256"] = save_bank(output / "references.npz", trajectories, bank)
        chosen, selected, all_rows, summaries, checkpoint_hashes = {}, {}, [], [], {}
        for variant in variants:
            env = make_environment(bank, variant, dt=manifest["dt"], reward_profile=manifest["reward_profile"])
            # Compile once per observation/action model; reuse across trained seeds.
            model, _, _ = load_policy(training / variant / f"seed_{original['seeds'][0]}")
            evaluate = make_evaluator(env, model, seed=manifest["initial_condition_seed"], record_state=True)
            variant_rows = []
            for seed in original["seeds"]:
                checkpoint = training / variant / f"seed_{seed}"
                _, params, config = load_policy(checkpoint)
                if config["steps"] != manifest["checkpoint_steps"] or config["observation_size"] != env.obs_size:
                    raise ValueError("Checkpoint step count or observation definition mismatch")
                if config.get("reward_profile", "legacy") != manifest["reward_profile"]:
                    raise ValueError("Checkpoint reward profile differs from the training manifest")
                metrics, frames = jax.block_until_ready(evaluate(params))
                metrics = np.asarray(metrics)
                frames = {name: np.asarray(value) for name, value in frames.items()}
                if not np.isfinite(metrics).all():
                    raise FloatingPointError(f"Non-finite evaluation: {variant}, seed {seed}")
                destination = output / variant / f"seed_{seed}"
                destination.mkdir(parents=True)
                np.savez_compressed(destination / "rollout.npz", **frames)
                checkpoint_hashes[f"{variant}/seed_{seed}"] = hashlib.sha256((checkpoint / "policy.msgpack").read_bytes()).hexdigest()
                rows = [dict(variant=variant, seed=seed, trajectory=index,
                             **dict(zip(EVALUATION_COLUMNS, row.tolist()))) for index, row in enumerate(metrics)]
                variant_rows.extend(rows)
                if seed == args.seed:
                    path = selected_path(frames, bank, args.trajectory, manifest["dt"])
                    chosen[variant] = path
                    selected[variant] = dict(rows[args.trajectory],
                        max_position_error_m=float(path["error"][1:].max()),
                        final_position_error_m=float(path["error"][-1]))
                    trace = [dict(time_s=float(t), x=float(p[0]), y=float(p[1]), z=float(p[2]),
                                  desired_x=float(d[0]), desired_y=float(d[1]), desired_z=float(d[2]),
                                  position_error_m=float(e))
                             for t, p, d, e in zip(path["time"], path["position"], path["reference"], path["error"])]
                    write_csv(destination / "display_path.csv", trace)
                print(f"Evaluated {variant} seed {seed}: {int((metrics[:, 7] == 0).sum())}/{args.count} completed; "
                      f"mean position RMSE {metrics[:, 0].mean():.4f} m", flush=True)
            all_rows.extend(variant_rows)
            completed = [row for row in variant_rows if not row["failed"]]
            summaries.append(dict(variant=variant, evaluations=len(variant_rows), completed=len(completed),
                failures=len(variant_rows) - len(completed),
                position_rmse_m=float(np.mean([row["position_rmse_m"] for row in variant_rows])),
                completed_only_position_rmse_m=float(np.mean([row["position_rmse_m"] for row in completed])) if completed else None,
                attitude_rmse_rad=float(np.mean([row["attitude_rmse_rad"] for row in variant_rows])),
                mean_survival_seconds=float(np.mean([row["survival_seconds"] for row in variant_rows]))))
            write_csv(output / "per_trajectory.csv", all_rows)
            write_csv(output / "summary.csv", summaries)
            jax.clear_caches()
        write_csv(output / "display_trajectory.csv", list(selected.values()))
        plot_paths(output, manifest, bank, chosen, selected)
        manifest.update(status="complete", completed_utc=datetime.now(timezone.utc).isoformat(),
                        checkpoint_sha256=checkpoint_hashes)
        write_json(output / "manifest.json", manifest)
        report = ["# Actual simulated paths and tracking error", "",
                  f"Final policies after {manifest['checkpoint_steps']:,} training steps per seed; no additional training.", "",
                  f"Figures use fixed training seed {args.seed}, new reference {args.trajectory}, "
                  "and the same physical starting condition. Paths stop at first failure.", "",
                  "Three-dimensional panels use individual axis limits. Failure conditions include position, "
                  "velocity, attitude and angular-velocity limits, not only distance from the path.", "",
                  *(["![Double extension](double_extension_paths.png)", ""]
                    if {"I_full_double_thrust_extension", "J_symmetry_double_thrust_extension"} <= set(variants) else []),
                  "![All policy paths](all_policy_paths.png)", "", "![Position error](position_error.png)", "",
                  "## Displayed trajectory", "",
                  "| Policy | Position RMSE [cm] | Maximum error [cm] | Duration [s] | Failed |",
                  "|---|---:|---:|---:|---|"]
        for variant, row in selected.items():
            report.append(f"| {LABELS[variant]} | {row['position_rmse_m'] * 100:.2f} | "
                          f"{row['max_position_error_m'] * 100:.2f} | {row['survival_seconds']:.2f} | {bool(row['failed'])} |")
        report.extend(["", "## All new references and trained seeds", "",
                       f"Each policy is evaluated on the same {args.count} new trajectories for each of {len(original['seeds'])} trained seeds. "
                       "RMSE is calculated up to first failure, including the failing transition; short failed runs "
                       "must be interpreted with completion counts and survival time.", "",
                       "| Policy | Completed | Mean position RMSE [cm] | Completed-only RMSE [cm] | Mean duration [s] |",
                       "|---|---:|---:|---:|---:|"])
        for row in summaries:
            completed_rmse = "--" if row["completed_only_position_rmse_m"] is None else f"{row['completed_only_position_rmse_m'] * 100:.2f}"
            report.append(f"| {LABELS[row['variant']]} | {row['completed']}/{row['evaluations']} | "
                          f"{row['position_rmse_m'] * 100:.2f} | {completed_rmse} | {row['mean_survival_seconds']:.2f} |")
        report.extend(["", "Raw traces, reference bank, checkpoint hashes and evaluation source are saved here. "
                       "Each rollout archive includes position/velocity/attitude/angular velocity/rotor commands, "
                       "initial states, and active/failure masks. Thrust-extension cases also record F and Fdot features."])
        (output / "REPORT.md").write_text("\n".join(report) + "\n", encoding="utf-8")
        print(f"Saved path report: {output / 'REPORT.md'}", flush=True)
    except BaseException as error:
        manifest.update(status="failed", error=f"{type(error).__name__}: {error}")
        write_json(output / "manifest.json", manifest)
        raise


def replot(output):
    """Regenerate figures from saved traces; retain the original evaluation source."""
    output = Path(output).resolve()
    manifest = json.loads((output / "manifest.json").read_text(encoding="utf-8"))
    if manifest["status"] != "complete":
        raise ValueError("Choose a completed path evaluation")
    with np.load(output / "references.npz") as archive:
        bank = Reference(*(archive["reference_" + name] for name in Reference._fields))
    selected = {row["variant"]: {key: value if key == "variant" else float(value)
                                 for key, value in row.items()}
                for row in read_csv(output / "display_trajectory.csv")}
    chosen = {}
    for variant in manifest["variants"]:
        path = output / variant / f"seed_{manifest['display_seed']}" / "rollout.npz"
        with np.load(path) as archive:
            frames = dict(archive)
        chosen[variant] = selected_path(frames, bank, manifest["display_trajectory"], manifest["dt"])
    plot_paths(output, manifest, bank, chosen, selected)
    (output / "plot_source").mkdir(exist_ok=True)
    source = Path(__file__)
    shutil.copy2(source, output / "plot_source" / source.name)
    manifest["plot_source_sha256"] = {source.name: hashlib.sha256(source.read_bytes()).hexdigest()}
    manifest["replotted_utc"] = datetime.now(timezone.utc).isoformat()
    write_json(output / "manifest.json", manifest)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("run", type=Path)
    parser.add_argument("--output", type=Path)
    parser.add_argument("--plot-only", action="store_true", help="Regenerate figures from a completed path evaluation")
    parser.add_argument("--variants", nargs="+")
    parser.add_argument("--count", type=int, default=12)
    parser.add_argument("--seed", type=int, default=0, help="Fixed display seed; all saved seeds are evaluated")
    parser.add_argument("--trajectory", type=int, default=0, help="Fixed display index; no performance selection")
    args = parser.parse_args()
    if args.plot_only:
        replot(args.run)
    elif args.output is None:
        parser.error("--output is required for a new evaluation")
    else:
        run(args)
