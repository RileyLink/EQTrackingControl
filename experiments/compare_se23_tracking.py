"""Evaluate saved B/L checkpoints on the earlier 3D path and a level figure eight."""

import argparse
import hashlib
import json
from pathlib import Path
import shutil

import jax
import jax.numpy as jnp
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from envs.quadrotor_smooth import Reference, Trajectory, make_environment, reference_at
from experiments.quadrotor_comparison import COLORS, OBSERVATION_SIZES, load_policy, read_csv, write_csv, write_json
from experiments.quadrotor_paths import selected_path, set_3d_limits
from rl.quadrotor_ppo import EVALUATION_COLUMNS, make_evaluator


TRAINING = (Path("runs/position_focus_abkl_2m"), Path("runs/se23_bl_2m"))
EARLIER = Path("runs/path_tracking_position_focus_abkl_2m")
PAIRS = (("B_symmetry", "B_se23"),
         ("L_symmetry_desired_thrust_extension", "L_se23_desired_thrust_extension"))
WRENCH_TRAINING = (Path("runs/se23_b_prime_2m"), TRAINING[1])
WRENCH_PAIR = ("B_prime_se23_wrench", "L_se23_desired_thrust_extension")
PATH_NAMES = {"maneuver": "Earlier 3D maneuver", "figure_eight": "Level XY figure eight"}


def digest(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def manifest(path):
    return json.loads((path / "manifest.json").read_text(encoding="utf-8"))


def reference_paths(earlier, times):
    archive = EARLIER / "references.npz"
    if digest(archive) != earlier["reference_bank_sha256"]:
        raise ValueError("Saved maneuver reference archive hash changed")
    index = earlier["display_trajectory"]
    with np.load(archive) as saved:
        maneuver = Reference(*(jnp.asarray(saved["reference_" + k][index]) for k in Reference._fields))
    frequency = 2 * jnp.pi / times[-1]
    trajectory = Trajectory(center=jnp.array([0., 0., 1.]), amplitude=jnp.array([.7, .5, 0.]),
                            frequency=frequency * jnp.array([1., 2., 1.]), phase=jnp.zeros(3),
                            heading=0., yaw_amplitude=0., yaw_frequency=0., yaw_phase=0.)
    eight = jax.jit(jax.vmap(reference_at, in_axes=(0, None)))(times, trajectory)
    np.testing.assert_array_equal(eight.pos[:, 2], np.ones(len(times)))
    np.testing.assert_allclose(eight.pos[0], eight.pos[-1], atol=1e-6)
    np.testing.assert_array_equal(eight.yaw, np.zeros(len(times)))
    for path in (maneuver, eight):
        if not all(np.isfinite(x).all() for x in path) or path.action.min() < 0 or path.action.max() > 1:
            raise ValueError("Reference is nonfinite or exceeds the rotor command bounds")
    return {"maneuver": maneuver, "figure_eight": eight}


def angles(rotation):
    """Absolute ZYX pitch and angle between thrust axis and world vertical, in degrees."""
    pitch = np.abs(np.rad2deg(np.arcsin(np.clip(-rotation[..., 2, 0], -1., 1.))))
    tilt = np.rad2deg(np.arccos(np.clip(rotation[..., 2, 2], -1., 1.)))
    return pitch, tilt


def plot(output, references, traces, seed, *, include_baseline=True, wrench_ablation=False):
    pairs = (WRENCH_PAIR,) if wrench_ablation else PAIRS
    visible_pairs = pairs if include_baseline else tuple(pair[1:] for pair in pairs)
    columns = len(visible_pairs)
    width = 8 if wrench_ablation else 11.3
    fig = plt.figure(figsize=(width, 10 if wrench_ablation else 9.1), layout="constrained")
    if wrench_ablation:
        fig.get_layout_engine().set(hspace=.12)
    grid = fig.add_gridspec(2, columns, height_ratios=[1.12, 1.])
    errors, error_axes = plt.subplots(2, columns, figsize=(width, 6.6), sharex=True, sharey=True,
                                      squeeze=False, layout="constrained")
    for row, (path_name, reference) in enumerate(references.items()):
        desired = np.asarray(reference.pos)
        three_dimensional = path_name == "maneuver"
        all_positions = [desired] + [traces[path_name, v]["position"] for pair in visible_pairs for v in pair]
        for column, pair in enumerate(visible_pairs):
            ax = fig.add_subplot(grid[row, column], projection="3d" if three_dimensional else None)
            error_ax = error_axes[row, column]
            controller = "B′ and L · thrust and body torque" if wrench_ablation else (
                "B · rotor commands" if column == 0 else "L · thrust and body torque")
            title = f"{PATH_NAMES[path_name]}  |  {controller}"
            ax.set_title(title, fontsize=11)
            error_ax.set_title(title, fontsize=11)
            dimensions = 3 if three_dimensional else 2
            ax.plot(*desired[:, :dimensions].T, color="#171717", linestyle="--", lw=2, label="Desired")
            for variant in pair:
                trace = traces[path_name, variant]
                baseline = variant == pairs[column][0]
                if wrench_ablation:
                    name = "B′ · current desired wrench" if baseline else "L · adds desired thrust derivatives"
                else:
                    name = "Yaw + translation" if baseline else "SE₂(3)"
                label = f"{name} (obs {OBSERVATION_SIZES[variant]}, act 4)"
                color = COLORS[variant]
                linestyle = "-." if baseline else "-"
                ax.plot(*trace["position"][:, :dimensions].T, color=color, linestyle=linestyle, lw=1.85, label=label)
                error_ax.plot(trace["time"], 100 * trace["error"], color=color, linestyle=linestyle, lw=1.8, label=label)
                if trace["failed"]:
                    ax.scatter(*trace["position"][-1, :dimensions], marker="x", color=color)
                    error_ax.scatter(trace["time"][-1], 100 * trace["error"][-1], marker="x", color=color)
            ax.scatter(*traces[path_name, pair[0]]["position"][0, :dimensions],
                       marker="o", s=24, color="#ea580c", label="Actual start")
            if three_dimensional:
                # Use the same limits and camera for B and L on this path.
                set_3d_limits(ax, all_positions)
            else:
                points = np.concatenate(all_positions, axis=0)
                lower, upper = points.min(axis=0) - .1, points.max(axis=0) + .1
                ax.set(xlabel="x [m]", ylabel="y [m]", aspect="equal",
                       xlim=(lower[0], upper[0]), ylim=(lower[1], upper[1]))
                ax.grid(alpha=.2)
            if wrench_ablation and not three_dimensional:
                ax.legend(fontsize=8, loc="upper center", bbox_to_anchor=(.5, -.16), ncol=2)
            else:
                ax.legend(fontsize=8, loc="upper left" if three_dimensional else "upper right")
            error_ax.grid(alpha=.22)
            if column == 0:
                error_ax.set_ylabel("3D position error [cm]")
            if row == 1:
                error_ax.set_xlabel("Time [s]")
            error_ax.legend(fontsize=8, loc="upper right")
    title = f"Desired and actual paths  |  fixed checkpoint seed {seed}, start 0"
    fig.suptitle(title, fontsize=15)
    fig.savefig(output / "actual_paths.png", dpi=180)
    errors.suptitle(f"Tracking error  |  fixed checkpoint seed {seed}, start 0", fontsize=15)
    errors.savefig(output / "position_error.png", dpi=160)
    plt.close(fig)
    plt.close(errors)


def run(output, *, wrench_ablation=False):
    training_dirs = WRENCH_TRAINING if wrench_ablation else TRAINING
    pairs = (WRENCH_PAIR,) if wrench_ablation else PAIRS
    training = [manifest(path) for path in training_dirs]
    earlier = manifest(EARLIER)
    for key in ("status", "config", "seeds", "horizon", "dt", "bank_sha256", "reward_profile", "reward_settings"):
        if training[0][key] != training[1][key]:
            raise ValueError(f"Checkpoint runs differ in {key}")
    if training[0]["status"] != "complete" or earlier["status"] != "complete":
        raise ValueError("Use completed training and reference runs")
    count = earlier["trajectory_count"]
    dt = earlier["dt"]
    times = jnp.arange(earlier["horizon"] + 1) * dt
    references = reference_paths(earlier, times)
    display_seed = earlier["display_seed"]
    seeds = training[0]["seeds"]
    reward_profile = training[0]["reward_profile"]
    output.mkdir(parents=True, exist_ok=False)
    (output / "source").mkdir()
    sources = [Path(__file__), Path("experiments/quadrotor_paths.py"), Path("experiments/quadrotor_comparison.py"),
               Path("envs/quadrotor_smooth.py"), Path("envs/quadrotor_desired_thrust.py"),
               Path("envs/quadrotor_double_thrust.py"), Path("envs/quadrotor_thrust_rate.py"),
               Path("rl/quadrotor_ppo.py"), Path("requirements-quadrotor-lock.txt")]
    for source in sources:
        shutil.copy2(source, output / "source" / source.name)
    provenance = dict(status="running", training_manifests={str(p): digest(p / "manifest.json") for p in training_dirs},
        earlier_reference=str(EARLIER / "references.npz"), earlier_reference_sha256=earlier["reference_bank_sha256"],
        earlier_reference_index=earlier["display_trajectory"], display_seed=display_seed,
        initial_condition_seed=earlier["initial_condition_seed"], starts=count, seeds=seeds, dt=dt,
        horizon=earlier["horizon"], reward_profile=reward_profile, checkpoint_steps=training[0]["config"]["total_steps"],
        mode="deterministic actor mean", checkpoint_sha256={}, reference_sha256={},
        source_sha256={p.name: digest(p) for p in sources})
    write_json(output / "manifest.json", provenance)
    rows, summaries, traces = [], [], {}
    for path_name, reference in references.items():
        destination = output / path_name
        destination.mkdir()
        np.savez_compressed(destination / "reference.npz", time_s=np.asarray(times), **reference._asdict())
        provenance["reference_sha256"][path_name] = digest(destination / "reference.npz")
        bank = jax.tree.map(lambda x: jnp.broadcast_to(x, (count, *x.shape)), reference)
        pitch_d, tilt_d = angles(np.asarray(reference.rot))
        if path_name == "figure_eight" and tilt_d.max() >= 3.:
            raise ValueError("Figure-eight reference exceeds the intended near-hover tilt")
        paired_initial = None
        for pair in pairs:
            for variant, directory in zip(pair, training_dirs):
                env = make_environment(bank, variant, dt=dt, reward_profile=reward_profile)
                model, _, _ = load_policy(directory / variant / f"seed_{seeds[0]}")
                evaluator = make_evaluator(env, model, seed=earlier["initial_condition_seed"], record_state=True)
                variant_rows = []
                for seed in seeds:
                    checkpoint = directory / variant / f"seed_{seed}"
                    _, params, config = load_policy(checkpoint)
                    if (config["observation_size"], config["steps"], config["reward_profile"]) != (
                            env.obs_size, provenance["checkpoint_steps"], reward_profile):
                        raise ValueError(f"Incompatible checkpoint: {checkpoint}")
                    provenance["checkpoint_sha256"][f"{variant}/seed_{seed}"] = digest(checkpoint / "policy.msgpack")
                    metrics, frames = jax.block_until_ready(evaluator(params))
                    metrics = np.asarray(metrics)
                    frames = {k: np.asarray(v) for k, v in frames.items()}
                    if not np.isfinite(metrics).all():
                        raise FloatingPointError(f"Nonfinite tracking metrics for {variant} seed {seed}")
                    initial = [frames["initial_" + k] for k in ("position", "velocity", "rotation", "omega")]
                    if paired_initial is None:
                        paired_initial = initial
                    for actual, expected in zip(initial, paired_initial):
                        np.testing.assert_allclose(actual, expected, rtol=0., atol=1e-7)
                    trial = destination / variant / f"seed_{seed}"
                    trial.mkdir(parents=True)
                    np.savez_compressed(trial / "rollout.npz", **frames)
                    for start, metric in enumerate(metrics):
                        trace = selected_path(frames, bank, start, dt)
                        rotation = np.concatenate([frames["initial_rotation"][start:start + 1],
                                                   frames["rotation"][frames["active"][:, start], start]])
                        pitch, tilt = angles(rotation)
                        delta = trace["position"][1:] - trace["reference"][1:]
                        np.testing.assert_allclose(np.sqrt(np.mean(np.sum(delta ** 2, axis=1))), metric[0], atol=1e-6)
                        variant_rows.append(dict(path=path_name, variant=variant, seed=seed, start=start,
                            **dict(zip(EVALUATION_COLUMNS, metric.tolist())),
                            position_max_m=float(trace["error"].max()), final_position_error_m=float(trace["error"][-1]),
                            altitude_rmse_m=float(np.sqrt(np.mean(delta[:, 2] ** 2))),
                            max_pitch_deg=float(pitch.max()), max_tilt_deg=float(tilt.max())))
                        if seed == display_seed and start == 0:
                            traces[path_name, variant] = trace
                rows.extend(variant_rows)
                completed = sum(not r["failed"] for r in variant_rows)
                values = np.array([r["position_rmse_m"] for r in variant_rows])
                summaries.append(dict(path=path_name, variant=variant, completed=completed, trials=len(variant_rows),
                    mean_position_rmse_m=float(values.mean()), worst_position_rmse_m=float(values.max()),
                    mean_attitude_rmse_deg=float(np.rad2deg(np.mean([r["attitude_rmse_rad"] for r in variant_rows]))),
                    mean_altitude_rmse_m=float(np.mean([r["altitude_rmse_m"] for r in variant_rows])),
                    max_pitch_deg=max(r["max_pitch_deg"] for r in variant_rows),
                    max_tilt_deg=max(r["max_tilt_deg"] for r in variant_rows),
                    desired_max_pitch_deg=float(pitch_d.max()), desired_max_tilt_deg=float(tilt_d.max())))
                print(f"{path_name} / {variant}: {completed}/{len(variant_rows)} completed, "
                      f"mean 3D position RMSE {100 * values.mean():.2f} cm", flush=True)

    if wrench_ablation:
        saved = read_csv(Path("runs/se23_two_paths/per_rollout.csv"))
        saved_l = {(r["path"], int(r["seed"]), int(r["start"])): float(r["position_rmse_m"])
                   for r in saved if r["variant"] == WRENCH_PAIR[1]}
        for r in rows:
            if r["variant"] == WRENCH_PAIR[1]:
                np.testing.assert_allclose(r["position_rmse_m"], saved_l[r["path"], r["seed"], r["start"]], atol=2e-6)
        provenance["reproduced_saved_l_rmse"] = True
    else:
        old_display = {r["variant"]: r for r in read_csv(EARLIER / "display_trajectory.csv")}
        for variant in (pair[0] for pair in pairs):
            first = next(r for r in rows if r["path"] == "maneuver" and r["variant"] == variant
                         and r["seed"] == display_seed and r["start"] == 0)
            np.testing.assert_allclose(first["position_rmse_m"], float(old_display[variant]["position_rmse_m"]), atol=2e-6)
        provenance["reproduced_earlier_display_rmse"] = True
    write_csv(output / "per_rollout.csv", rows)
    write_csv(output / "summary.csv", summaries)
    plot(output, references, traces, display_seed, wrench_ablation=wrench_ablation)
    provenance.update(status="complete", paired_initial_states=True, wrench_ablation=wrench_ablation)
    write_json(output / "manifest.json", provenance)
    report = ["# Tracking on two paths", "",
              *(["B′ and L use the same SE₂(3) error coordinates and direct collective thrust and body torque actions. "
                 "L adds two desired thrust derivative channels to B′’s input.", ""] if wrench_ablation else []),
              "Each row averages 12 paired starting perturbations across three trained seeds. The plots show seed 0 and start 0.", "",
              "The figure eight uses x = 0.7 sin(2πt/20), y = 0.5 sin(4πt/20), z = 1 m and zero yaw.", "",
              "![Actual paths](actual_paths.png)", "", "![Position error](position_error.png)", "",
              "| Path | Policy | Completed | Mean 3D RMSE [cm] | Mean altitude RMSE [cm] | Peak actual tilt [deg] | Peak absolute pitch [deg] |",
              "|---|---|---:|---:|---:|---:|---:|"]
    for r in summaries:
        report.append(f"| {PATH_NAMES[r['path']]} | {r['variant']} | {r['completed']}/{r['trials']} | "
                      f"{100*r['mean_position_rmse_m']:.2f} | {100*r['mean_altitude_rmse_m']:.2f} | "
                      f"{r['max_tilt_deg']:.2f} | {r['max_pitch_deg']:.2f} |")
    repeat = "--wrench-ablation --output runs/se23_wrench_paths_repeat" if wrench_ablation else "--output runs/se23_two_paths_repeat"
    report += ["", "Reproduce from the repository root with a new output path.", "", "```powershell",
               "..\\.venv\\Scripts\\python.exe -m experiments.compare_se23_tracking " + repeat, "```"]
    (output / "REPORT.md").write_text("\n".join(report) + "\n", encoding="utf-8")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path)
    parser.add_argument("--wrench-ablation", action="store_true", help="Compare B′ and L with identical action coordinates")
    args = parser.parse_args()
    output = args.output or Path("runs/se23_wrench_paths" if args.wrench_ablation else "runs/se23_two_paths")
    run(output, wrench_ablation=args.wrench_ablation)
