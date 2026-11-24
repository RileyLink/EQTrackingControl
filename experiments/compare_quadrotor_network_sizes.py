"""Compare fully trained pilot- and paper-size policies on paired saved rollouts."""

import argparse
import hashlib
import json
from pathlib import Path
import shutil

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
from matplotlib.ticker import MaxNLocator
import numpy as np

from experiments.quadrotor_comparison import read_csv, write_csv, write_json
from experiments.quadrotor_thrust_preview import B_PRIME, L


RUNS = {
    "3×64": (Path("runs/se23_thrust_preview"), Path("runs/se23_random_control")),
    "6×128": (Path("runs/se23_paper_size_full"), Path("runs/se23_paper_size_random_control")),
}
CASES = {"rate": "Opposite thrust rates", "curvature": "Opposite thrust curvature",
         "paper_sampler": "Paper sampler · held rotor draws", "smooth_random": "Smooth random controls"}
POLICIES = {B_PRIME: "B′", L: "L"}
STYLES = {("3×64", B_PRIME): ("#60a5fa", "--"), ("6×128", B_PRIME): ("#1e3a8a", "-"),
          ("3×64", L): ("#84a32d", "--"), ("6×128", L): ("#365314", "-")}


def digest(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def manifests(paths, fields):
    values = [json.loads((p / "manifest.json").read_text(encoding="utf-8")) for p in paths]
    if any(v["status"] != "complete" for v in values):
        raise ValueError("Both source runs must be complete")
    for name in fields:
        if values[0][name] != values[1][name]:
            raise ValueError(f"Source runs differ in {name}")
    return values


def validate_sources():
    training, random = map(tuple, zip(*RUNS.values()))
    tm = manifests(training, ("seeds", "dt", "horizon", "reward_profile", "warm_updates", "warm_learning_rate",
                              "ppo_teacher_weight", "initial_policy_sha256", "reference_sha256", "derivative_scales"))
    rm = manifests(random, ("seeds", "dt", "horizon", "checkpoint_steps", "reference_count", "reference_seed",
                            "start_seed", "reference_sha256"))
    configs = [{k: v for k, v in m["config"].items() if k not in ("layers", "width")} for m in tm]
    if configs[0] != configs[1]:
        raise ValueError("PPO settings differ beyond network size")
    for dims, path, train, evaluation in zip(((3, 64), (6, 128)), training, tm, rm):
        if (train["config"]["layers"], train["config"]["width"]) != dims:
            raise ValueError("Unexpected network size")
        if evaluation["checkpoint_manifest_sha256"] != digest(path / "manifest.json"):
            raise ValueError("Random rollouts belong to different training checkpoints")
        for checkpoint, sha in evaluation["checkpoint_sha256"].items():
            if digest(path / checkpoint / "policy.msgpack") != sha:
                raise ValueError(f"Checkpoint changed: {checkpoint}")
    return tm, rm


def case_data(case, paths):
    train, random = paths
    if case in ("rate", "curvature"):
        archive = train / "held_out_references.npz"
        with np.load(archive) as data:
            ids = np.flatnonzero(data["family"] == (case == "curvature"))
            return data["pos"][ids], data["rot"][ids], ids, train, "rollouts_full.npz"
    with np.load(random / case / "references.npz") as data:
        return data["pos"], data["rot"], np.arange(len(data["pos"])), random / case, "rollout_full.npz"


def calculate(output, tm):
    summaries, curves, examples, references = [], {}, {}, {}
    start_states = {}
    for case in CASES:
        for size, source in RUNS.items():
            pos, rot, ids, directory, rollout_name = case_data(case, source)
            if case in references:
                np.testing.assert_array_equal(pos[0], references[case])
            references[case] = pos[0]
            for variant in POLICIES:
                errors, completion, angles, tilt_angles = [], [], [], []
                for seed in tm[0]["seeds"]:
                    with np.load(directory / variant / f"seed_{seed}" / rollout_name) as frames:
                        active = frames["active"][:, ids]
                        np.testing.assert_array_equal(frames["reference"][:, ids], pos[:, 1:].swapaxes(0, 1))
                        initial = [frames[f"initial_{k}"][ids] for k in ("position", "velocity", "rotation", "omega")]
                        key = (case, seed)
                        if key not in start_states:
                            start_states[key] = initial
                        for x, y in zip(initial, start_states[key]):
                            np.testing.assert_array_equal(x, y)
                        actual = frames["position"][:, ids]
                        delta = actual - pos[:, 1:].swapaxes(0, 1)
                        errors.append(np.where(active, np.sum(delta**2, axis=-1), np.nan))
                        completion.extend(~frames["failed"][:, ids].any(axis=0))
                        r = frames["rotation"][:, ids]
                        rd = rot[:, 1:].swapaxes(0, 1)
                        cos = (np.sum(r * rd, axis=(-2, -1)) - 1.) / 2.
                        tilt = np.sum(r[..., :, 2] * rd[..., :, 2], axis=-1)
                        angles.extend(np.rad2deg(np.sqrt(np.nanmean(
                            np.where(active, np.arccos(np.clip(cos, -1, 1))**2, np.nan), axis=0))))
                        tilt_angles.extend(np.rad2deg(np.sqrt(np.nanmean(
                            np.where(active, np.arccos(np.clip(tilt, -1, 1))**2, np.nan), axis=0))))
                        if seed == tm[0]["seeds"][0]:
                            examples[case, size, variant] = np.concatenate(
                                [initial[0][:1], actual[active[:, 0], 0]], axis=0)
                joined = np.concatenate(errors, axis=1)
                curves[case, size, variant] = 100 * np.sqrt(np.nanmean(joined, axis=1))
                late = joined[int(8 / tm[0]["dt"]) - 1:]
                late_trials = np.isfinite(late).any(axis=0)
                summaries.append(dict(case=case, network=size, variant=variant, trials=len(completion),
                    completed=int(sum(completion)), position_rmse_cm=float(100 * np.sqrt(np.nanmean(joined, axis=0)).mean()),
                    position_rmse_last4s_cm=(float(100 * np.sqrt(np.nanmean(late[:, late_trials], axis=0)).mean())
                                             if late_trials.any() else None),
                    attitude_rmse_deg=float(np.mean(angles)), thrust_axis_rmse_deg=float(np.mean(tilt_angles))))
    for size, (train, random) in RUNS.items():
        for data in (read_csv(train / "rollout_summary.csv"), read_csv(random / "summary.csv")):
            for row in data:
                if row["condition"] != "full" or row.get("phase", "ppo") != "ppo":
                    continue
                case = row.get("case", row.get("family"))
                actual = next(r for r in summaries if (r["case"], r["network"], r["variant"]) == (case, size, row["variant"]))
                np.testing.assert_allclose(actual["position_rmse_cm"], float(row["mean_position_rmse_cm"]), atol=3e-4)
    write_csv(output / "summary.csv", summaries)
    plot(output, curves, examples, references, tm[0]["dt"])
    return summaries


def plot(output, curves, examples, references, dt):
    fig, axes = plt.subplots(2, 2, figsize=(11, 7.2), sharex=True, sharey=True, layout="constrained")
    legend = [Line2D([], [], color=c, linestyle=ls, lw=2, label=f"{POLICIES[v]} · {s}")
              for (s, v), (c, ls) in STYLES.items()]
    for ax, (case, title) in zip(axes.flat, CASES.items()):
        for (size, variant), (color, ls) in STYLES.items():
            y = curves[case, size, variant]
            ax.plot(dt * np.arange(1, len(y)+1), y, color=color, ls=ls, lw=1.8)
        ax.set(title=title, xlabel="Time [s]", ylabel="3D position RMSE [cm]", yscale="log")
        ax.set_yticks([1, 2, 3, 5, 10, 20, 30], labels=["1", "2", "3", "5", "10", "20", "30"])
        ax.grid(alpha=.22)
    fig.suptitle("Position tracking at equal training budget\nPooled across identical starts and three trained seeds")
    fig.legend(handles=legend, loc="outside lower center", ncol=4)
    fig.savefig(output / "position_rmse.png", dpi=175)
    plt.close(fig)

    fig = plt.figure(figsize=(11.2, 8.3), layout="constrained")
    fig.get_layout_engine().set(hspace=.08)
    grid = fig.add_gridspec(2, 2, height_ratios=[.8, 1.2])
    for i, (case, title) in enumerate(CASES.items()):
        spatial = i >= 2
        ax = fig.add_subplot(grid[i // 2, i % 2], projection="3d" if spatial else None)
        target = references[case] - references[case][0]
        traces = {key: examples[case, *key] - references[case][0] for key in STYLES}
        ax.set_title(title, fontsize=11)
        if spatial:
            ax.plot(*target.T, color="#242424", ls=":", lw=2.6)
            for key, (color, ls) in STYLES.items():
                ax.plot(*traces[key].T, color=color, ls=ls, lw=1.5, alpha=.87)
            ax.scatter(*next(iter(traces.values()))[0], color="#ea580c", s=18)
            ax.set(xlabel="Δx [m]", ylabel="Δy [m]", zlabel="Δz [m]")
            for axis in (ax.xaxis, ax.yaxis, ax.zaxis):
                axis.set_major_locator(MaxNLocator(5))
            ax.view_init(24, -58)
            ax.set_box_aspect((1.2, 1., .8))
        else:
            ax.plot(dt * np.arange(len(target)), target[:, 2], color="#242424", ls=":", lw=2.6)
            for key, (color, ls) in STYLES.items():
                z = traces[key][:, 2]
                ax.plot(dt * np.arange(len(z)), z, color=color, ls=ls, lw=1.45, alpha=.88)
            ax.scatter(0, next(iter(traces.values()))[0, 2], color="#ea580c", s=18)
            ax.set(xlabel="Time [s]", ylabel="Altitude from reference start [m]")
            ax.grid(alpha=.2)
    legend = [Line2D([], [], color="#242424", ls=":", lw=2.6, label="Desired"), *legend]
    fig.legend(handles=legend, loc="outside lower center", ncol=5, fontsize=9)
    fig.suptitle("Actual paths from the same starts\nCheckpoint seed 0 · first reference in each family · orange marks the start")
    fig.savefig(output / "actual_paths.png", dpi=175)
    plt.close(fig)


def run(output):
    tm, rm = validate_sources()
    for train, evaluation in zip(tm, rm):
        if (train["dt"], train["horizon"], train["seeds"]) != (evaluation["dt"], evaluation["horizon"], evaluation["seeds"]):
            raise ValueError("Rollout and training protocols differ")
    for (train, random), t, r in zip(RUNS.values(), tm, rm):
        if digest(train / "held_out_references.npz") != t["reference_sha256"]["held_out"]:
            raise ValueError("Held-out reference hash changed")
        for case, sha in r["reference_sha256"].items():
            if digest(random / case / "references.npz") != sha:
                raise ValueError(f"Random reference hash changed in {case}")
    output.mkdir(parents=True, exist_ok=False)
    rows = calculate(output, tm)
    shutil.copy2(__file__, output / Path(__file__).name)
    write_json(output / "manifest.json", dict(status="complete", paired_physical_starts=True,
        display_seed=tm[0]["seeds"][0], display_rule="first reference in each family", source_sha256=digest(Path(__file__)),
        input_manifests={str(path): digest(path / "manifest.json") for paths in RUNS.values() for path in paths}))
    report = ["# Matched network-size comparison", "",
              "Both sizes trained for 1,048,576 PPO steps per policy and seed after 1,200 imitation updates. "
              "The comparisons use the same trained seeds, held-out references, and physical starting states.", "",
              "![Position RMSE](position_rmse.png)", "", "![Actual paths](actual_paths.png)", "",
              "| Test | Network | Policy | Completed | Position RMSE [cm] | Last 4 s [cm] | Attitude RMSE [deg] |",
              "|---|---|---|---:|---:|---:|---:|"]
    for row in rows:
        late = row["position_rmse_last4s_cm"]
        late_text = f"{late:.2f}" if late is not None else "—"
        report.append(f"| {CASES[row['case']]} | {row['network']} | {POLICIES[row['variant']]} | "
                      f"{row['completed']}/{row['trials']} | {row['position_rmse_cm']:.2f} | "
                      f"{late_text} | {row['attitude_rmse_deg']:.2f} |")
    report += ["", "Rerender with a fresh output path.", "", "```powershell",
               "..\\.venv\\Scripts\\python.exe -m experiments.compare_quadrotor_network_sizes --output runs/se23_network_size_comparison_repeat", "```"]
    (output / "REPORT.md").write_text("\n".join(report)+"\n", encoding="utf-8")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, default=Path("runs/se23_network_size_comparison"))
    run(parser.parse_args().output)
