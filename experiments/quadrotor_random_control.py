"""Test saved thrust-preview policies on sampled reference rotor controls."""

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

from envs.quadrotor_double_thrust import desired_thrust_jet
from envs.quadrotor_smooth import (
    ALLOCATION, DT, MASS, GRAVITY, State, Reference, actual_derivative_features, advance, make_environment,
)
from experiments.diagnose_thrust_inputs import MaskedActor
from experiments.quadrotor_comparison import load_policy, write_csv, write_json
from experiments.quadrotor_thrust_preview import B_PRIME, L, DERIVATIVE_CHANNELS
from rl.quadrotor_ppo import EVALUATION_COLUMNS, make_evaluator


SOURCE = Path("runs/se23_thrust_preview")
HOVER = MASS * GRAVITY
CASE_NAMES = {"paper_sampler": "Paper sampler · held rotor draws", "smooth_random": "Smooth random controls"}
POLICY_NAMES = {(B_PRIME, "full"): "B′", (L, "full"): "L", (L, "masked"): "L · derivatives masked"}


def digest(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def integrate_references(mid_actions, substeps):
    count = mid_actions.shape[0]
    zeros = jnp.zeros((count, 3))
    initial = State(zeros + jnp.array([0., 0., 2.]), zeros, jnp.broadcast_to(jnp.eye(3), (count, 3, 3)),
                    zeros, jnp.arange(count), jnp.zeros(count, dtype=jnp.int32), jnp.full((count, 4), HOVER))

    def step(state, action):
        state = jax.vmap(lambda s, a: advance(s, a, DT / substeps))(state, action)
        if substeps > 1:
            # At small angular increments, float32 roundoff accumulates over
            # substeps. Keep the reference thrust direction at unit length.
            x, y = state.rot[..., :, 0], state.rot[..., :, 1]
            x = x / jnp.linalg.norm(x, axis=-1, keepdims=True)
            y = y - jnp.sum(x * y, axis=-1, keepdims=True) * x
            y = y / jnp.linalg.norm(y, axis=-1, keepdims=True)
            state = state._replace(rot=jnp.stack([x, y, jnp.cross(x, y)], axis=-1))
        return state, state

    _, after = jax.jit(lambda actions: jax.lax.scan(step, initial, actions))(jnp.swapaxes(mid_actions, 0, 1))
    return jax.tree.map(lambda start, frames: jnp.concatenate(
        [start[:, None], jnp.swapaxes(frames[substeps - 1::substeps], 0, 1)], axis=1), initial, after)


def smooth_controls(knots, times, period):
    phase = times / period
    index = jnp.minimum(phase.astype(jnp.int32), knots.shape[1] - 2)
    r = (phase - index)[None, :, None]
    start, finish = knots[:, index], knots[:, index + 1]
    delta = finish - start
    blend = r**3 * (10 - 15 * r + 6 * r**2)
    rate = 30 * r**2 * (r - 1)**2 / period
    acceleration = 60 * r * (r - 1) * (2*r - 1) / period**2
    return start + blend * delta, rate * delta, acceleration * delta


def random_reference(case, count, horizon, seed):
    key = jax.random.PRNGKey(seed)
    times = DT * jnp.arange(horizon + 1)
    if case == "paper_sampler":
        action = .6573 + jax.random.truncated_normal(key, -.001, .001, (count, horizon + 1, 4))
        mid_actions, substeps = action[:, :-1], 1
        dot = ddot = jnp.zeros(action.shape[:-1])
    else:
        period, substeps = .5, 5
        knot_count = int(np.ceil(horizon * DT / period)) + 1
        collective_key, differential_key = jax.random.split(key)
        collective = .12 * jax.random.truncated_normal(collective_key, -1., 1., (count, knot_count, 1))
        differential = jax.random.truncated_normal(differential_key, -.001, .001, (count, knot_count, 4))
        differential -= differential.mean(axis=-1, keepdims=True)
        knots = HOVER + collective + differential
        action, rate, acceleration = smooth_controls(knots, times, period)
        mids = (jnp.arange(horizon * substeps) + .5) * (DT / substeps)
        mid_actions, _, _ = smooth_controls(knots, mids, period)
        dot, ddot = rate @ ALLOCATION[0], acceleration @ ALLOCATION[0]
    if not all(np.all((u >= 0.) & (u <= 1.)) for u in (action, mid_actions)):
        raise ValueError("Infeasible reference rotor controls")
    states = integrate_references(mid_actions, substeps)._replace(previous_action=action)
    derivative = jax.jit(jax.vmap(jax.vmap(actual_derivative_features)))(states)
    b3 = states.rot[..., :, 2]
    e3 = jnp.array([0., 0., 1.])
    db3 = jnp.einsum("...ij,...j->...i", states.rot, jnp.cross(states.omega, e3))
    jerk = derivative.jerk + dot[..., None] * b3 / MASS
    snap = derivative.snap + (ddot[..., None] * b3 + 2 * dot[..., None] * db3) / MASS
    yaw = jnp.arctan2(derivative.heading[..., 1], derivative.heading[..., 0])
    bank = Reference(states.pos, states.vel, states.rot, states.omega, action, derivative.acc, jerk, snap,
                     yaw, derivative.yaw_rate, derivative.yaw_acc)
    force = action @ ALLOCATION[0]
    np.testing.assert_allclose(desired_thrust_jet(bank), jnp.stack([force, dot, ddot], axis=-1), atol=4e-6, rtol=3e-6)
    if not all(np.isfinite(x).all() for x in bank):
        raise FloatingPointError("Nonfinite reference trajectory")
    return bank, jnp.stack([force, dot, ddot], axis=-1)


def plot(output, banks, error_series):
    fig, axes = plt.subplots(2, 2, figsize=(11, 6), sharex="col", layout="constrained")
    colors = {B_PRIME: "#2563eb", L: "#365314"}
    for column, (case, bank) in enumerate(banks.items()):
        t = DT * np.arange(bank.pos.shape[1])
        force = np.asarray(bank.action[0] @ ALLOCATION[0])
        axes[0, column].set(title=CASE_NAMES[case], ylabel="Reference thrust offset [mN]")
        axes[0, column].plot(t, (force - HOVER) * 1000, color="#475569", lw=1.1,
                            drawstyle="steps-post" if case == "paper_sampler" else "default")
        ax = axes[1, column]
        for (group, variant, condition), series in error_series.items():
            if group == case:
                ax.plot(t[1:], 100 * np.sqrt(np.nanmean(series, axis=(0, 2))), color=colors[variant],
                        linestyle="--" if condition == "masked" else "-", label=POLICY_NAMES[variant, condition])
        ax.set(xlabel="Time [s]", ylabel="Pooled position RMSE [cm]")
        ax.legend(fontsize=9)
    for ax in axes.flat:
        ax.grid(alpha=.18)
    fig.suptitle("Random reference controls · frozen B′ and L checkpoints\n"
                 "Top shows reference 0. Bottom pools active tracking samples across references and seeds.")
    fig.savefig(output / "tracking.png", dpi=175)
    plt.close(fig)


def run(output, count, seed, *, checkpoint_run=SOURCE):
    saved = json.loads((checkpoint_run / "manifest.json").read_text(encoding="utf-8"))
    if saved["status"] != "complete" or saved["dt"] != DT or count < 1:
        raise ValueError("Need completed checkpoints at the simulator timestep and a positive reference count")
    output.mkdir(parents=True, exist_ok=False)
    (output / "source").mkdir()
    sources = [Path(__file__), Path("rl/quadrotor_ppo.py"), Path("envs/quadrotor_smooth.py"),
               Path("envs/quadrotor_desired_thrust.py"), Path("envs/quadrotor_double_thrust.py"),
               Path("envs/quadrotor_thrust_rate.py"), Path("experiments/quadrotor_comparison.py"),
               Path("experiments/diagnose_thrust_inputs.py"), Path("experiments/quadrotor_thrust_preview.py"),
               Path("envs/quadrotor_envs.py"), Path("requirements-quadrotor-lock.txt")]
    for path in sources:
        shutil.copy2(path, output / "source" / path.name)
    metadata = dict(status="running", reference_count=count, reference_seed=seed, start_seed=80103,
                    dt=DT, horizon=saved["horizon"], seeds=saved["seeds"], checkpoint_steps=saved["config"]["total_steps"],
                    checkpoint_run=str(checkpoint_run.resolve()),
                    checkpoint_manifest_sha256=digest(checkpoint_run / "manifest.json"),
                    checkpoint_sha256={}, reference_sha256={}, source_sha256={p.name: digest(p) for p in sources})
    write_json(output / "manifest.json", metadata)
    banks, summaries, trials, error_series, ref_stats = {}, [], [], {}, []
    starts = None
    for index, case in enumerate(CASE_NAMES):
        bank, jets = random_reference(case, count, saved["horizon"], seed + index)
        banks[case] = bank
        case_dir = output / case
        case_dir.mkdir()
        archive = case_dir / "references.npz"
        np.savez_compressed(archive, **bank._asdict(), thrust_jet=np.asarray(jets))
        metadata["reference_sha256"][case] = digest(archive)
        ref_stats.append(dict(case=case, force_rms_offset_N=float(jnp.sqrt(jnp.mean((jets[..., 0] - HOVER)**2))),
            rate_rms_Nps=float(jnp.sqrt(jnp.mean(jets[..., 1]**2))),
            curvature_rms_Nps2=float(jnp.sqrt(jnp.mean(jets[..., 2]**2))),
            mean_terminal_displacement_m=float(jnp.mean(jnp.linalg.norm(bank.pos[:, -1] - bank.pos[:, 0], axis=1)))))
        for variant in (B_PRIME, L):
            env = make_environment(bank, variant, reward_profile=saved["reward_profile"])
            first = checkpoint_run / variant / f"seed_{saved['seeds'][0]}"
            model, _, config = load_policy(first)
            env.scale = jnp.asarray(config["observation_scale"])
            actor = MaskedActor(model, jnp.zeros(env.obs_size))
            evaluator = make_evaluator(env, actor, seed=metadata["start_seed"], record_state=True)
            conditions = ("full", "masked") if case == "smooth_random" and variant == L else ("full",)
            for seed_id in saved["seeds"]:
                checkpoint = checkpoint_run / variant / f"seed_{seed_id}"
                _, params, check = load_policy(checkpoint)
                if any(check[k] != config[k] for k in ("observation_scale", "observation_size", "reward_profile", "steps")):
                    raise ValueError(f"Mismatched checkpoint config {checkpoint}")
                metadata["checkpoint_sha256"][f"{variant}/seed_{seed_id}"] = digest(checkpoint / "policy.msgpack")
                for condition in conditions:
                    mask = np.zeros(env.obs_size, dtype=bool)
                    if variant == L and (case == "paper_sampler" or condition == "masked"):
                        mask[list(DERIVATIVE_CHANNELS)] = True
                    metrics, frames = jax.device_get(evaluator((params, mask)))
                    current_start = [frames[f"initial_{name}"] for name in ("position", "velocity", "rotation", "omega")]
                    if starts is None:
                        starts = current_start
                    for value, expected in zip(current_start, starts):
                        np.testing.assert_array_equal(value, expected)
                    if not np.isfinite(metrics).all():
                        raise FloatingPointError("Nonfinite rollout metrics")
                    destination = case_dir / variant / f"seed_{seed_id}"
                    destination.mkdir(parents=True, exist_ok=True)
                    np.savez_compressed(destination / f"rollout_{condition}.npz", **frames)
                    error_sq = np.sum((frames["position"] - frames["reference"])**2, axis=-1)
                    active = frames["active"]
                    np.testing.assert_allclose(np.sqrt((error_sq * active).sum(0) / active.sum(0)), metrics[:, 0], atol=3e-6)
                    error_series.setdefault((case, variant, condition), []).append(np.where(active, error_sq, np.nan))
                    v_z = frames["velocity"][..., 2] - np.asarray(bank.vel[:, 1:, 2]).T
                    vertical = np.sqrt(((v_z**2)*active).sum(0) / active.sum(0))
                    trials.extend(dict(case=case, variant=variant, condition=condition, seed=seed_id, trajectory=i,
                        vertical_velocity_rmse_mps=float(vertical[i]), **dict(zip(EVALUATION_COLUMNS, row.tolist())))
                        for i, row in enumerate(metrics))
    for key in error_series:
        rows = [r for r in trials if (r["case"], r["variant"], r["condition"]) == key]
        completed = [r for r in rows if not r["failed"]]
        summaries.append(dict(case=key[0], variant=key[1], condition=key[2], trials=len(rows), completed=len(completed),
            mean_position_rmse_cm=float(100 * np.mean([r["position_rmse_m"] for r in rows])),
            mean_velocity_rmse_mps=float(np.mean([r["velocity_rmse_mps"] for r in rows])),
            mean_vertical_velocity_rmse_mps=float(np.mean([r["vertical_velocity_rmse_mps"] for r in rows])),
            mean_attitude_rmse_deg=float(np.rad2deg(np.mean([r["attitude_rmse_rad"] for r in rows])))))
    write_csv(output / "summary.csv", summaries)
    write_csv(output / "rollout_trials.csv", trials)
    write_csv(output / "reference_statistics.csv", ref_stats)
    plot(output, banks, error_series)
    metadata.update(status="complete", paired_physical_starts=True)
    write_json(output / "manifest.json", metadata)
    report = ["# Random-control reference tracking", "",
              "The paper's released sampler draws independent truncated-normal offsets within ±0.001 per normalized rotor command "
              "around 0.6573. Here the draws drive the local simulator over 50 ms holds. L receives zero thrust derivatives on those holds.", "",
              "The separate smooth test draws common thrust offsets within ±0.12 N every 0.5 s, with small differential rotor offsets. "
              "Quintic interpolation makes thrust twice differentiable. The reference is integrated at 10 ms and L receives analytic derivatives.", "",
              f"Each row uses {count} references across {len(saved['seeds'])} saved seeds. The physical starting states are paired.", "",
              "![Reference thrust and tracking error](tracking.png)", "",
              "| Reference | Policy | Completed | Position RMSE [cm] | Velocity RMSE [m/s] | Vertical velocity RMSE [m/s] | Attitude RMSE [deg] |",
              "|---|---|---:|---:|---:|---:|---:|"]
    for r in summaries:
        report.append(f"| {CASE_NAMES[r['case']]} | {POLICY_NAMES[r['variant'], r['condition']]} | {r['completed']}/{r['trials']} | "
                      f"{r['mean_position_rmse_cm']:.2f} | {r['mean_velocity_rmse_mps']:.4f} | "
                      f"{r['mean_vertical_velocity_rmse_mps']:.4f} | {r['mean_attitude_rmse_deg']:.2f} |")
    report += ["", "Reproduce with a fresh output directory.", "", "```powershell",
               "..\\.venv\\Scripts\\python.exe -m experiments.quadrotor_random_control "
               f"--checkpoint-run '{checkpoint_run}' --count {count} --seed {seed} --output runs/se23_random_control_repeat", "```"]
    (output / "REPORT.md").write_text("\n".join(report) + "\n", encoding="utf-8")
    print(json.dumps(summaries, indent=2), flush=True)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, default=Path("runs/se23_random_control"))
    parser.add_argument("--count", type=int, default=48)
    parser.add_argument("--seed", type=int, default=71501)
    parser.add_argument("--checkpoint-run", type=Path, default=SOURCE)
    args = parser.parse_args()
    run(args.output, args.count, args.seed, checkpoint_run=args.checkpoint_run)
