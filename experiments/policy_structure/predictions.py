"""Offline physical-action scores and plots shared by the fitted policies."""

import jax
import jax.numpy as jnp
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from envs.quadrotor_smooth import at, attitude_error
from experiments.policy_structure.common import ACTION_NAMES, ACTION_SCALES, ACTION_UNITS, feasible_wrench
from experiments.quadrotor_comparison import write_csv


project = jax.jit(jax.vmap(feasible_wrench))


def save_predictions(data, output, requests):
    target = data.rows["rl_wrench"]
    applied = {name: np.asarray(project(jnp.asarray(request))) for name, request in requests.items()}
    rows = []
    for split, family, source, indices in data.groups():
        truth = target[indices]
        variance = np.sum((truth - np.mean(truth, axis=0))**2, axis=0)
        for name, predictions in applied.items():
            for stage, values in (("requested", requests[name]), ("applied", predictions)):
                error = values[indices] - truth
                rmse = np.sqrt(np.mean(error**2, axis=0))
                mae = np.mean(np.abs(error), axis=0)
                bias = np.mean(error, axis=0)
                for i, action in enumerate(ACTION_NAMES):
                    rows.append(dict(
                        controller=name, stage=stage, split=split, family=family, source=source,
                        samples=len(indices), output=action, unit=ACTION_UNITS[i],
                        rmse=rmse[i], mae=mae[i], bias=bias[i], action_scale=ACTION_SCALES[i],
                        nrmse=rmse[i] / ACTION_SCALES[i],
                        r2=1 - np.sum(error[:, i]**2) / variance[i] if variance[i] > 1e-12 else None,
                    ))
    write_csv(output / "action_errors.csv", rows)
    np.savez_compressed(output / "predictions.npz", rl=target,
                        **{f"{name}__requested": value for name, value in requests.items()},
                        **{f"{name}__applied": value for name, value in applied.items()})
    return applied


def plot_predictions(data, output, applied):
    ids = np.flatnonzero(data.sample_split == "test")
    rng = np.random.default_rng(103)
    ids = rng.choice(ids, min(len(ids), 2400), replace=False)
    truth = data.rows["rl_wrench"][ids]
    units = [f"{name} ({unit})" for name, unit in zip(ACTION_NAMES, ACTION_UNITS)]

    fig, axes = plt.subplots(2, 2, figsize=(9, 8), layout="constrained")
    for i, axis in enumerate(axes.flat):
        bounds = [truth[:, i], *(values[ids, i] for values in applied.values())]
        span = [min(map(np.min, bounds)), max(map(np.max, bounds))]
        axis.plot(span, span, "k--", lw=1)
        for name, values in applied.items():
            axis.scatter(truth[:, i], values[ids, i], s=7, alpha=.20, label=name, rasterized=True)
        axis.set(xlabel="RL " + units[i], ylabel="Predicted " + units[i])
    axes[0, 0].legend(markerscale=2, fontsize=8)
    fig.suptitle("Applied wrench on held-out inputs")
    fig.savefig(output / "predicted_vs_rl.png", dpi=165)
    plt.close(fig)

    state = data.states
    ref = at(data.env.bank, state.trajectory[ids], state.step[ids])
    attitude = np.asarray(jax.vmap(attitude_error)(state.rot[ids], ref.rot))
    errors = [np.linalg.norm(state.pos[ids] - ref.pos, axis=1),
              np.linalg.norm(state.vel[ids] - ref.vel, axis=1), attitude,
              np.linalg.norm(state.omega[ids], axis=1)]
    labels = ["Position error (m)", "Velocity error (m/s)", "Attitude error (rad)", "Angular speed (rad/s)"]
    fig, axes = plt.subplots(4, 4, figsize=(13, 10), layout="constrained")
    for i, row in enumerate(axes):
        for j, axis in enumerate(row):
            axis.axhline(0, ls="--", color="#333333", lw=.8)
            for name, values in applied.items():
                axis.scatter(errors[j], values[ids, i] - truth[:, i], s=6, alpha=.20, label=name, rasterized=True)
            if i == len(axes) - 1:
                axis.set_xlabel(labels[j])
            if j == 0:
                axis.set_ylabel("Residual " + units[i])
    axes[0, 0].legend(markerscale=2, fontsize=8)
    fig.suptitle("Applied wrench residuals on held-out inputs")
    fig.savefig(output / "residuals.png", dpi=165)
    plt.close(fig)
