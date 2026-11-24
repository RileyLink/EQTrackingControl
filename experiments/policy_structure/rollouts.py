"""Paired rollouts using the checkpoint environment's reset and transition."""

import math

import jax
import jax.numpy as jnp
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from envs.quadrotor_smooth import ALLOCATION, State, at, rotation_exp
from envs.quadrotor_thrust_rate import ThrustRateInputs, ThrustState
from experiments.policy_structure.common import (
    ACTION_NAMES, ACTION_SCALES, ACTION_UNITS, concatenate,
)
from experiments.quadrotor_comparison import write_csv


def reset_states(env, indices, seed):
    keys = jax.random.split(jax.random.PRNGKey(seed), len(env.bank.pos))
    indices = jnp.asarray(indices)
    states, _ = jax.vmap(env.reset)(keys[indices], indices)
    return states


def reference_states(env, indices, steps):
    """On-reference physical states with history from prior reference commands."""
    ref = at(env.bank, indices, steps)
    previous = env.bank.action[indices, jnp.maximum(steps - 1, 0)]
    states = State(ref.pos, ref.vel, ref.rot, ref.omega, indices, steps, previous)
    if isinstance(env, ThrustRateInputs):
        preceding = env.bank.action[indices, jnp.maximum(steps - 2, 0)]
        rate = ((previous - preceding) @ ALLOCATION[0]) / env.dt
        return ThrustState(*states, previous @ ALLOCATION[0], rate)
    return states


def perturb_states(states, rng, bounds, amplitudes=1.):
    """Offsets expressed in the nominal body frame, with an SO(3) update."""
    offsets = jnp.asarray(rng.uniform(-1., 1., (len(states.pos), 4, 3))
                          * np.asarray(bounds)[None, :, None] * amplitudes)
    pos, vel, rot, omega = offsets.transpose(1, 0, 2)
    rotate = lambda vector: jnp.einsum("nij,nj->ni", states.rot, vector)
    new_rot = states.rot @ jax.vmap(rotation_exp)(rot)
    new_omega = jnp.einsum("nji,njk,nk->ni", new_rot, states.rot, states.omega) + omega
    return states._replace(pos=states.pos + rotate(pos), vel=states.vel + rotate(vel),
                           rot=new_rot, omega=new_omega)


def simulate(env, controller, initial_states, *, steps=None, jit=True):
    """Record each active pre-action state and the corresponding transition."""
    def step(carry, _):
        states, active = carry
        actions = jax.vmap(controller)(states)
        next_states, _, _, failed, truncated, info = jax.vmap(env.transition)(states, actions)
        wrench = next_states.previous_action @ ALLOCATION.T
        errors = jnp.sqrt(jnp.stack([info[name + "_sq"] for name in
                                     ("position", "velocity", "attitude", "omega")], axis=-1))
        frame = dict(state=states, position=next_states.pos, errors=errors, wrench=wrench,
                     active=active, failed=active & failed, saturation=info["saturation"],
                     rotors=next_states.previous_action, slew_sq=info["command_rate_sq"])

        def freeze(new, old):
            shape = active.shape + (1,) * (old.ndim - active.ndim)
            return jnp.where(active.reshape(shape), new, old)

        next_states = jax.tree_util.tree_map(freeze, next_states, states)
        return (next_states, active & ~(failed | truncated)), frame

    def run(states):
        active = jnp.ones(len(states.pos), dtype=bool)
        return jax.lax.scan(step, (states, active), xs=None, length=env.horizon if steps is None else steps)[1]

    return (jax.jit(run) if jit else run)(initial_states)


STRESS_BOUNDS = (0.65, 0.55, 0.38, 0.4)  # m, m/s, rad, rad/s per component
RECOVERY_POSITION = .20
RECOVERY_ATTITUDE = .15
RECOVERY_HOLD = .5


def trial_metrics(frames, column, dt):
    active = frames["active"][:, column]
    errors = frames["errors"][active, column]
    wrench = frames["wrench"][active, column]
    finite = np.isfinite(errors).all() and np.isfinite(wrench).all()
    failed = bool(np.any(frames["failed"][:, column]))
    recovered = (errors[:, 0] <= RECOVERY_POSITION) & (errors[:, 2] <= RECOVERY_ATTITUDE)
    hold = math.ceil(RECOVERY_HOLD / dt)
    begins = (np.flatnonzero(np.convolve(recovered, np.ones(hold), "valid") >= hold)
              if len(errors) >= hold else [])
    rms = np.sqrt(np.mean(errors**2, axis=0)) if finite else [None] * 4
    return dict(
        failed=int(failed), nonfinite=int(not finite), survival_s=dt * len(errors),
        position_rmse_m=rms[0], velocity_rmse_mps=rms[1], attitude_rmse_rad=rms[2],
        omega_rmse_radps=rms[3], end_position_m=errors[-1, 0] if finite else None,
        end_attitude_rad=errors[-1, 2] if finite else None,
        recovered=int(len(begins) > 0), recovery_s=(int(begins[0]) + 1) * dt if len(begins) else None,
        rotor_saturation_fraction=np.mean(frames["saturation"][active, column]),
        rotor_command_rms=np.sqrt(np.mean(frames["rotors"][active, column]**2)) if finite else None,
        rotor_slew_rms=np.sqrt(np.mean(frames["slew_sq"][active, column])) if finite else None,
        scaled_wrench_rms=np.sqrt(np.mean(np.sum((wrench / ACTION_SCALES)**2, axis=1))) if finite else None,
        **{f"{name}_rms": value for name, value in zip(
            ACTION_NAMES, np.sqrt(np.mean(wrench**2, axis=0)) if finite else [None] * 4)},
    )


def run_closed_loop(data, output, controllers):
    """Run every policy on both saved-reset and bounded stressed test starts."""
    env = data.env
    indices = np.flatnonzero(data.split == "test")
    nominal = reset_states(env, indices, data.metadata["reset_seed"])
    exact = reference_states(env, jnp.asarray(indices), jnp.zeros(len(indices), dtype=jnp.int32))
    stressed = perturb_states(exact, np.random.default_rng(data.metadata["stress_seed"]), STRESS_BOUNDS)
    starts = dict(nominal=nominal, stressed=stressed)
    paired_starts = concatenate(starts.values())
    trials, traces = [], {}
    for name, controller in controllers.items():
        result = jax.tree_util.tree_map(np.asarray, simulate(env, controller, paired_starts))
        for case_index, case in enumerate(starts):
            offset = len(indices) * case_index
            for column, trajectory in enumerate(indices):
                trials.append(dict(controller=name, start=case, family=data.family[trajectory],
                                   trajectory=trajectory, **trial_metrics(result, offset + column, env.dt)))
            traces[name, case] = {key: value[:, offset:offset + len(indices)] for key, value in result.items()
                                  if key != "state"}
    write_csv(output / "closed_loop_trials.csv", trials)
    summary = []
    groups = {(row["controller"], row["start"], row["family"]) for row in trials}
    for name, case, family in sorted(groups):
        rows = [row for row in trials if (row["controller"], row["start"], row["family"]) == (name, case, family)]
        aggregate = {}
        for key in rows[0]:
            if key in ("controller", "start", "family", "trajectory"):
                continue
            values = [row[key] for row in rows if row[key] is not None]
            aggregate[key + ("_fraction" if key in ("failed", "nonfinite", "recovered") else "_mean")] = (
                np.mean(values) if values else None)
        summary.append(dict(controller=name, start=case, family=family, trajectories=len(rows), **aggregate))
    write_csv(output / "closed_loop_summary.csv", summary)
    np.savez_compressed(output / "closed_loop_traces.npz", trajectory=indices,
                        reference=np.asarray(env.bank.pos[indices, 1:]).swapaxes(0, 1),
                        **{f"initial__{case}__{field}": value for case, state in starts.items()
                           for field, value in state._asdict().items()},
                        **{f"{name}__{case}__{key}": value for (name, case), trace in traces.items()
                           for key, value in trace.items()})
    plot_closed_loop(data, output, indices, traces)


def plot_closed_loop(data, output, indices, traces):
    """Always plot the first test trajectory of each family at the stressed start."""
    time = data.env.dt * (np.arange(data.env.horizon) + 1)
    labels = ("Position error (m)", "Attitude error (rad)",
              *(f"{name} ({unit})" for name, unit in zip(ACTION_NAMES, ACTION_UNITS)))
    families, columns = np.unique(data.family[indices], return_index=True)
    paths = plt.figure(figsize=(5 * len(columns), 4.6), layout="constrained")
    for panel, (family, column) in enumerate(zip(families, columns), start=1):
        fig, axes = plt.subplots(3, 2, figsize=(10, 8), sharex=True, layout="constrained")
        path = paths.add_subplot(1, len(columns), panel, projection="3d")
        ref = np.asarray(data.env.bank.pos[indices[column]])
        path.plot(*ref.T, "k--", lw=1, label="Reference")
        for (name, case), trace in traces.items():
            if case != "stressed":
                continue
            active = trace["active"][:, column]
            errors = trace["errors"][active, column]
            values = np.column_stack((errors[:, [0, 2]], trace["wrench"][active, column]))
            for axis, label, signal in zip(axes.flat, labels, values.T):
                axis.plot(time[active], signal, lw=1.1, label=name)
                axis.set_ylabel(label)
            path.plot(*trace["position"][active, column].T, lw=1.1, label=name)
        for axis in axes[-1]:
            axis.set_xlabel("Time (s)")
        axes[0, 0].legend(fontsize=8, ncol=2)
        fig.suptitle(f"Stressed start · {family} · test trajectory {indices[column]}")
        fig.savefig(output / f"closed_loop_{family}.png", dpi=150)
        plt.close(fig)
        path.set(title=family, xlabel="x (m)", ylabel="y (m)", zlabel="z (m)")
    paths.axes[0].legend(fontsize=7, loc="upper left")
    paths.savefig(output / "closed_loop_paths.png", dpi=150)
    plt.close(paths)
