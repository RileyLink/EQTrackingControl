"""Experiment 1. Differentiate the applied actor wrench and Lee feedback on SO(3)."""

import argparse

import jax
import jax.numpy as jnp
import matplotlib
matplotlib.use("Agg")
from matplotlib.lines import Line2D
import matplotlib.pyplot as plt
import numpy as np

from envs.quadrotor_desired_thrust import DesiredThrustExtension
from envs.quadrotor_smooth import at, hat
from experiments.policy_structure.common import (
    ACTION_NAMES, ACTION_SCALES, Dataset, actor_requested_wrench, actor_wrench, experiment_arguments,
    feasible_wrench, lee_controller, take,
)
from experiments.quadrotor_comparison import write_csv


COORDINATES = tuple(f"{name}_{axis}" for name in ("p_world", "v_world", "eta_body", "omega_body")
                    for axis in "xyz")


def perturb(state, delta):
    """Right Cayley chart on SO(3), with the same tangent as R exp(hat(eta))."""
    cross = hat(delta[6:9]) / 2
    eye = jnp.eye(3, dtype=state.rot.dtype)
    increment = jnp.linalg.solve(eye - cross, eye + cross)
    return state._replace(pos=state.pos + delta[:3], vel=state.vel + delta[3:6],
                          rot=state.rot @ increment, omega=state.omega + delta[9:12])


def sample_indices(data, samples_per_group, seed):
    rng = np.random.default_rng(seed)
    selected = [rng.choice(ids, min(samples_per_group, len(ids)), replace=False)
                for _, family, source, ids in data.groups() if family != "all" and source != "all"]
    return np.sort(np.concatenate(selected))


def teacher_functions(data):
    env = data.env
    actor = data.policy.controller(env)
    lee = lee_controller(env, data.gains)
    return {"RL": lambda state: actor_wrench(env, actor(state)),
            "Lee applied": lambda state: feasible_wrench(lee(state)),
            "Lee requested": lee}


def state_jacobians(states, wrench):
    def differentiate(state):
        return jax.jacfwd(lambda delta: wrench(perturb(state, delta)))(jnp.zeros(12, dtype=state.pos.dtype))
    return np.asarray(jax.jit(jax.vmap(differentiate))(states))


def actor_jacobians(data, states, *, requested=False):
    env = data.env
    decode = actor_requested_wrench if requested else actor_wrench

    def wrench(physical_observation):
        return decode(env, data.policy.from_observation(physical_observation / env.scale))

    def differentiate(state):
        def observation(delta):
            return env.observe(perturb(state, delta)) * env.scale
        zero = jnp.zeros(12, dtype=state.pos.dtype)
        jacobian = jax.jacrev(wrench)(observation(zero))
        return jacobian, jacobian @ jax.jacfwd(observation)(zero)

    obs, physical = jax.jit(jax.vmap(differentiate))(states)
    return np.asarray(obs), np.asarray(physical)


def tracking_coordinates(data, states, jacobians):
    """Convert attitude columns to fixed planner e_omega and planner e_R."""
    ref = at(data.env.bank, states.trajectory, states.step)
    Q = np.asarray(ref.rot).transpose(0, 2, 1) @ np.asarray(states.rot)
    q = jnp.einsum("nji,nj->ni", Q, ref.omega)  # R.T @ Rd @ omega_d
    fixed_rate = jacobians[..., 6:9] + jacobians[..., 9:12] @ np.asarray(jax.vmap(hat)(q))
    # d e_R / d eta for e_R = vee(Rd.T R - R.T Rd) / 2.
    mapping = (np.trace(Q, axis1=1, axis2=2)[:, None, None] * np.eye(3) - Q.transpose(0, 2, 1)) / 2
    singular_values = np.linalg.svd(mapping, compute_uv=False)
    valid = (singular_values[:, -1] >= .05) & (singular_values[:, 0] <= 50 * singular_values[:, -1])
    error = np.full_like(fixed_rate, np.nan)
    error[valid] = np.linalg.solve(mapping[valid].transpose(0, 2, 1),
                                   fixed_rate[valid].transpose(0, 2, 1)).transpose(0, 2, 1)
    return fixed_rate, error, valid


def projection(gradient, body_z):
    direction = body_z / np.linalg.norm(body_z, axis=1, keepdims=True)
    projected = np.sum(gradient * direction, axis=1)
    normal = np.linalg.norm(gradient - projected[:, None] * direction, axis=1)
    norm = np.linalg.norm(gradient, axis=1)
    cosine = np.divide(projected, norm, out=np.full_like(projected, np.nan), where=norm > 1e-7)
    fraction = np.divide(normal, norm, out=np.full_like(projected, np.nan), where=norm > 1e-7)
    return dict(projected_gain=projected, restoring_gain=-projected, orthogonal_norm=normal,
                orthogonal_fraction=fraction, cosine_with_body_z=np.clip(cosine, -1., 1.))


def moment_gains(block):
    gain = -np.trace(block, axis1=1, axis2=2) / 3
    residual = np.linalg.norm(block + gain[:, None, None] * np.eye(3), axis=(1, 2))
    return dict(restoring_gain=gain, isotropic_residual_frobenius=residual)


def finite_differences(function, states, h):
    deltas = h * jnp.eye(12, dtype=states.pos.dtype)

    def compare(state):
        center = function(state)
        plus = jax.vmap(lambda delta: function(perturb(state, delta)))(deltas)
        minus = jax.vmap(lambda delta: function(perturb(state, -delta)))(deltas)
        return ((plus - minus) / (2 * h)).T, ((plus + minus - 2 * center) / h).T

    return tuple(map(np.asarray, jax.jit(jax.vmap(compare))(states)))


def describe(values):
    values = values[np.isfinite(values)]
    if not len(values):
        return dict(defined=0, mean=None, std=None, p05=None, median=None, p95=None)
    lo, median, hi = np.quantile(values, [.05, .5, .95])
    return dict(defined=len(values), mean=np.mean(values), std=np.std(values), p05=lo, median=median, p95=hi)


def save_comparison(data, output, indices, jacobians, *, plots=False):
    """Save gain summaries, moment blocks, and per-state comparisons for any controllers."""
    states = take(data.states, indices)
    body_z = np.asarray(states.rot[..., 2])
    ref = at(data.env.bank, states.trajectory, states.step)
    state_errors = np.column_stack((np.linalg.norm(states.pos - ref.pos, axis=1),
                                   np.linalg.norm(states.vel - ref.vel, axis=1)))
    metrics, blocks, arrays = {}, {}, {"sample_index": indices, "body_z": body_z}
    direct = isinstance(data.env, DesiredThrustExtension)
    if direct:
        native_thrust = data.rows["rl_native"][indices, 0]
        tanh_slope = 4 * native_thrust * (1 - native_thrust)
        arrays["rl_thrust_relative_tanh_slope"] = tanh_slope
    for name, jac in jacobians.items():
        fixed, error, valid = tracking_coordinates(data, states, jac)
        arrays.update({f"{name}__state_tangent": jac, f"{name}__eta_fixed_eomega": fixed,
                       f"{name}__planner_eR": error, "planner_eR_valid": valid})
        blocks[name] = dict(eta_fixed_omega=jac[:, 1:, 6:9], eta_fixed_eomega=fixed[:, 1:],
                            planner_eR=error[:, 1:], planner_eomega=jac[:, 1:, 9:12])
        metrics[name] = {block: moment_gains(matrix) for block, matrix in blocks[name].items()}
        metrics[name].update(thrust_position=projection(jac[:, 0, :3], body_z),
                             thrust_velocity=projection(jac[:, 0, 3:6], body_z))
        if direct and name in ("RL", "RL requested"):
            for block in ("thrust_position", "thrust_velocity"):
                gain = metrics[name][block]["restoring_gain"]
                metrics[name][block]["restoring_gain_at_tanh_midpoint"] = np.divide(
                    gain, tanh_slope, out=np.full_like(gain, np.nan), where=tanh_slope > 1e-6,
                )
        discrepancy = (jac - jacobians["RL"]) / ACTION_SCALES[None, :, None]
        metrics[name]["comparison"] = {
            "scaled_frobenius_from_RL": np.linalg.norm(discrepancy, axis=(1, 2)),
        }
        arrays.update({f"{name}__{key}__{quantity}": value for key, entries in metrics[name].items()
                       for quantity, value in entries.items()})
    binding = np.max(np.abs(data.rows["rl_requested"][indices, 1:] - data.rows["rl_wrench"][indices, 1:]),
                     axis=1) > 1e-6
    np.savez_compressed(output / "feedback_jacobians.npz", **arrays, rl_torque_reallocated=binding)

    summary, moment_rows, allocator_rows = [], [], []
    for split, family, source, group in data.groups():
        subset = np.isin(indices, group)
        if not np.any(subset):
            continue
        for name, sections in metrics.items():
            base = dict(split=split, family=family, source=source, controller=name, samples=int(subset.sum()))
            for block, values in sections.items():
                summary += [dict(**base, block=block, metric=key, **describe(value[subset]))
                            for key, value in values.items()]
                if name in ("RL", "RL requested") and family == "all" and source == "all":
                    for active in (False, True):
                        matching = subset & (binding == active)
                        if np.any(matching):
                            allocator_rows += [dict(split=split, controller=name, reallocated=int(active),
                                                    samples=int(matching.sum()), block=block, metric=key,
                                                    **describe(value[matching])) for key, value in values.items()]
            for block, matrices in blocks[name].items():
                for i, moment in enumerate(ACTION_NAMES[1:]):
                    for j, axis in enumerate("xyz"):
                        moment_rows.append(dict(**base, block=block, moment=moment, axis=axis,
                                                **describe(matrices[subset, i, j])))
    write_csv(output / "jacobian_summary.csv", summary)
    write_csv(output / "moment_jacobians.csv", moment_rows)
    write_csv(output / "jacobian_by_allocator.csv", allocator_rows)
    if plots:
        plot_sensitivities(data, output, indices, state_errors, metrics, blocks)


def plot_sensitivities(data, output, indices, errors, metrics, blocks):
    test = data.sample_split[indices] == "test"
    families = data.sample_family[indices]
    sources = data.sample_source[indices]
    markers = dict(zip(np.unique(sources), ("o", "^", "s")))
    fig, axes = plt.subplots(2, 3, figsize=(13, 7), layout="constrained")
    for i, key in enumerate(("thrust_position", "thrust_velocity")):
        rl, lee = metrics["RL"][key], metrics["Lee applied"][key]
        for source, marker in markers.items():
            for family, color in zip(np.unique(families), ("C0", "C1", "C2", "C3")):
                show = test & (sources == source) & (families == family)
                axes[i, 0].scatter(errors[show, i], rl["restoring_gain"][show], alpha=.38,
                                   color=color, marker=marker, s=11,
                                   label=family if source == next(iter(markers)) else None)
                axes[i, 1].scatter(errors[show, i], rl["orthogonal_fraction"][show], alpha=.38,
                                   color=color, marker=marker, s=11)
        expected = [data.gains.position, data.gains.velocity][i]
        axes[i, 0].axhline(expected, color="k", ls="--", lw=1, label="Lee requested")
        for name in metrics:
            if name.startswith("Lee fit"):
                fitted = metrics[name][key]["restoring_gain"][test]
                axes[i, 0].axhline(np.median(fitted), ls=":", lw=1.3, label=f"{name} median")
        axes[i, 0].set_ylabel("RL restoring gain " + ("(N/m)" if i == 0 else "(N s/m)"))
        axes[i, 1].set_ylabel("RL orthogonal fraction")
        for axis in axes[i, :2]:
            axis.set_xlabel(("Position error norm (m)", "Velocity error norm (m/s)")[i])
        axes[i, 2].hist(rl["cosine_with_body_z"][test], bins=np.linspace(-1, 1, 26), alpha=.7, label="RL")
        axes[i, 2].hist(lee["cosine_with_body_z"][test], bins=np.linspace(-1, 1, 26), histtype="step",
                        lw=1.5, label="Lee applied")
        axes[i, 2].set(xlabel="Cosine with body thrust axis", ylabel="Samples")
    axes[0, 0].legend(fontsize=7)
    axes[0, 1].legend(handles=[Line2D([], [], color="#555555", marker=marker, linestyle="none",
                                     label=source.replace("_", " ")) for source, marker in markers.items()],
                      fontsize=7, loc="upper left")
    axes[0, 2].legend(fontsize=8)
    fig.suptitle("Local thrust feedback on test trajectories")
    fig.savefig(output / "thrust_jacobians.png", dpi=165)
    plt.close(fig)

    names = list(blocks)
    keys = ("eta_fixed_omega", "eta_fixed_eomega", "planner_eR", "planner_eomega")
    titles = ("Body tangent, fixed ω", "Body tangent, fixed eω", "Planner attitude error", "Planner rate error")
    fig, axes = plt.subplots(len(names), len(keys), figsize=(12, 2.8 * len(names)), layout="constrained")
    for column, (key, title) in enumerate(zip(keys, titles)):
        means = [np.nanmean(blocks[name][key][test], axis=0) for name in names]
        scale = max(1e-6, *(np.nanmax(np.abs(matrix)) for matrix in means))
        for row, (name, matrix) in enumerate(zip(names, means)):
            axis = axes[row, column]
            panel = axis.imshow(matrix, cmap="RdBu_r", vmin=-scale, vmax=scale)
            axis.set(xticks=range(3), xticklabels=list("xyz"), yticks=range(3), yticklabels=ACTION_NAMES[1:])
            for i, j in np.ndindex(matrix.shape):
                axis.text(j, i, f"{matrix[i, j]:.2f}", ha="center", va="center",
                          color="white" if abs(matrix[i, j]) > .65 * scale else "black", fontsize=8)
            if column == 0:
                axis.set_ylabel(name)
            if row == 0:
                axis.set_title(title, fontsize=10)
        fig.colorbar(panel, ax=axes[:, column], shrink=.68)
    fig.suptitle("Mean moment sensitivity on test trajectories")
    fig.savefig(output / "moment_jacobians.png", dpi=165)
    plt.close(fig)


def run(data, output, *, samples_per_group=128, seed=107):
    if samples_per_group < 1:
        raise ValueError("samples-per-group must be positive")
    output.mkdir(parents=True, exist_ok=False)
    indices = sample_indices(data, samples_per_group, seed)
    states = take(data.states, indices)
    observation, actor = actor_jacobians(data, states)
    request_observation, request_actor = actor_jacobians(data, states, requested=True)
    functions = teacher_functions(data)
    jacobians = {"RL": actor, "RL requested": request_actor,
                 **{name: state_jacobians(states, function) for name, function in functions.items() if name != "RL"}}
    if (not all(np.isfinite(jac).all() for jac in (observation, request_observation))
            or not all(np.isfinite(jac).all() for jac in jacobians.values())):
        raise ValueError("Non-finite physical or observation Jacobians")
    np.savez_compressed(output / "observation_jacobians.npz", sample_index=indices,
                        physical_observation_jacobian=observation,
                        requested_physical_observation_jacobian=request_observation,
                        normalized_observation_jacobian=observation * np.asarray(data.env.scale),
                        observation_scale=np.asarray(data.env.scale))
    save_comparison(data, output, indices, jacobians, plots=True)

    rng = np.random.default_rng(seed + 1)
    probes = np.sort(rng.choice(len(indices), min(48, len(indices)), replace=False))
    rows = []
    for name, function in functions.items():
        automatic = jacobians[name][probes]
        for h in (.001, .00025):
            numerical, one_sided = finite_differences(function, take(states, probes), h)
            for row, original in enumerate(indices[probes]):
                residual = numerical[row] - automatic[row]
                rows.append(dict(controller=name, sample_index=original, step=h,
                                 max_abs_error=np.max(np.abs(residual)),
                                 relative_frobenius=np.linalg.norm(residual)
                                 / max(1e-8, np.linalg.norm(automatic[row])),
                                 max_one_sided_difference=np.max(np.abs(one_sided[row]))))
    write_csv(output / "finite_difference.csv", rows)
    data.analysis_manifest(output, samples=len(indices), samples_per_group=samples_per_group, seed=seed,
                           action_order=ACTION_NAMES, state_tangent_order=COORDINATES,
                           state_tangent="Right Cayley chart, fixed reference and action history",
                           error_tangent="World e_p/e_v; planner e_R, holding planner e_omega fixed",
                           torque_allocator_tolerance_Nm=1e-6,
                           planner_eR_min_singular_value=.05, planner_eR_max_condition=50)
    print(f"Saved Jacobians for {len(indices):,} states to {output}", flush=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    experiment_arguments(parser)
    parser.add_argument("--samples-per-group", type=int, default=128)
    parser.add_argument("--seed", type=int, default=107)
    args = parser.parse_args()
    run(Dataset.load(args.dataset, args.checkpoint), args.output, samples_per_group=args.samples_per_group,
        seed=args.seed)


if __name__ == "__main__":
    main()
