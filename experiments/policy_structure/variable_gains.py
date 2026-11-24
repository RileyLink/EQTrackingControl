"""Identify local gains first, then fit their dependence on the L observation."""

from collections.abc import Sequence

import jax
import jax.numpy as jnp
import matplotlib.pyplot as plt
import numpy as np

from experiments.policy_structure.common import actor_requested_wrench, actor_wrench
from experiments.policy_structure.observation_lee import command_terms, unpack
from experiments.policy_structure.observation_sparse import (
    FEATURES, RIDGE, THRESHOLDS, choose_threshold, coefficients, ridge_system,
)


CONTEXT = (
    "1", "tilt", "Fd_minus_mg", "Fdot", "Fddot", "sp", "sv", "Ep_norm2", "Enu_norm2",
    "ep_xy2", "ev_xy2", "eR_d_norm2", "ew_d_norm2",
    "eR_c_x", "eR_c_y", "eR_c_z", "ew_c_x", "ew_c_y", "ew_c_z",
    "Omega_d_x", "Omega_d_y", "Omega_d_z", "Md_x", "Md_y", "Md_z", "sp*sp", "sv*sv",
)
CONTEXT_COLUMNS = tuple(FEATURES["broad"].index(name) for name in CONTEXT)
GAIN_NAMES = ("kp", "kv", *(f"{kind}_{axis}" for kind in ("kR", "kOmega") for axis in "xyz"))
STAGES = ("requested", "applied")
DIFFERENCE_STEP = .001
MAX_MOMENT_CONDITION = 60.
GAIN_TOLERANCE = .01


def gain_context(features):
    return features[..., np.asarray(CONTEXT_COLUMNS)]


def state_tangent(obs, delta):
    """Tangent coordinates use actual body axes and hold the planner jet fixed."""
    tracking = unpack(obs)
    x, y, z = delta[:3] / 2
    skew = jnp.array([[0., -z, y], [z, 0., -x], [-y, x, 0.]])
    inc = jnp.linalg.solve(jnp.eye(3) - skew, jnp.eye(3) + skew)
    rotation = tracking.rot @ inc
    velocity = inc.T @ tracking.body_vel
    rate_error = tracking.omega + delta[3:] - rotation.T @ tracking.omega_d
    return obs.at[:9].set(rotation.ravel()).at[12:15].set(velocity).at[15:18].set(rate_error)


def probes(data, observations, gains):
    """Use AD on the policy and subtract the observation-only feedforward jet."""
    scale, env = data.env.scale, data.env

    def wrenches(obs):
        channels = data.policy.from_observation(obs / scale)
        return jnp.stack((actor_requested_wrench(env, channels), actor_wrench(env, channels)))

    def local(obs):
        actor = jax.jacrev(wrenches)(obs)
        body_z = unpack(obs).rot[:, 2]
        thrust = -jnp.stack((actor[:, 0, 9:12] @ body_z, actor[:, 0, 14]), axis=-1)
        position_nudge = jnp.zeros_like(obs).at[9:12].set(body_z)
        velocity_nudge = jnp.zeros_like(obs).at[14].set(1.)
        directions = jnp.stack((position_nudge, velocity_nudge))
        h = DIFFERENCE_STEP
        numerical = jax.vmap(lambda d: -(wrenches(obs + h*d)[:, 0] - wrenches(obs - h*d)[:, 0]) / (2*h))(
            directions).T
        push = jax.jacfwd(state_tangent, argnums=1)(obs, jnp.zeros(6, dtype=obs.dtype))
        moments = actor[:, 1:] @ push

        def geometry(delta):
            terms = command_terms(unpack(state_tangent(obs, delta)), gains.position, gains.velocity)
            return terms.rotation_errors, terms.feedforward[1:]

        derivative, ff = jax.jacfwd(geometry)(jnp.zeros(6, dtype=obs.dtype))
        return thrust, numerical, derivative, ff, moments

    thrust, finite, derivative, ff, moments = map(np.asarray, jax.jit(jax.vmap(local))(observations))
    # One two-parameter local fit per moment output over all six tangent directions.
    columns = np.stack((-derivative[:, :3, :], -derivative[:, 3:, :]), axis=-1).astype(float)
    u, s, vh = np.linalg.svd(columns, full_matrices=False)
    valid = (s[..., 1] > .025) & (s[..., 0] < MAX_MOMENT_CONDITION * s[..., 1])
    inverse = (vh.swapaxes(-1, -2) * (1 / np.maximum(s, .025))[..., None, :]) @ u.swapaxes(-1, -2)
    moment = np.einsum("nijk,nsik->nsij", inverse, moments - ff[:, None])
    fit_error = np.linalg.norm(np.einsum("nijk,nsik->nsij", columns, moment) - (moments - ff[:, None]), axis=-1)
    fit_error /= np.maximum(np.linalg.norm(moments - ff[:, None], axis=-1), .01)
    return dict(gains=np.concatenate((thrust, moment.swapaxes(-1, -2).reshape(-1, 2, 6)), axis=-1),
                thrust_finite_difference=finite, moment_valid=valid, moment_projection_error=fit_error,
                moment_singular_values=s)


def constant_or_sparse(x, y, val_x, val_y, scale):
    """Select a compact gain equation by validation, with an explicit constant candidate."""
    system = ridge_system(x, y, scale, RIDGE)
    candidates = [coefficients(system, threshold, scale) for threshold in THRESHOLDS]
    constant = np.zeros(x.shape[1])
    constant[0] = np.mean(y)
    candidates[-1] = constant
    coefficient, threshold, path = choose_threshold(candidates, val_x, val_y, scale, GAIN_TOLERANCE)
    for row in path:
        row["candidate"] = "constant" if row["threshold"] == THRESHOLDS[-1] else "STLSQ"
    return coefficient, threshold, path


def identify(context, local, train, val, test):
    weights = {}
    paths, metrics = [], []
    for s, stage in enumerate(STAGES):
        weights[stage] = np.zeros((context.shape[1], len(GAIN_NAMES)))
        for i, name in enumerate(GAIN_NAMES):
            y = local["gains"][:, s, i]
            valid = np.isfinite(y) if i < 2 else local["moment_valid"][:, (i - 2) % 3] & np.isfinite(y)
            tr, va, te = (np.flatnonzero(split & valid) for split in (train, val, test))
            if min(len(tr), len(va), len(te)) < len(CONTEXT):
                raise ValueError(f"Too few well-conditioned gain probes for {name}")
            scale = max(.1, float(np.std(y[tr])))
            coef, _, path = constant_or_sparse(context[tr], y[tr], context[va], y[va], scale)
            weights[stage][:, i] = coef
            paths += [dict(stage=stage, gain=name, **row) for row in path]
            prediction = context[te] @ coef
            rmse = np.sqrt(np.mean((prediction - y[te])**2))
            constant = np.sqrt(np.mean((np.mean(y[tr]) - y[te])**2))
            low, mid, high = np.quantile(y[te], [.05, .5, .95])
            variance = np.var(y[te])
            metrics.append(dict(stage=stage, gain=name, train_samples=len(tr), test_samples=len(te),
                                p05=low, median=mid, p95=high, constant_rmse=constant, variable_rmse=rmse,
                                r2=1 - rmse**2 / variance if variance else None,
                                terms=int(np.count_nonzero(coef))))
    return weights, paths, metrics


def predict(context, errors, feedforward, weights, *, gain_bias=None):
    gains = context @ weights
    thrust = jnp.sum(errors[..., :2] * gains[..., :2], axis=-1)
    moment = jnp.sum((errors[..., 2:] * gains[..., 2:]).reshape(*errors.shape[:-1], 2, 3), axis=-2)
    action = feedforward - jnp.concatenate((thrust[..., None], moment), axis=-1)
    return action if gain_bias is None else action + context @ gain_bias


def fit_actions(context, errors, feedforward, target, train, val, scales: Sequence[float]):
    """Fit the same gain model to actions with a bias and a thrust-preview offset."""
    gains = np.zeros((context.shape[1], len(GAIN_NAMES)))
    bias = np.zeros((context.shape[1], 4))
    constant_gains, constant_bias = np.zeros_like(gains), np.zeros_like(bias)
    paths = []
    for i, name in enumerate(("f", "Mx", "My", "Mz")):
        pair = [0, 1] if i == 0 else [i + 1, i + 4]
        bias_cols = [CONTEXT.index(n) for n in (("1", "Fdot", "Fddot") if i == 0 else ("1",))]
        x = np.column_stack((-errors[:, pair[0], None] * context, -errors[:, pair[1], None] * context,
                             context[:, bias_cols]))
        y = target[:, i] - feedforward[:, i]
        coef, constant, path = sparse_path_with_base(x, y, train, val, scales[i], context.shape[1], len(bias_cols))
        gains[:, pair] = coef[:2*context.shape[1]].reshape(2, -1).T
        bias[bias_cols, i] = coef[2*context.shape[1]:]
        constant_gains[:, pair] = constant[:2*context.shape[1]].reshape(2, -1).T
        constant_bias[bias_cols, i] = constant[2*context.shape[1]:]
        paths += [dict(output=name, **row) for row in path]
    return (gains, bias), (constant_gains, constant_bias), paths


def sparse_path_with_base(x, y, train, val, scale, context_size, bias_count):
    system = ridge_system(x[train], y[train], scale)
    candidates = [coefficients(system, threshold, scale) for threshold in THRESHOLDS]
    # Constant feedback gains with the same bias and preview terms.
    base = np.r_[0, context_size, np.arange(x.shape[1] - bias_count, x.shape[1])]
    candidates[-1] = np.zeros(x.shape[1])
    candidates[-1][base] = np.linalg.lstsq(x[train][:, base], y[train], rcond=None)[0]
    fit, _, path = choose_threshold(candidates, x[val], y[val], scale, tolerance=.005)
    for row in path:
        row["candidate"] = "constant" if row["threshold"] == THRESHOLDS[-1] else "STLSQ"
    return fit, candidates[-1], path


def plot_local(output, context, local, coefficients_by_stage, test):
    fig, axes = plt.subplots(2, 4, figsize=(13, 7), layout="constrained")
    for i, (ax, name) in enumerate(zip(axes.flat, GAIN_NAMES)):
        values = local["gains"][:, 0, i]
        valid = test if i < 2 else test & local["moment_valid"][:, (i-2) % 3]
        estimate = (context @ coefficients_by_stage["requested"])[:, i]
        lo, hi = np.quantile(values[valid], [.01, .99])
        ax.scatter(values[valid], estimate[valid], s=6, alpha=.20, color="#176c93")
        ax.plot([lo, hi], [lo, hi], "k--", lw=1)
        ax.set(title=name, xlabel="Measured local gain", ylabel="Predicted gain", xlim=(lo, hi))
    fig.suptitle("State-dependent gains on held-out observations, before allocation")
    fig.savefig(output / "local_gains.png", dpi=155)
    plt.close(fig)
