"""Nested, observation-only libraries and validation-selected sparse fits."""

from itertools import combinations_with_replacement

import jax.numpy as jnp
import numpy as np

from envs.quadrotor_smooth import GRAVITY, MASS, vee
from experiments.policy_structure.common import ACTION_NAMES, ACTION_SCALES
from experiments.policy_structure.sparse_fit import normalize_features, stlsq


AXES = "xyz"
THRUST = ("Fd_Q33", "sp", "sv")
MOMENT = ("eR_c", "ew_c", "gyro", "transport_c", "alpha_ff_c")
EXACT = (*THRUST, *(f"{name}_{axis}" for name in MOMENT for axis in AXES))
CORRECTIONS = (
    "1", "Fdot", "Fddot", "Ep_x", "Ep_y", "Enu_x", "Enu_y",
    *(f"{name}_{axis}" for name in ("d_eR", "d_ew") for axis in AXES),
)
PRIMITIVES = (
    "Ep_x", "Ep_y", "Ep_z", "Enu_x", "Enu_y", "sv",
    *(f"{name}_{axis}" for name in ("eR_d", "ew_d", "Omega_d", "Md") for axis in AXES),
    "Fd_minus_mg", "Fdot", "Fddot", "tilt", "sp",
)
ADDITIONAL = (
    *(f"Omega_{axis}" for axis in AXES), "Ep_norm2", "Enu_norm2", "Ep_dot_Ev",
    "eR_d_norm2", "ew_d_norm2", "ep_xy2", "ev_xy2",
    *(f"dQ_{i}{j}" for i in range(1, 4) for j in range(1, 4) if (i, j) != (3, 3)),
)
PAIR_INDICES = np.array(list(combinations_with_replacement(range(len(PRIMITIVES)), 2)))
PAIR_NAMES = tuple(f"{PRIMITIVES[i]}*{PRIMITIVES[j]}" for i, j in PAIR_INDICES)
INTERACTION_ERRORS = (*THRUST[1:], *(f"{name}_{axis}" for name in MOMENT[:2] for axis in AXES))
INTERACTION_CONTEXT = (
    "Fd_minus_mg", "Fdot", "Fddot", "tilt", "Ep_norm2", "Enu_norm2", "eR_d_norm2", "ew_d_norm2",
)
INTERACTIONS = tuple((name, context) for name in INTERACTION_ERRORS for context in INTERACTION_CONTEXT
                     if f"{name}*{context}" not in PAIR_NAMES)
CUBIC = ("sp", "sv", *(f"{name}_{axis}" for name in ("eR_d", "ew_d") for axis in AXES))
BROAD_LINEAR = tuple(name for name in PRIMITIVES if name not in EXACT and name not in CORRECTIONS)
BROAD = (*BROAD_LINEAR, *ADDITIONAL, *PAIR_NAMES, *(f"{name}*{context}" for name, context in INTERACTIONS),
         *(f"{name}*abs({name})" for name in INTERACTION_ERRORS), *(f"{name}^3" for name in CUBIC))
FEATURES = {"exact": EXACT, "corrections": EXACT + CORRECTIONS, "broad": EXACT + CORRECTIONS + BROAD}
THRESHOLDS = (0., .0005, .001, .002, .0035, .005, .0075, .01, .015, .02, .03, .045, .065, .1, .15)
RIDGE = 1e-4


def vector_features(tracking, command, library):
    exact = jnp.concatenate((command.thrust_terms, command.moment_terms.ravel()))
    if library == "exact":
        return exact
    skew = vee(tracking.rot - tracking.rot.T) / 2
    corrections = jnp.concatenate((
        jnp.ones(1, dtype=exact.dtype), tracking.thrust_jet[1:], tracking.pos[:2], tracking.body_vel[:2],
        skew - command.moment_terms[0], tracking.rate_error - command.moment_terms[1],
    ))
    prefix = jnp.concatenate((exact, corrections))
    if library == "corrections":
        return prefix

    primitives = jnp.concatenate((
        tracking.pos, tracking.body_vel, skew, tracking.rate_error, tracking.omega_d, tracking.moment_d,
        tracking.thrust_jet - jnp.array([MASS * GRAVITY, 0., 0.]),
        jnp.array([1. - tracking.rot[2, 2], command.thrust_terms[1]]),
    ))
    additional = jnp.concatenate((
        tracking.omega,
        jnp.array([tracking.pos @ tracking.pos, tracking.body_vel @ tracking.body_vel,
                   tracking.pos @ tracking.vel, skew @ skew, tracking.rate_error @ tracking.rate_error,
                   tracking.pos[:2] @ tracking.pos[:2], tracking.body_vel[:2] @ tracking.body_vel[:2]]),
        (tracking.rot - jnp.eye(3)).ravel()[:8],
    ))
    columns = dict(zip((*PRIMITIVES, *ADDITIONAL, *EXACT), (*primitives, *additional, *exact)))
    left = jnp.stack([columns[name] for name, _ in INTERACTIONS])
    right = jnp.stack([columns[name] for _, name in INTERACTIONS])
    signed = jnp.stack([columns[name] for name in INTERACTION_ERRORS])
    cubic = jnp.stack([columns[name] for name in CUBIC])
    return jnp.concatenate((
        prefix, jnp.stack([columns[name] for name in BROAD_LINEAR]), additional,
        primitives[PAIR_INDICES[:, 0]] * primitives[PAIR_INDICES[:, 1]],
        left * right, signed * jnp.abs(signed), cubic**3,
    ))


def allowed_columns(library, axis):
    names = FEATURES[library]
    if library == "broad":
        return np.arange(len(names))
    if axis == 0:
        allowed = THRUST if library == "exact" else THRUST + CORRECTIONS[:7]
    else:
        i = AXES[axis - 1]
        allowed = tuple(f"{name}_{i}" for name in MOMENT)
        if library == "corrections":
            allowed += ("1", f"d_eR_{i}", f"d_ew_{i}")
            allowed += tuple(f"{name}_{j}" for name in MOMENT[:2] for j in AXES if j != i)
    return np.array([names.index(name) for name in allowed])


def ridge_system(x, y, scale, ridge=RIDGE):
    normalized, rms, active = normalize_features(x)
    gram = normalized.T @ normalized / len(x) + ridge * np.eye(x.shape[1])
    cross = normalized.T @ (y / scale) / len(x)
    return gram, cross, rms, active


def coefficients(system, threshold, scale):
    gram, cross, rms, active = system
    return stlsq(gram, cross, active, threshold) * scale / np.where(active, rms, 1.)


def choose_threshold(candidates, val_x, val_y, scale, tolerance):
    predictions = val_x @ np.stack(candidates, axis=1)
    errors = np.sqrt(np.mean(((predictions - val_y[:, None]) / scale)**2, axis=0))
    counts = np.count_nonzero(candidates, axis=1)
    feasible = np.flatnonzero(errors <= errors.min() + tolerance)
    best = min(feasible, key=lambda j: (counts[j], errors[j], -THRESHOLDS[j]))
    path = [dict(threshold=value, validation_nrmse=float(errors[i]), terms=int(counts[i]), selected=int(i == best))
            for i, value in enumerate(THRESHOLDS)]
    return candidates[best], THRESHOLDS[best], path


def sparse_path(x, y, val_x, val_y, scale, tolerance, *, ridge=RIDGE):
    system = ridge_system(x, y, scale, ridge)
    candidates = [coefficients(system, threshold, scale) for threshold in THRESHOLDS]
    return choose_threshold(candidates, val_x, val_y, scale, tolerance)


def fit_library(matrix, target, library, train, val, bootstraps, tolerance):
    names = FEATURES[library]
    coefs = np.zeros((len(names), 4))
    paths, entries = [], []
    thresholds = []
    for i, name in enumerate(ACTION_NAMES):
        cols = allowed_columns(library, i)
        x, y = matrix[:, cols], target[:, i]
        train_x, train_y = x[train], y[train]
        scale = ACTION_SCALES[i]
        fit, threshold, path = sparse_path(train_x, train_y, x[val], y[val], scale, tolerance)
        coefs[cols, i] = fit
        thresholds.append(threshold)
        paths.extend(dict(output=name, candidates=len(cols), **row) for row in path)
        draws = np.stack([coefficients(ridge_system(train_x[ids], train_y[ids], scale), threshold, scale)
                          for ids in bootstraps])
        rms = np.sqrt(np.mean(train_x**2, axis=0))
        for j, column in enumerate(cols):
            lo, hi = np.quantile(draws[:, j], [.1, .9])
            entries.append(dict(output=name, feature=names[column], coefficient=fit[j],
                                scaled_rms=fit[j] * rms[j] / scale,
                                selection_fraction=np.mean(draws[:, j] != 0.),
                                sign_agreement=np.mean(np.sign(draws[:, j]) == np.sign(fit[j])),
                                coefficient_p10=lo, coefficient_p90=hi))
    return coefs, thresholds, paths, entries
