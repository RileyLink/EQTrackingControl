"""Experiment 3. Identify direct and Lee-residual wrench laws with STLSQ."""

import argparse

import jax
import jax.numpy as jnp
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from envs.quadrotor_double_thrust import desired_thrust_jet
from envs.quadrotor_smooth import ALLOCATION, GRAVITY, MASS, at
from experiments.policy_structure.common import (
    ACTION_NAMES, ACTION_SCALES, ACTION_UNITS, Dataset, experiment_arguments,
    feasible_wrench, lee_controller, wrench_controller,
)
from experiments.policy_structure.lee import lee_terms, lee_wrench, planner_alpha, rotation_terms
from experiments.policy_structure.predictions import plot_predictions, project, save_predictions
from experiments.policy_structure.rollouts import run_closed_loop
from experiments.quadrotor_comparison import write_csv, write_json


LIBRARIES = ("polynomial", "geometric", "nonlinear")
THRESHOLDS = (0., .00025, .0005, .001, .002, .003, .005, .0075, .010, .015, .025, .04, .065, .1)
RIDGE = 1e-6


def _components(name, vector):
    return {f"{name}_{axis}": vector[i] for i, axis in enumerate("xyz")}


def candidate_terms(state, ref, gains, include_thrust_derivatives):
    terms = lee_terms(state, ref, gains)
    planner = rotation_terms(state, ref.rot, ref.omega, planner_alpha(ref))
    ep, ev = ref.rot.T @ terms.position, ref.rot.T @ terms.velocity
    ep_z, ev_z = terms.position @ terms.body_z, terms.velocity @ terms.body_z
    polynomial = dict(
        {"1": jnp.array(1., dtype=ep.dtype)}, **_components("ep_d", ep), **_components("ev_d", ev),
        **_components("eR_d", planner.attitude), **_components("ew_d", planner.rate),
    )
    for i, axis in enumerate("xyz"):
        for j, other in enumerate("xyz"):
            if i != j or axis == "z":
                polynomial[f"ep_d_{axis}*ev_d_{other}"] = ep[i] * ev[j]
        polynomial[f"eR_d_{axis}^2"] = planner.attitude[i]**2
        polynomial[f"omega_{axis}^2"] = state.omega[i]**2
    polynomial.update({"ep_d_z^2": ep[2]**2, "ev_d_z^2": ev[2]**2})

    f_d, f_dot, f_ddot = desired_thrust_jet(ref)
    f_delta = f_d - MASS * GRAVITY
    tau_d = (ALLOCATION @ ref.action)[1:]
    # Express near-duplicates as changes of frame or command. This keeps the
    # geometric corrections distinct from the polynomial and feedforward terms.
    geometric = {
        "ep_projection_change": ep_z - ep[2], "ev_projection_change": ev_z - ev[2],
        "b3_world_z_minus_1": terms.body_z[2] - 1,
        "Fd_projection_change": f_d * (ref.rot[:, 2] @ terms.body_z - 1),
        "Fd_minus_mg": f_delta,
        **_components("eR_command_change", terms.rotation.attitude - planner.attitude),
        **_components("ew_command_change", terms.rotation.rate - planner.rate),
        **_components("gyro", planner.gyro),
        **_components("transport_d", planner.transport),
        **_components("alpha_ff_d_minus_tau_d", planner.acceleration - tau_d),
        **_components("transport_command_change", terms.rotation.transport - planner.transport),
        **_components("alpha_ff_command_change", terms.rotation.acceleration - planner.acceleration),
        **_components("omega_d", ref.omega), **_components("tau_d", tau_d),
    }
    nonlinear = {
        "norm_ep_sq": ep @ ep, "norm_ev_sq": ev @ ev, "ep_dot_ev": ep @ ev,
        "ep_dot_b3^2": ep_z**2, "ev_dot_b3^2": ev_z**2, "ep_dot_b3*ev_dot_b3": ep_z * ev_z,
        "ep_dot_b3*Fd_minus_mg": ep_z * f_delta, "ev_dot_b3*Fd_minus_mg": ev_z * f_delta,
        "Fd_minus_mg^2": f_delta**2,
    }
    for i, axis in enumerate("xyz"):
        nonlinear[f"eR_d_{axis}*Fd_minus_mg"] = planner.attitude[i] * f_delta
        nonlinear[f"ew_d_{axis}*Fd_minus_mg"] = planner.rate[i] * f_delta
    if include_thrust_derivatives:
        nonlinear.update(Fdot_d=f_dot, Fddot_d=f_ddot, **{
            "ep_dot_b3*Fdot_d": ep_z * f_dot, "ev_dot_b3*Fdot_d": ev_z * f_dot,
            "ep_dot_b3*Fddot_d": ep_z * f_ddot, "ev_dot_b3*Fddot_d": ev_z * f_ddot,
        })
    return (polynomial, geometric, nonlinear), lee_wrench(terms, gains)


def normalize_features(x):
    x = np.asarray(x, dtype=np.float64)
    rms = np.sqrt(np.mean(x**2, axis=0))
    active = rms > 1e-8
    return x / np.where(active, rms, 1.), rms, active


def normal_equations(x, y):
    normalized, rms, active = normalize_features(x)
    gram = normalized.T @ normalized / len(x)
    gram += RIDGE * np.eye(x.shape[1])
    cross = normalized.T @ (y / ACTION_SCALES) / len(x)
    return gram, cross, rms, active


def stlsq(gram, cross, active, threshold):
    """Alternate normalized ridge least squares and hard coefficient thresholding."""
    keep = np.flatnonzero(active)
    coefficient = np.zeros(len(active))
    while len(keep):
        estimate = np.linalg.solve(gram[np.ix_(keep, keep)], cross[keep])
        selected = np.abs(estimate) >= threshold
        if selected.all():
            coefficient[keep] = estimate
            break
        keep = keep[selected]
    return coefficient


def physical_coefficients(gram, cross, rms, active, thresholds):
    coefficients = [stlsq(gram, cross[:, i], active, threshold)
                    for i, threshold in enumerate(thresholds)]
    return np.stack(coefficients, axis=1) * ACTION_SCALES / np.where(active, rms, 1.)[:, None]


def select_thresholds(x, target, val_x, val_target, tolerance, thresholds=THRESHOLDS):
    gram, cross, rms, active = normal_equations(x, target)
    candidates = []
    for threshold in thresholds:
        coefficient = physical_coefficients(gram, cross, rms, active, [threshold] * len(ACTION_NAMES))
        error = (val_x @ coefficient - val_target) / ACTION_SCALES
        candidates.append((coefficient, np.sqrt(np.mean(error**2, axis=0)),
                           np.count_nonzero(coefficient, axis=0)))
    errors = np.stack([error for _, error, _ in candidates])
    best = errors.min(axis=0)
    selected, path = [], []
    for i, name in enumerate(ACTION_NAMES):
        viable = np.flatnonzero(errors[:, i] <= best[i] + tolerance)
        winner = min(viable, key=lambda row: (candidates[row][2][i], errors[row, i], -thresholds[row]))
        selected.append(winner)
        path.extend(dict(output=name, threshold=threshold, validation_nrmse=error[i],
                         terms=int(terms[i]), selected=int(j == winner))
                    for j, (threshold, (_, error, terms)) in enumerate(zip(thresholds, candidates)))
    coefficient = np.stack([candidates[j][0][:, i] for i, j in enumerate(selected)], axis=1)
    return coefficient, [thresholds[j] for j in selected], rms, path


def bootstrap_samples(data, train, replicates, seed):
    rng = np.random.default_rng(seed)
    trajectory = data.rows["trajectory"][train]
    families = data.sample_family[train]
    indices = {i: np.flatnonzero(trajectory == i) for i in np.unique(trajectory)}
    groups = [np.unique(trajectory[families == name]) for name in np.unique(families)]
    for _ in range(replicates):
        yield np.concatenate([indices[i] for group in groups for i in rng.choice(group, len(group))])


def equations(model, names, coefficient, residual):
    lines = [model]
    for i, output in enumerate(ACTION_NAMES):
        intercept = f"Lee_applied_{output}" if residual else "0"
        terms = [f" {value:+.9g} * ({name})"
                 for name, value in zip(names, coefficient[:, i]) if value != 0]
        lines.append(f"{output}_requested = {intercept}" + "".join(terms))
    return "\n".join(lines)


def plot_coefficients(output, names, coefficient_rows):
    fig, axes = plt.subplots(len(names), 4, figsize=(16, 3.8 * len(names)), squeeze=False, layout="constrained")
    for r, name in enumerate(names):
        for i, action in enumerate(ACTION_NAMES):
            axis = axes[r, i]
            rows = [row for row in coefficient_rows if row["model"] == name and row["output"] == action]
            rows = sorted(rows, key=lambda row: abs(row["scaled_rms_contribution"]))[-8:]
            bars = axis.barh([row["feature"] for row in rows],
                             [row["scaled_rms_contribution"] for row in rows],
                             color=["#176e93" if row["bootstrap_selection_fraction"] >= .75 else "#a9c4ce"
                                    for row in rows])
            axis.axvline(0, color="#444444", linewidth=.8)
            axis.tick_params(axis="y", labelsize=8)
            axis.set(title=f"{name} · {action}", xlabel="Signed contribution / action scale")
            if not bars:
                axis.text(.5, .5, "No selected terms", transform=axis.transAxes, ha="center")
    fig.suptitle("Largest RMS term contributions · pale bars selected in fewer than 75% of bootstraps")
    fig.savefig(output / "selected_coefficients.png", dpi=165)
    plt.close(fig)


def run(data, output, *, tolerance=.005, bootstraps=12, seed=468):
    if not np.isfinite(tolerance) or tolerance < 0 or bootstraps < 1:
        raise ValueError("Tolerance must be finite and nonnegative, and bootstraps must be positive")
    output.mkdir(parents=True, exist_ok=False)
    env = data.env
    use_thrust_derivatives = getattr(env, "reference_thrust_size", 0) == 3

    def features(state):
        ref = at(env.bank, state.trajectory, state.step)
        return candidate_terms(state, ref, data.gains, use_thrust_derivatives)

    blocks, _ = jax.jit(jax.vmap(features))(data.states)
    train, val = data.sample_split == "train", data.sample_split == "validation"
    target, lee = data.rows["rl_wrench"], data.rows["lee_wrench"]
    samples = list(bootstrap_samples(data, train, bootstraps, seed))
    requests = {"Lee": data.rows["lee_requested"]}
    controllers = {"RL": data.policy.controller(env), "Lee": wrench_controller(env, lee_controller(env, data.gains))}
    all_terms, threshold_path, coefficient_rows, symbolic, models = {}, [], [], [], {}
    diagnostics = {}
    for block, library in zip(blocks, LIBRARIES):
        all_terms.update(block)
        names = list(all_terms)
        x = np.column_stack([np.asarray(all_terms[name], dtype=np.float64) for name in names])
        normalized, _, active = normalize_features(x[train])
        singular = np.linalg.svd(normalized[:, active], compute_uv=False)
        rank = int(np.sum(singular > singular[0] * max(normalized.shape) * np.finfo(float).eps))
        diagnostics[library] = dict(features=len(names), active_columns=int(active.sum()), rank=rank,
                                    condition=float(singular[0] / singular[-1]) if rank == active.sum() else None)
        for mode, offset in (("Direct", np.zeros_like(lee)), ("Residual", lee)):
            model = f"{mode} {library}"
            residual = mode == "Residual"
            y = target - offset
            coef, thresholds, rms, path = select_thresholds(x[train], y[train], x[val], y[val], tolerance)
            threshold_path.extend(dict(model=model, **row) for row in path)
            requests[model] = x @ coef + offset
            validation_prediction = np.asarray(project(jnp.asarray(requests[model][val])))
            validation_nrmse = float(np.sqrt(np.mean(((validation_prediction - target[val]) / ACTION_SCALES)**2)))
            models[model] = dict(library=library, mode=mode, feature_names=names,
                                 coefficients=coef.tolist(), thresholds=thresholds,
                                 validation_applied_nrmse=validation_nrmse,
                                 terms_per_output=np.count_nonzero(coef, axis=0).tolist(),
                                 terms=int(np.count_nonzero(coef)))
            symbolic.append(equations(model, names, coef, residual))

            train_x, train_y = x[train], y[train]
            draws = np.stack([physical_coefficients(*normal_equations(train_x[ids], train_y[ids]), thresholds)
                              for ids in samples])
            p10, p90 = np.quantile(draws, [.10, .90], axis=0)
            selected = np.mean(draws != 0, axis=0)
            sign_agreement = np.mean(np.sign(draws) == np.sign(coef), axis=0)
            for j, i in zip(*np.nonzero(coef)):
                coefficient_rows.append(dict(
                    model=model, output=ACTION_NAMES[i], output_unit=ACTION_UNITS[i], feature=names[j],
                    coefficient=coef[j, i], feature_train_rms=rms[j],
                    scaled_rms_contribution=coef[j, i] * rms[j] / ACTION_SCALES[i],
                    bootstrap_selection_fraction=selected[j, i], bootstrap_sign_agreement=sign_agreement[j, i],
                    bootstrap_coefficient_p10=p10[j, i], bootstrap_coefficient_p90=p90[j, i],
                ))
            sparse = {name: jnp.asarray(row) for name, row in zip(names, coef) if np.any(row)}

            def request(state, coefficients=sparse, residual=residual):
                terms, baseline = features(state)
                columns = {name: value for block in terms for name, value in block.items()}
                correction = sum((columns[name] * coef for name, coef in coefficients.items()),
                                 jnp.zeros(4, dtype=state.pos.dtype))
                return correction + feasible_wrench(baseline) if residual else correction

            controllers[model] = wrench_controller(env, request)

    winners = {}
    for mode in ("Direct", "Residual"):
        candidates = {name: value for name, value in models.items() if value["mode"] == mode}
        best = min(value["validation_applied_nrmse"] for value in candidates.values())
        winners[mode] = min((name for name, value in candidates.items()
                             if value["validation_applied_nrmse"] <= best + tolerance),
                            key=lambda name: (models[name]["terms"], models[name]["validation_applied_nrmse"]))
    write_json(output / "models.json", dict(selected_by_validation=winners, output_order=ACTION_NAMES,
                                           library_diagnostics=diagnostics, models=models))
    write_csv(output / "threshold_path.csv", threshold_path)
    write_csv(output / "coefficients.csv", coefficient_rows)
    (output / "symbolic_equations.txt").write_text(
        "Physical output units are N, N m, N m, N m. Apply rotor feasibility to each requested wrench.\n"
        "Lee_applied is the nominal Lee wrench after that same feasibility map.\n\n"
        + "\n\n".join(symbolic) + "\n", encoding="utf-8")
    applied = save_predictions(data, output, requests)
    plot_predictions(data, output, {name: applied[name] for name in ("Lee", *winners.values())})
    plot_coefficients(output, list(winners.values()), coefficient_rows)
    run_closed_loop(data, output, controllers)
    data.analysis_manifest(output, thresholds=THRESHOLDS, normalized_ridge=RIDGE, tolerance_nrmse=tolerance,
                           bootstrap_replicates=bootstraps, bootstrap_seed=seed,
                           uses_desired_thrust_derivatives=use_thrust_derivatives, library_order=LIBRARIES)
    print(f"Saved six symbolic laws to {output}", flush=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    experiment_arguments(parser)
    parser.add_argument("--tolerance", type=float, default=.005, help="Validation NRMSE allowance for sparsity")
    parser.add_argument("--bootstraps", type=int, default=12, help="Training trajectory resamples")
    parser.add_argument("--seed", type=int, default=468)
    args = parser.parse_args()
    run(Dataset.load(args.dataset, args.checkpoint), args.output, tolerance=args.tolerance,
        bootstraps=args.bootstraps, seed=args.seed)


if __name__ == "__main__":
    main()
