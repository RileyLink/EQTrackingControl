"""Experiments 2–4 using the Lee-style law available from the L actor's inputs."""

import argparse
from pathlib import Path

import jax
import jax.numpy as jnp
import numpy as np
from scipy.optimize import least_squares

from envs.quadrotor_smooth import at
from experiments.policy_structure import variable_gains
from experiments.policy_structure.common import (
    ACTION_NAMES, ACTION_SCALES, ACTION_UNITS, Dataset, experiment_arguments, lee_controller, wrench_controller,
)
from experiments.policy_structure.lee_fit import fit, predict
from experiments.policy_structure.observation_lee import (
    L_VARIANT, command_terms, gain_design, request, unpack, world_heading_in_desired_frame,
)
from experiments.policy_structure.observation_sparse import FEATURES, RIDGE, THRESHOLDS, fit_library, vector_features
from experiments.policy_structure.predictions import project
from experiments.policy_structure.rollouts import run_closed_loop
from experiments.policy_structure.sparse_fit import bootstrap_samples
from experiments.quadrotor_comparison import write_csv, write_json

import matplotlib.pyplot as plt


WORLD = "Lee world heading"
NOMINAL = "Lee body heading"
SCALAR = "Gain fit scalar"
DIAGONAL = "Gain fit diagonal"
COUPLED = "Gain fit coupled"
ACTION_CONSTANT = "Constant gains plus offsets"
ACTION_VARIABLE = "Variable gains action fit"
LOCAL_VARIABLE = "Variable gains from derivatives"
COUPLED_BOUNDS = ([0., 0., 0., 0.], [1.2, 1.6, 25., 12.])


def linear_formula(names, weights, offset="0"):
    return offset + "".join(f" {value:+.8g}*({name})" for name, value in zip(names, weights) if value != 0.)


def coupled_gains(observations, target, train, val, nominal, scalar):
    """Refit the scalar attitude construction while minimizing physical-wrench error."""
    train_obs, val_obs = jnp.asarray(observations[train]), jnp.asarray(observations[val])
    teacher = jnp.asarray(target[train])
    scale = jnp.asarray(ACTION_SCALES, dtype=teacher.dtype)
    batch = jax.vmap(request, in_axes=(0, None))

    def residual(gains):
        return ((batch(train_obs, gains) - teacher) / scale).ravel()

    evaluate = jax.jit(residual)
    derivative = jax.jit(jax.jacfwd(residual))
    validation_predict = jax.jit(lambda gains: batch(val_obs, gains))
    starts = [tuple(nominal), scalar, (.30, .55, .8, .5), (.12, .24, .45, .28), (.4, .8, 1.8, 1.0)]
    trials = []
    for start in starts:
        initial = np.clip(start, np.asarray(COUPLED_BOUNDS[0]) + 1e-5, np.asarray(COUPLED_BOUNDS[1]) - 1e-5)
        convert = lambda gains: jnp.asarray(gains, dtype=teacher.dtype)
        result = least_squares(lambda gains: np.asarray(evaluate(convert(gains)), dtype=float), initial,
                               jac=lambda gains: np.asarray(derivative(convert(gains)), dtype=float),
                               bounds=COUPLED_BOUNDS, ftol=3e-6, xtol=3e-6, gtol=3e-6, max_nfev=100)
        predicted = np.asarray(validation_predict(convert(result.x)))
        score = np.sqrt(np.mean(((predicted - target[val]) / ACTION_SCALES)**2))
        trials.append(dict(start=initial.tolist(), gains=result.x.tolist(), training_cost=result.cost,
                           validation_rmse=score, success=bool(result.success), evaluations=result.nfev))
    best = min(trials, key=lambda trial: trial["validation_rmse"])
    return best["gains"], trials


def score_actions(data, output, requested, model_counts):
    applied = {name: np.asarray(project(jnp.asarray(value, dtype=data.env.scale.dtype)))
               for name, value in requested.items()}
    teacher = {"requested": data.rows["rl_requested"], "applied": data.rows["rl_wrench"]}
    score_rows, detail_rows = [], []
    for split, family, source, ids in data.groups():
        for name in requested:
            for stage, predicted in (("requested", requested[name]), ("applied", applied[name])):
                error = predicted[ids] - teacher[stage][ids]
                lee_difference = predicted[ids] - applied[WORLD][ids] if stage == "applied" else (
                    predicted[ids] - requested[WORLD][ids])
                by_output = np.sqrt(np.mean(error**2, axis=0))
                base = dict(model=name, stage=stage, split=split, family=family, source=source, samples=len(ids))
                score_rows.append(dict(**base, **dict(zip(ACTION_NAMES, by_output)),
                                       nrmse=np.sqrt(np.mean((error / ACTION_SCALES)**2)),
                                       nrmse_from_lee=np.sqrt(np.mean((lee_difference / ACTION_SCALES)**2)),
                                       terms=model_counts.get(name)))
                truth = teacher[stage][ids]
                sst = np.sum((truth - truth.mean(axis=0))**2, axis=0)
                for i, channel in enumerate(ACTION_NAMES):
                    detail_rows.append(dict(
                        **base, output=channel, unit=ACTION_UNITS[i], rmse=by_output[i],
                        nrmse=by_output[i] / ACTION_SCALES[i], bias=error[:, i].mean(),
                        mae=np.mean(np.abs(error[:, i])),
                        r2=1 - np.sum(error[:, i]**2) / sst[i] if sst[i] > 1e-12 else None,
                    ))
    write_csv(output / "action_errors.csv", detail_rows)
    write_csv(output / "scores.csv", score_rows)
    np.savez_compressed(output / "predictions.npz", rl_requested=teacher["requested"], rl_applied=teacher["applied"],
                        **{f"{name}__requested": value for name, value in requested.items()},
                        **{f"{name}__applied": value for name, value in applied.items()})
    return score_rows


def plot_scores(output, scores, paths):
    rows = [row for row in scores if row["split"] == "test" and row["family"] == row["source"] == "all"]
    names = [row["model"] for row in rows if row["stage"] == "requested"]
    stages = [{row["model"]: row["nrmse"] * 100 for row in rows if row["stage"] == stage}
              for stage in ("requested", "applied")]
    fig, axes = plt.subplots(1, 2, figsize=(13, 7), layout="constrained",
                             gridspec_kw={"width_ratios": [1.1, 1]})
    indices = np.arange(len(names))
    for i, (stage, values) in enumerate(zip(("Requested", "Applied"), stages)):
        axes[0].barh(indices + (i-.5)*.35, [values[name] for name in names], height=.32,
                     label=stage, color=("#377998", "#e7964d")[i])
    axes[0].set(yticks=indices, yticklabels=names, xlabel="Held-out normalized wrench RMSE (%)")
    axes[0].invert_yaxis()
    axes[0].legend()
    for model in ("Direct corrections", "Direct broad", "Residual corrections", "Residual broad"):
        values = [row for row in paths if row["model"] == model and row["output"] == "f"]
        axes[1].plot([row["terms"] for row in values], [row["validation_nrmse"] * 100 for row in values],
                     marker=".", lw=1.3, label=model)
        picked = next(row for row in values if row["selected"])
        axes[1].scatter(picked["terms"], picked["validation_nrmse"] * 100, marker="D", s=27)
    axes[1].set(xlabel="Active thrust terms", ylabel="Validation thrust RMSE (% of 1 N)", xscale="symlog")
    axes[1].legend(fontsize=8)
    for ax in axes:
        ax.grid(alpha=.22)
        ax.set_axisbelow(True)
    fig.savefig(output / "fit_and_sparsity.png", dpi=155)
    plt.close(fig)


def run(data, output, *, tolerance=.0025, bootstrap_count=12):
    if data.policy.config["variant"] != L_VARIANT:
        raise ValueError("This formulation expects the 27-channel L SE_2(3) observation")
    if tolerance < 0 or not np.isfinite(tolerance) or bootstrap_count < 1:
        raise ValueError("Use a finite nonnegative tolerance and at least one bootstrap")
    output = Path(output)
    output.mkdir(parents=True, exist_ok=False)
    env, nominal = data.env, data.gains
    obs = data.rows["observation"] * np.asarray(env.scale)
    target = data.rows["rl_requested"]
    train, val, test = (data.sample_split == split for split in ("train", "validation", "test"))
    model_counts = {}
    notes = [
        "f is in N. Mx, My and Mz are in N m. Terms use unscaled L observations.",
        "All equations give requested wrenches. Apply the repository's rotor feasibility map for simulation.",
        "Lee_body uses the nominal translation gains to construct its command and the desired body-forward heading.",
    ]
    bootstraps = list(bootstrap_samples(data, train, bootstrap_count, seed=468))

    def geom(observation):
        tracking = unpack(observation)
        return command_terms(tracking, nominal.position, nominal.velocity)

    def as_request(state_request):
        return wrench_controller(env, lambda state: state_request(env.observe(state) * env.scale))

    checks = jax.jit(jax.vmap(
        lambda obs, state: request(obs, nominal, world_heading_in_desired_frame(at(env.bank, state.trajectory, state.step)))
    ))(obs, data.states)
    equivalence = np.max(np.abs(np.asarray(checks) - data.rows["lee_requested"]), axis=0)
    if np.any(equivalence > 1e-4):
        raise ValueError(f"World-heading reformulation disagrees with Lee: {equivalence}")

    batch = jax.jit(jax.vmap(geom))(obs)
    batch_design = jax.jit(jax.vmap(gain_design))(batch)
    base = np.asarray(jax.jit(jax.vmap(lambda obs: request(obs, nominal)))(obs))
    requested = {WORLD: data.rows["lee_requested"], NOMINAL: base}
    controllers = {"RL": data.policy.controller(env), WORLD: wrench_controller(env, lee_controller(env, nominal)),
                   NOMINAL: as_request(lambda obs: request(obs, nominal))}
    models = {NOMINAL: dict(gains=list(nominal))}
    print("Fitting constant gains", flush=True)
    for diagonal, model in ((False, SCALAR), (True, DIAGONAL)):
        a, b, diagnostic = fit(*(np.asarray(x)[train] for x in batch_design), target[train], diagonal=diagonal)
        a_jax, b_jax = jnp.asarray(a), jnp.asarray(b)
        requested[model] = np.asarray(predict(*batch_design, a_jax, b_jax))
        models[model] = dict(kp=a[:3].tolist() if diagonal else a[0], kv=a[3:].tolist() if diagonal else a[3],
                             kR=b[:, 0].tolist() if diagonal else b[0], kOmega=b[:, 1].tolist() if diagonal else b[1],
                             diagnostics=diagnostic)
        controllers[model] = as_request(lambda obs, a=a_jax, b=b_jax: predict(*gain_design(geom(obs)), a, b))
        names = ("kp", "kv", "kR", "kOmega")
        notes.append(model + "\n" + "\n".join(f"{name} = {models[model][name]}" for name in names))
    scalar = [models[SCALAR][name] for name in ("kp", "kv", "kR", "kOmega")]
    coupled, trials = coupled_gains(obs, target, train, val, nominal, scalar)
    gains = jnp.asarray(coupled, dtype=env.scale.dtype)
    requested[COUPLED] = np.asarray(jax.jit(jax.vmap(lambda obs: request(obs, gains)))(obs))
    controllers[COUPLED] = as_request(lambda obs: request(obs, gains))
    models[COUPLED] = dict(kp=coupled[0], kv=coupled[1], kR=coupled[2], kOmega=coupled[3], multistart=trials)
    notes.append(f"{COUPLED}\n(kp, kv, kR, kOmega) = {coupled}")

    def features(obs):
        return vector_features(unpack(obs), geom(obs), "broad")

    broad = np.asarray(jax.jit(jax.vmap(features))(obs), dtype=np.float64)
    coefficient_rows, sparse_paths = [], []
    for library, names in FEATURES.items():
        print(f"Fitting direct and residual {library} libraries ({len(names)} columns)", flush=True)
        columns = broad[:, :len(names)]
        for mode in ("Direct", "Residual"):
            model = f"{mode} {library}"
            residual = mode == "Residual"
            bias = base if residual else np.zeros_like(base)
            coef, thresholds, paths, entries = fit_library(columns, target - bias, library, train, val,
                                                            bootstraps, tolerance)
            requested[model] = columns @ coef + bias
            model_counts[model] = int(np.count_nonzero(coef))
            models[model] = dict(library=library, mode=mode, feature_names=names, coefficients=coef.tolist(),
                                 thresholds=thresholds, terms_per_output=np.count_nonzero(coef, axis=0).tolist())
            sparse_paths += [dict(model=model, **row) for row in paths]
            coefficient_rows += [dict(model=model, **row) for row in entries]
            notes.append(model + "\n" + "\n".join(f"{channel} = " + linear_formula(
                names, coef[:, i], f"Lee_body_{channel}" if residual else "0")
                for i, channel in enumerate(ACTION_NAMES)))
            used = np.flatnonzero(np.any(coef, axis=1))
            weights = jnp.asarray(coef[used])

            def formula(obs, library=library, used=used, weights=weights, residual=residual):
                terms = geom(obs)
                row = vector_features(unpack(obs), terms, library)
                correction = row[used] @ weights
                return request(obs, nominal) + correction if residual else correction

            controllers[model] = as_request(formula)
    write_csv(output / "sparse_thresholds.csv", sparse_paths)
    write_csv(output / "sparse_coefficients.csv", coefficient_rows)

    print("Estimating local gains and fitting state dependence", flush=True)
    context = variable_gains.gain_context(broad)
    errors = np.column_stack((batch.thrust_terms[:, 1:], batch.moment_terms[:, :2].reshape(-1, 6)))
    ff = np.asarray(jax.vmap(lambda term: term.feedforward)(batch))
    local = variable_gains.probes(data, obs, nominal)
    local_weights, gain_paths, gain_metrics = variable_gains.identify(context, local, train, val, test)
    action_fit, constant_fit, action_paths = variable_gains.fit_actions(context, errors, ff, target, train, val,
                                                                        ACTION_SCALES)
    local_fit = (local_weights["requested"], np.zeros_like(action_fit[1]))
    write_csv(output / "local_gain_metrics.csv", gain_metrics)
    write_csv(output / "local_gain_thresholds.csv", gain_paths)
    write_csv(output / "variable_action_thresholds.csv", action_paths)
    np.savez_compressed(output / "local_gains.npz", **local, context=context,
                        **{f"{stage}_coefficients": value for stage, value in local_weights.items()})
    variable_gains.plot_local(output, context, local, local_weights, test)
    for stage, weights in local_weights.items():
        notes.append(f"Local gains measured from {stage} RL commands\n" + "\n".join(
            f"{name}(z) = {linear_formula(variable_gains.CONTEXT, weights[:, i])}"
            for i, name in enumerate(variable_gains.GAIN_NAMES)))

    for model, (weights, bias) in ((LOCAL_VARIABLE, local_fit), (ACTION_CONSTANT, constant_fit),
                                    (ACTION_VARIABLE, action_fit)):
        requested[model] = np.asarray(variable_gains.predict(context, errors, ff, weights, gain_bias=bias))
        model_counts[model] = int(np.count_nonzero(weights) + np.count_nonzero(bias))
        models[model] = dict(context=variable_gains.CONTEXT, gain_names=variable_gains.GAIN_NAMES,
                             gain_weights=weights.tolist(), bias_weights=bias.tolist())
        notes.append(model + "\n" + "\n".join(
            [f"{name}(z) = {linear_formula(variable_gains.CONTEXT, weights[:, i])}"
             for i, name in enumerate(variable_gains.GAIN_NAMES)] +
            [f"offset_{name}(z) = {linear_formula(variable_gains.CONTEXT, bias[:, i])}"
             for i, name in enumerate(ACTION_NAMES)]))
        w, b = jnp.asarray(weights), jnp.asarray(bias)

        def scheduled(obs, w=w, b=b):
            terms = geom(obs)
            context = variable_gains.gain_context(vector_features(unpack(obs), terms, "broad"))
            error = jnp.concatenate((terms.thrust_terms[1:], terms.rotation_errors))
            return variable_gains.predict(context, error, terms.feedforward, w, gain_bias=b)

        controllers[model] = as_request(scheduled)

    (output / "equations.txt").write_text("\n\n".join(notes) + "\n", encoding="utf-8")
    write_json(output / "models.json", models)
    scores = score_actions(data, output, requested, model_counts)
    plot_scores(output, scores, sparse_paths)
    print(f"Running paired closed-loop trials for {len(controllers)} controllers", flush=True)
    run_closed_loop(data, output, controllers)
    data.analysis_manifest(output, gains=list(nominal), action_target="RL after tanh and before rotor allocation",
                           variants=(L_VARIANT,), tolerance=tolerance, bootstrap_replicates=bootstrap_count,
                           sparse_ridge=RIDGE, sparse_thresholds=THRESHOLDS, coupled_gain_bounds=COUPLED_BOUNDS,
                           world_heading_max_abs_difference=equivalence.tolist(),
                           feature_counts={name: len(columns) for name, columns in FEATURES.items()})
    print(f"Saved all results to {output}", flush=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    experiment_arguments(parser)
    parser.add_argument("--tolerance", type=float, default=.0025)
    parser.add_argument("--bootstraps", type=int, default=12)
    args = parser.parse_args()
    run(Dataset.load(args.dataset, args.checkpoint), args.output, tolerance=args.tolerance,
        bootstrap_count=args.bootstraps)


if __name__ == "__main__":
    main()
