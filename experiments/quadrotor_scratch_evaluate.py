"""Evaluate scratch-trained policies and fit the exact Lee basis at each checkpoint."""

import argparse
from collections import defaultdict
from pathlib import Path
from time import perf_counter

import jax
import jax.numpy as jnp
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from envs.quadrotor_smooth import at
from experiments.policy_structure.common import (
    ACTION_NAMES, ACTION_SCALES, Policy, actor_requested_wrench, lee_controller, wrench_controller,
)
from experiments.policy_structure.lee import DEFAULT_GAINS
from experiments.policy_structure.observation_lee import command_terms, unpack
from experiments.policy_structure.observation_sparse import FEATURES, allowed_columns, vector_features
from experiments.policy_structure.rollouts import reference_states, simulate, trial_metrics
from experiments.quadrotor_comparison import write_csv
from experiments.quadrotor_scratch import hash_file, load_bank, read_json, save_json, timestamp
from experiments.quadrotor_thrust_preview import B_PRIME, L, held_thrust
from experiments.quadrotor_yaw_practice import FAMILIES, YAW_MODES, lateral_start, yaw_start
from rl.quadrotor_ppo import command


CASES = ("nominal", *YAW_MODES, "lateral")
BASELINE = "Lee"
LABELS = {B_PRIME: "B′", L: "L", BASELINE: "Lee"}
COLORS = {B_PRIME: "#246b93", L: "#bf591f", BASELINE: "#353a46"}
FIT_MODELS = ("Lee nominal", "Lee exact basis fit")


def evaluation_starts(env, families, seed):
    groups = [(np.arange(len(families)), 0)]
    groups.extend((np.flatnonzero(families == "yaw"), mode) for mode in range(1, 4))
    groups.append((np.flatnonzero(families == "combined"), 4))
    trajectories = np.concatenate([indices for indices, _ in groups])
    modes = np.concatenate([np.full(len(indices), mode) for indices, mode in groups])
    keys = jax.random.split(jax.random.PRNGKey(seed), len(trajectories))

    def reset(key, index, mode):
        normal, perturbation = jax.random.split(key)
        state, _ = env.reset(normal, index)
        ref = at(env.bank, index, 0)
        return jax.lax.switch(
            mode, (lambda: state, *(lambda i=i: yaw_start(state, ref, perturbation, i) for i in range(3)),
                   lambda: lateral_start(state, ref, perturbation)),
        )

    return jax.vmap(reset)(keys, jnp.asarray(trajectories), jnp.asarray(modes)), np.asarray(CASES)[modes]


def trial_rows(model, seed, steps, env, starts, cases, families, frames):
    ids = np.asarray(starts.trajectory)
    refs = at(env.bank, jnp.asarray(ids)[None, :], jnp.arange(env.horizon)[:, None])
    states = frames["state"]
    relative = jnp.swapaxes(refs.rot, -1, -2) @ states.rot
    # Signed relative yaw in the desired frame, including a local yaw axis for tilted cases.
    yaw = np.asarray(jnp.arctan2(relative[..., 1, 0], relative[..., 0, 0]))
    rate_error = np.asarray(states.omega - (jnp.swapaxes(relative, -1, -2) @ refs.omega[..., None])[..., 0])
    tilt = np.asarray(jnp.arccos(jnp.clip(relative[..., 2, 2], -1., 1.)))
    host = jax.tree_util.tree_map(np.asarray, frames)
    rows = []
    for i, index in enumerate(ids):
        active = host["active"][:, i]
        angle_deg = np.rad2deg(yaw[active, i])
        rate = rate_error[active, i, 2]
        body_tilt = np.rad2deg(tilt[active, i])
        rms = lambda x: float(np.sqrt(np.mean(x**2)))
        # Overshoot is defined relative to the direction of the initial yaw error.
        direction = np.sign(angle_deg[0])
        overshoot = max(0., float(np.max(-direction * angle_deg))) if abs(angle_deg[0]) >= 5. else None
        within = (np.abs(angle_deg) <= 5.) & (np.abs(rate) <= .15)
        window = int(np.ceil(.5 / env.dt))
        recovered = (np.flatnonzero(np.convolve(within, np.ones(window), mode="valid") >= window)
                     if len(within) >= window else ())
        rows.append(dict(
            model=model, seed=seed, steps=steps, trajectory=int(index), family=families[index], start=cases[i],
            yaw_rmse_deg=rms(angle_deg), yaw_rate_rmse_radps=rms(rate), tilt_rmse_deg=rms(body_tilt),
            yaw_overshoot_deg=overshoot, yaw_recovered=int(len(recovered) > 0),
            yaw_recovery_s=float(recovered[0] * env.dt) if len(recovered) else None,
            **trial_metrics(host, i, env.dt),
        ))
    return rows


def preview_evaluator(model, env, bank_path, families, *, masked=False):
    ids = np.flatnonzero(families == "vertical")
    starts = reference_states(env, jnp.asarray(ids), jnp.zeros(len(ids), dtype=jnp.int32))
    truth = np.asarray(held_thrust(env.bank))[ids, 0]
    horizon = min(4, env.horizon)
    ref_p = np.asarray(env.bank.pos[ids, 1:horizon+1, 2]).T
    ref_v = np.asarray(env.bank.vel[ids, 1:horizon+1, 2]).T
    with np.load(bank_path, allow_pickle=False) as arrays:
        family = arrays["vertical_family"]
    assert np.array_equal(family[::2], family[1::2])

    @jax.jit
    def rollout(params):
        def controller(state):
            obs = env.observe(state)
            if masked:
                obs = obs.at[jnp.array([22, 23])].set(0.)
            return command(model.apply(params, obs)[0])

        return simulate(env, controller, starts, steps=min(horizon+1, env.horizon), jit=False)

    def measure(params):
        frames = rollout(params)
        first = np.asarray(frames["wrench"][0, :, 0])
        gap = first[::2] - first[1::2]
        expected_gap = truth[::2] - truth[1::2]
        # Position is post-action and velocity is the next frame's pre-action state.
        pos = np.asarray(frames["position"][:horizon, :, 2]) - ref_p
        vel = np.asarray(frames["state"].vel[1:, :, 2])
        vel = vel - ref_v[:len(vel)]
        active_pos = np.asarray(frames["active"][:horizon])
        active_vel = active_pos[:len(vel)]
        rows = []
        for name, pair_mask in (("all", np.ones(len(gap), dtype=bool)),
                                ("rate", family[::2] == 0), ("curvature", family[::2] == 1)):
            mask = np.repeat(pair_mask, 2)
            rows.append(dict(
                preview_family=name, pairs=int(pair_mask.sum()), seconds=horizon * env.dt,
                mean_expected_gap_mn=float(expected_gap[pair_mask].mean() * 1000),
                mean_output_gap_mn=float(gap[pair_mask].mean() * 1000),
                gap_rmse_mn=float(np.sqrt(np.mean((gap[pair_mask] - expected_gap[pair_mask])**2)) * 1000),
                correct_direction=float(np.mean(gap[pair_mask] > 1e-6)),
                first_interval_thrust_rmse_mn=float(np.sqrt(np.mean((first[mask] - truth[mask])**2)) * 1000),
                short_vertical_position_rmse_cm=float(np.sqrt(np.mean(pos[:, mask][active_pos[:, mask]]**2)) * 100),
                short_vertical_velocity_rmse_mps=float(np.sqrt(np.mean(vel[:, mask][active_vel[:, mask]]**2))),
            ))
        return rows

    return measure


def fit_states(env, families, seed):
    """Shared physical states, partitioned by whole reference trajectory."""
    starts, cases = evaluation_starts(env, families, seed)
    count = len(families)
    rng = np.random.default_rng(seed + 1)
    trajectory = np.repeat(np.arange(count), 30)
    step = rng.integers(0, env.horizon, size=len(trajectory))
    states = reference_states(env, jnp.asarray(trajectory), jnp.asarray(step))
    keys = jax.random.split(jax.random.PRNGKey(seed + 2), len(trajectory))
    refs = at(env.bank, states.trajectory, states.step)
    perturbed = jax.vmap(lateral_start)(states, refs, keys)
    # Retain near-reference states along with broader perturbations.
    mild = jax.vmap(lambda state, ref, key: lateral_start(state, ref, key, bounds=(.13, .11, .06, .085)))(
        states, refs, keys,
    )
    broad = np.arange(len(trajectory)) % 3 == 0
    states = jax.tree_util.tree_map(
        lambda a, b: jnp.where(jnp.asarray(broad).reshape((len(broad),) + (1,) * (a.ndim-1)), a, b),
        perturbed, mild,
    )
    sample_kind = np.where(broad, "broader", "near_reference")
    all_states = jax.tree_util.tree_map(lambda a, b: jnp.concatenate((a, b)), states, starts)
    sample_kind = np.r_[sample_kind, cases]
    split = np.zeros(count, dtype=bool)
    for family in FAMILIES:
        group = np.flatnonzero(families == family)
        split[rng.permutation(group)[len(group)//2:]] = True
    split = split[np.asarray(all_states.trajectory)]
    return all_states, sample_kind, split


def exact_terms(env, states):
    def evaluate(state):
        tracking = unpack(env.observe(state) * env.scale)
        command = command_terms(tracking, DEFAULT_GAINS.position, DEFAULT_GAINS.velocity)
        return vector_features(tracking, command, "exact")

    return np.asarray(jax.jit(jax.vmap(evaluate))(states), dtype=float)


def exact_fits(target, nominal, features, families, sample_kind, test, *, variant, seed, steps):
    prediction = np.zeros_like(target)
    coefficients = []
    for i, channel in enumerate(ACTION_NAMES):
        cols = allowed_columns("exact", i)
        basis = features[:, cols]
        # Axial symmetry makes the yaw gyroscopic term identically zero here.
        # Exclude columns at the float32 roundoff scale before the float64 solve.
        training = basis[~test]
        rms = np.sqrt(np.mean(training**2, axis=0))
        active = rms > 1e-7
        coef = np.zeros(len(cols))
        scaled, _, rank, _ = np.linalg.lstsq(training[:, active] / rms[active], target[~test, i], rcond=None)
        coef[active] = scaled / rms[active]
        prediction[:, i] = basis @ coef
        coefficients.extend(dict(model=variant, seed=seed, steps=steps, output=channel, rank=int(rank),
                                 feature=FEATURES["exact"][j], coefficient=float(c)) for j, c in zip(cols, coef))
    rows = []
    for name, pred in zip(FIT_MODELS, (nominal, prediction)):
        groups = [("all", test), *((family, test & (families == family)) for family in FAMILIES),
                  ("yaw_probes", test & np.isin(sample_kind, YAW_MODES))]
        for family, indices in groups:
            error = (pred - target)[indices]
            if len(error) == 0:
                continue
            rmse = np.sqrt(np.mean(error**2, axis=0))
            variance = np.var(target[indices], axis=0)
            for i, channel in enumerate(ACTION_NAMES):
                rows.append(dict(model=variant, seed=seed, steps=steps, comparison=name, family=family,
                                 output=channel, samples=len(error), rmse=float(rmse[i]),
                                 nrmse=float(rmse[i] / ACTION_SCALES[i]),
                                 r2=float(1. - rmse[i]**2/variance[i]) if variance[i] > 1e-12 else None))
    return rows, coefficients


def summarize(rows):
    groups = defaultdict(list)
    for row in rows:
        groups[row["model"], row["steps"], row["family"], row["start"]].append(row)
    keys = ("position_rmse_m", "attitude_rmse_rad", "yaw_rmse_deg", "yaw_rate_rmse_radps", "tilt_rmse_deg",
            "yaw_overshoot_deg", "yaw_recovered", "yaw_recovery_s", "survival_s",
            "rotor_saturation_fraction", "scaled_wrench_rms")
    summary = []
    for (model, steps, family, start), group in sorted(groups.items()):
        values = {key: [x[key] for x in group if x[key] is not None] for key in keys}
        summary.append(dict(model=model, steps=steps, family=family, start=start, trials=len(group),
                            failures=sum(x["failed"] for x in group),
                            **{key: float(np.mean(vals)) if vals else None for key, vals in values.items()}))
    return summary


def progress_plots(output, rollouts, preview, fits):
    fig, axes = plt.subplots(2, 2, figsize=(12, 7.5), layout="constrained")
    for variant in LABELS:
        entries = sorted({row["steps"] for row in rollouts if row["model"] == variant})
        series = []
        selections = (lambda r: r["start"] == "nominal",
                      lambda r: r["family"] == "yaw" and r["start"] != "nominal",
                      lambda r: r["family"] == "combined" and r["start"] == "lateral")
        metrics = ("position_rmse_m", "yaw_rmse_deg", "position_rmse_m")
        for key, include in zip(metrics, selections):
            series.append([np.mean([r[key] for r in rollouts if r["model"] == variant and r["steps"] == step
                                    and include(r) and r[key] is not None]) for step in entries])
        for i, (axis, values) in enumerate(zip(axes.flat, series)):
            x = np.array(entries) / 1e6
            if variant == BASELINE:
                axis.axhline(values[0] * (100 if i != 1 else 1), ls="--", color=COLORS[variant], label=LABELS[variant])
            else:
                axis.plot(x, np.array(values) * (100 if i != 1 else 1), "o-", color=COLORS[variant],
                          label=LABELS[variant])
    for variant, mode in ((B_PRIME, "full"), (L, "full"), (L, "masked")):
        selected = [r for r in preview if r["model"] == variant and r["inputs"] == mode
                    and r["preview_family"] == "all"]
        steps = sorted({r["steps"] for r in selected})
        axes[1, 1].plot(np.array(steps)/1e6, [np.mean([r["gap_rmse_mn"] for r in selected if r["steps"] == step])
                                             for step in steps], "o--" if mode == "masked" else "o-",
                        color=COLORS[variant], alpha=.62 if mode == "masked" else 1,
                        label="L, derivatives zeroed" if mode == "masked" else LABELS[variant])
    for axis, title, unit in zip(axes.flat,
                                ("Nominal tracking", "Yaw recovery", "Lateral recovery", "Vertical thrust preview"),
                                ("Position RMSE (cm)", "Yaw RMSE (degrees)", "Position RMSE (cm)", "Thrust gap RMSE (mN)")):
        axis.set(title=title, xlabel="Training steps per seed (millions)", ylabel=unit)
        axis.grid(alpha=.25)
        axis.legend(fontsize=9)
    fig.savefig(output / "tracking_progress.png", dpi=155)
    plt.close(fig)
    fig, axes = plt.subplots(2, 2, figsize=(11, 7), layout="constrained")
    for channel, axis in zip(ACTION_NAMES, axes.flat):
        for variant in (B_PRIME, L):
            rows = [r for r in fits if r["model"] == variant and r["output"] == channel and r["family"] == "all"]
            steps = sorted({r["steps"] for r in rows})
            for name, style in zip(FIT_MODELS, ("o--", "o-")):
                y = [100*np.mean([r["nrmse"] for r in rows if r["steps"] == step and r["comparison"] == name])
                     for step in steps]
                axis.plot(np.array(steps)/1e6, y, style, color=COLORS[variant], label=f"{LABELS[variant]}, {name}")
        axis.set(title=channel, xlabel="Training steps per seed (millions)", ylabel="Requested-action error (% of scale)")
        axis.grid(alpha=.25)
    axes[0, 0].legend(fontsize=8)
    fig.savefig(output / "lee_agreement_progress.png", dpi=155)
    plt.close(fig)


def evaluate(output):
    timer = perf_counter()
    manifest_path = output / "manifest.json"
    manifest = read_json(manifest_path)
    if manifest["status"] not in ("trained", "evaluating", "evaluation_failed", "complete"):
        raise ValueError("Finish both independent training runs before evaluating")
    manifest.update(status="evaluating", updated_utc=timestamp())
    manifest.pop("error", None)
    save_json(manifest_path, manifest)
    try:
        bank_path = output / "test_references.npz"
        if hash_file(bank_path) != manifest["reference_sha256"][bank_path.name]:
            raise ValueError("Held-out reference bank hash does not match")
        bank, families = load_bank(bank_path)
        steps = manifest["ppo"]["total_steps"]
        eval_seed = manifest["evaluation_seed"]
        variants, seeds = manifest["variants"], manifest["seeds"]
        final_tag = f"step_{steps:09d}"
        output_eval = output / "evaluation"
        output_eval.mkdir(exist_ok=True)
        reference_policy = Policy.load(output / L / f"seed_{seeds[0]}" / final_tag)
        env_l = reference_policy.environment(bank, manifest["dt"])
        starts, cases = evaluation_starts(env_l, families, eval_seed)
        states, sample_kind, test = fit_states(env_l, families, eval_seed)
        features = exact_terms(env_l, states)
        lee_request = lee_controller(env_l, DEFAULT_GAINS)
        nominal = np.asarray(jax.jit(jax.vmap(lee_request))(states))
        frames = simulate(env_l, wrench_controller(env_l, lee_request), starts)
        trials = trial_rows(BASELINE, -1, 0, env_l, starts, cases, families, frames)
        preview, scores, coefficients = [], [], []
        # Compile rollout and observation-query functions once per architecture variant.
        for variant in variants:
            first = Policy.load(output / variant / f"seed_{seeds[0]}" / final_tag)
            env = first.environment(bank, manifest["dt"])
            process = jax.jit(lambda params, obs: first.model.apply(params, obs)[0])
            observations = jax.jit(jax.vmap(env.observe))(states)

            def rollout(params, initial):
                actor = Policy(first.directory, first.model, params, first.config)
                return simulate(env, actor.controller(env), initial, jit=False)

            run_rollout = jax.jit(rollout)
            preview_modes = {mode: preview_evaluator(first.model, env, bank_path, families, masked=mode == "masked")
                             for mode in (("full", "masked") if variant == L else ("full",))}
            batch_decode = jax.jit(jax.vmap(lambda u: actor_requested_wrench(env, u)))
            for seed in seeds:
                folder = output / variant / f"seed_{seed}"
                for checkpoint in sorted(folder.glob("step_*")):
                    policy = Policy.load(checkpoint)
                    step = policy.config["steps"]
                    frames = run_rollout(policy.params, starts)
                    trials.extend(trial_rows(variant, seed, step, env, starts, cases, families, frames))
                    for mode, measure in preview_modes.items():
                        preview.extend(dict(model=variant, seed=seed, steps=step, inputs=mode, **row)
                                       for row in measure(policy.params))
                    target = np.asarray(batch_decode(command(process(policy.params, observations))))
                    error, weights = exact_fits(target, nominal, features, families[np.asarray(states.trajectory)],
                                                sample_kind, test, variant=variant, seed=seed, steps=step)
                    scores.extend(error)
                    coefficients.extend(weights)
                print(f"Evaluated {variant} seed {seed} across saved checkpoints", flush=True)
            jax.clear_caches()
        write_csv(output_eval / "rollout_trials.csv", trials)
        write_csv(output_eval / "rollout_summary.csv", summarize(trials))
        write_csv(output_eval / "vertical_preview.csv", preview)
        write_csv(output_eval / "lee_fit_errors.csv", scores)
        write_csv(output_eval / "lee_fit_coefficients.csv", coefficients)
        progress_plots(output_eval, trials, preview, scores)
        manifest.update(status="complete", updated_utc=timestamp(), completed_utc=timestamp(),
                        evaluation_seconds=perf_counter() - timer)
        save_json(manifest_path, manifest)
    except BaseException as error:
        manifest.update(status="evaluation_failed", updated_utc=timestamp(),
                        error=f"{type(error).__name__}: {error}")
        save_json(manifest_path, manifest)
        raise


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("run", type=Path)
    evaluate(parser.parse_args().run.resolve())
