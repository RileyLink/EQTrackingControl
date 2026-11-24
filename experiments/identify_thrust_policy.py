"""Identify sparse collective-thrust laws in frozen quadrotor policies."""

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

from envs.quadrotor_smooth import ALLOCATION, GRAVITY, MASS, Reference, State, at, make_environment, rotation_exp
from envs.quadrotor_thrust_rate import MAX_THRUST
from experiments.diagnose_thrust_inputs import pre_action_states
from experiments.quadrotor_comparison import load_policy, write_csv, write_json
from experiments.quadrotor_thrust_preview import B_PRIME, L, held_thrust
from rl.quadrotor_ppo import command, make_evaluator


RUN = Path("runs/se23_paper_size_full")
HOVER = MASS * GRAVITY
NAMES = {B_PRIME: "B′", L: "L"}
LINEAR = ("bias", "e_z", "e_vz", "F_d - mg")
DERIVATIVES = ("Fdot_d", "Fddot_d")
EXTRA = ("1 - cos(tilt)", "e_z |e_z|", "e_vz |e_vz|", "dF |dF|", "e_z e_vz", "e_z dF", "e_vz dF",
         "e_x", "e_y", "e_vx", "e_vy", "e_Rx", "e_Ry", "e_Rz", "e_omega_z", "|e_p,xy|^2", "|e_v,xy|^2")


def digest(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def features(obs, derivatives):
    """Use unnormalized observations and express velocity error in desired axes."""
    r = obs[..., :9].reshape((*obs.shape[:-1], 3, 3))
    p = obs[..., 9:12]
    v = jnp.einsum("...ij,...j->...i", r, obs[..., 12:15])
    ez, evz = p[..., 2], v[..., 2]
    force = obs[..., 21] - HOVER
    tilt = r[..., 2, 2]
    orientation = .5 * jnp.stack([r[..., 2, 1] - r[..., 1, 2], r[..., 0, 2] - r[..., 2, 0],
                                  r[..., 1, 0] - r[..., 0, 1]], axis=-1)
    cols = [jnp.ones_like(ez), ez, evz, force]
    if derivatives:
        cols += [obs[..., 22], obs[..., 23]]
    cols += [1 - tilt, ez * abs(ez), evz * abs(evz), force * abs(force), ez * evz, ez * force, evz * force,
             p[..., 0], p[..., 1], v[..., 0], v[..., 1], orientation[..., 0], orientation[..., 1], orientation[..., 2],
             obs[..., 17], jnp.sum(p[..., :2]**2, axis=-1), jnp.sum(v[..., :2]**2, axis=-1)]
    return jnp.stack(cols, axis=-1), tilt


def probes(seed, count):
    """Independent physically feasible vertical force jets and tracking errors."""
    rng = np.random.default_rng(seed)
    thrust = HOVER + rng.uniform(-.20, .20, count)
    rate = rng.uniform(-.65, .65, count)
    curvature = rng.uniform(-4.5, 4.5, count)
    zeros = jnp.zeros((count, 1, 3))
    scalar_zeros = jnp.zeros((count, 1))
    pos = zeros.at[..., 2].set(2.)
    bank = Reference(pos, zeros, jnp.broadcast_to(jnp.eye(3), (count, 1, 3, 3)), zeros,
        jnp.broadcast_to(jnp.asarray(thrust)[:, None, None] / MAX_THRUST, (count, 1, 4)),
        zeros.at[:, 0, 2].set(thrust / MASS - GRAVITY), zeros.at[:, 0, 2].set(rate / MASS),
        zeros.at[:, 0, 2].set(curvature / MASS), scalar_zeros, scalar_zeros, scalar_zeros)
    magnitude = np.where(rng.random(count)[:, None] < .65, .45, 1.)
    def offset(limits):
        return jnp.asarray(magnitude * rng.uniform(-1., 1., (count, 3)) * np.asarray(limits))
    state = State(pos[:, 0] + offset([.22, .22, .22]), offset([.28, .28, .34]),
        jax.vmap(rotation_exp)(offset([.22, .22, .35])), offset([.13, .13, .13]),
        jnp.arange(count), jnp.zeros(count, dtype=jnp.int32), bank.action[:, 0])
    return bank, state


def rmse(x, y):
    return float(np.sqrt(np.mean((x-y)**2)))


def fit_sparse(x, y, val, truth):
    """Sequential thresholded least squares, thresholded in RMS contribution units."""
    scale = np.sqrt(np.mean(x**2, axis=0))
    normalized = x / scale
    candidates = []
    for threshold in (0., .00025, .0005, .001, .002, .003, .005, .008):
        keep = np.ones(x.shape[1], dtype=bool)
        while True:
            coef = np.zeros(x.shape[1])
            coef[keep] = np.linalg.lstsq(normalized[:, keep], y, rcond=None)[0]
            selected = keep & (np.abs(coef) >= threshold)
            selected[0] = True
            if np.array_equal(keep, selected):
                break
            keep = selected
        coef /= scale
        candidates.append((rmse(val @ coef, truth), int(keep.sum()), coef, threshold))
    best = min(row[0] for row in candidates)
    return min((row for row in candidates if row[0] <= best + .0005), key=lambda row: (row[1], row[0]))


def force_from_formula(matrix, tilt, coefficient):
    return jnp.clip((HOVER + matrix @ coefficient) / jnp.maximum(tilt, .85), 0., MAX_THRUST)


def teacher_force(bank, mean_force, state):
    ref = at(bank, state.trajectory, state.step)
    feedback = MASS * (3. * (state.pos[..., 2] - ref.pos[..., 2]) + 3. * (state.vel[..., 2] - ref.vel[..., 2]))
    return jnp.clip((mean_force[state.trajectory, state.step] - feedback)
                    / jnp.maximum(state.rot[..., 2, 2], .85), .02, .98)


class ReplaceThrust:
    """Run the same actor torques and allocator with a replacement collective-thrust law."""
    def __init__(self, env, force):
        self.bank, self.dt, self.horizon = env.bank, env.dt, env.horizon
        self.reset, self.observe = env.reset, env.observe
        self.advance, self.force = env.transition, force

    def transition(self, state, action):
        return self.advance(state, action.at[0].set(self.force(state) / MAX_THRUST))


def initial_states(frames, bank):
    count = len(bank.pos)
    return State(*(jnp.asarray(frames[f"initial_{field}"]) for field in ("position", "velocity", "rotation", "omega")),
                 jnp.arange(count), jnp.zeros(count, dtype=jnp.int32), bank.action[:, 0])


def summarize_rollout(family, frames, bank, variant, seed, case):
    active = frames["active"]
    pos_error = frames["position"] - np.asarray(bank.pos[:, 1:]).swapaxes(0, 1)
    vertical = frames["velocity"][..., 2] - np.asarray(bank.vel[:, 1:, 2]).T
    pos_rmse = np.sqrt((np.sum(pos_error**2, axis=-1) * active).sum(0) / active.sum(0))
    vel_rmse = np.sqrt(((vertical**2) * active).sum(0) / active.sum(0))
    return [dict(variant=variant, seed=seed, controller=case, family=("rate", "curvature")[int(family[i])],
        trajectory=i, position_rmse_cm=float(100*pos_rmse[i]), vertical_velocity_rmse_mps=float(vel_rmse[i]),
        completed=int(not frames["failed"][:, i].any())) for i in range(len(pos_rmse))]


def plot(output, samples):
    fig, axes = plt.subplots(1, 2, figsize=(9.7, 4.7), sharex=True, sharey=True, layout="constrained")
    for ax, (variant, values) in zip(axes, samples.items()):
        actor, sparse, teacher = values
        ax.scatter(actor, teacher, s=8, color="#d97706", alpha=.23, label="Analytical teacher")
        ax.scatter(actor, sparse, s=8, color="#0369a1", alpha=.43, label="Sparse equation")
        ax.plot([.35, .96], [.35, .96], "--", color="#475569", lw=1, label="Actor match")
        ax.set(xlabel="Neural actor thrust [N]", ylabel="Predicted thrust [N]", title=NAMES[variant],
               xlim=(.35, .96), ylim=(.35, .96), aspect="equal")
        ax.grid(alpha=.18)
    axes[0].legend(fontsize=8)
    fig.suptitle("Thrust commands on unseen vertical rollouts\nPaper-size actors · checkpoint seed 0 · same physical states")
    fig.savefig(output / "command_fit.png", dpi=180)
    plt.close(fig)


def run(output, *, fit_data="rollouts"):
    manifest = json.loads((RUN / "manifest.json").read_text(encoding="utf-8"))
    archive = RUN / "held_out_references.npz"
    if manifest["status"] != "complete" or digest(archive) != manifest["reference_sha256"]["held_out"]:
        raise ValueError("Expected the completed paper-size run and its original references")
    with np.load(archive) as data:
        bank = Reference(*(jnp.asarray(data[key]) for key in Reference._fields))
        family = data["family"]
    training_archive = RUN / "train_references.npz"
    if fit_data == "rollouts":
        if digest(training_archive) != manifest["reference_sha256"]["train"]:
            raise ValueError("Training reference archive changed")
        with np.load(training_archive) as data:
            labels = data["family"]
            train_bank = Reference(*(jnp.asarray(data[key][:len(labels)]) for key in Reference._fields))
        # Leave complete plus/minus pairs out for validation in each family.
        fit_tracks = np.ones(len(labels), dtype=bool)
        for group in np.unique(labels):
            fit_tracks[np.flatnonzero(labels == group)[-8:]] = False
    output.mkdir(parents=True, exist_ok=False)
    mean_force = held_thrust(bank)
    probe_banks = [probes(seed, n) for seed, n in ((74211, 8192), (74212, 2048), (74213, 2048))]
    coef_rows, fit_rows, local_gains, rollouts, sample_plot, hashes = [], [], [], [], {}, {}
    for variant in (B_PRIME, L):
        derivatives = variant == L
        names = LINEAR + (DERIVATIVES if derivatives else ()) + EXTRA
        base_size = len(LINEAR) + (len(DERIVATIVES) if derivatives else 0)
        env = make_environment(bank, variant, dt=manifest["dt"], reward_profile=manifest["reward_profile"])
        reference_config = None
        for seed in manifest["seeds"]:
            checkpoint = RUN / variant / f"seed_{seed}"
            model, params, config = load_policy(checkpoint)
            hashes[f"{variant}/seed_{seed}"] = digest(checkpoint / "policy.msgpack")
            if reference_config is None:
                reference_config = config
                env.scale = jnp.asarray(config["observation_scale"])
                probe_inputs = []
                for local_bank, states in probe_banks:
                    local = make_environment(local_bank, variant)
                    local.scale = env.scale
                    obs = jax.jit(jax.vmap(local.observe))(states)
                    matrix, tilt = features(obs * env.scale, derivatives)
                    probe_inputs.append((obs, np.asarray(matrix, dtype=np.float64), np.asarray(tilt)))
                if fit_data == "rollouts":
                    fit_env = make_environment(train_bank, variant, dt=env.dt, reward_profile=env.reward_profile)
                    fit_env.scale = env.scale
                    fit_eval = make_evaluator(fit_env, model, seed=71413, record_state=True)
                    fit_observe = jax.jit(jax.vmap(fit_env.observe))
            elif config["observation_scale"] != reference_config["observation_scale"]:
                raise ValueError("Observation scaling differs across seeds")

            def actor_force(obs):
                return np.asarray(command(model.apply(params, obs)[0])[:, 0] * MAX_THRUST)

            outputs = [actor_force(obs) for obs, _, _ in probe_inputs]
            targets = [force * tilt - HOVER for (_, _, tilt), force in zip(probe_inputs, outputs)]
            (_, x, _), (_, val, _), (_, test, test_tilt) = probe_inputs
            if fit_data == "rollouts":
                _, fit_frames = jax.device_get(fit_eval(params))
                fit_states = pre_action_states(fit_frames, train_bank)
                fit_obs = fit_observe(fit_states)
                fit_matrix, fit_tilt = features(fit_obs * env.scale, derivatives)
                fit_matrix = np.asarray(fit_matrix, dtype=np.float64)
                fit_force = fit_frames["rotor_command"][fit_frames["active"]] @ np.asarray(ALLOCATION[0])
                fit_target = fit_force * np.asarray(fit_tilt) - HOVER
                select = fit_tracks[fit_states.trajectory]
                x, val = fit_matrix[select], fit_matrix[~select]
                targets[0], targets[1] = fit_target[select], fit_target[~select]
            compact = np.zeros(x.shape[1])
            compact[:base_size] = np.linalg.lstsq(x[:, :base_size], targets[0], rcond=None)[0]
            _, term_count, sparse, threshold = fit_sparse(x, targets[0], val, targets[1])
            for method, coef in (("linear", compact), ("sparse", sparse)):
                coef_rows.extend(dict(variant=variant, seed=seed, method=method, term=name, coefficient=float(c))
                                 for name, c in zip(names, coef))

            nominal = np.zeros(env.obs_size, dtype=np.float32)
            nominal[:9], nominal[21] = np.eye(3).ravel(), HOVER
            perturbations = [("e_z", 11, .003, -3*MASS), ("e_vz", 14, .005, -3*MASS), ("F_d - mg", 21, .004, 1.)]
            if derivatives:
                perturbations += [("Fdot_d", 22, .03, env.dt/2), ("Fddot_d", 23, .15, env.dt**2/6)]
            offsets = np.zeros((len(perturbations), env.obs_size), dtype=np.float32)
            for i, (_, col, step, _) in enumerate(perturbations):
                offsets[i, col] = step
            difference = actor_force((nominal + offsets) / env.scale) - actor_force((nominal - offsets) / env.scale)
            local_gains.extend(dict(variant=variant, seed=seed, term=name, actor_gain=float(delta / (2*step)),
                                    teacher_gain=expected)
                               for (name, _, step, expected), delta in zip(perturbations, difference))

            with np.load(checkpoint / "rollouts_full.npz") as saved:
                frames = dict(saved)
            np.testing.assert_array_equal(frames["reference"], np.asarray(bank.pos[:, 1:]).swapaxes(0, 1))
            states = pre_action_states(frames, bank)
            obs = jax.jit(jax.vmap(env.observe))(states)
            matrix, tilt = features(obs * env.scale, derivatives)
            truth = frames["rotor_command"][frames["active"]] @ np.asarray(ALLOCATION[0])
            np.testing.assert_allclose(actor_force(obs), truth, atol=2e-6, rtol=0.)
            linear_test = np.asarray(force_from_formula(test, test_tilt, compact))
            sparse_test = np.asarray(force_from_formula(test, test_tilt, sparse))
            linear_actual = np.asarray(force_from_formula(matrix, tilt, compact))
            sparse_actual = np.asarray(force_from_formula(matrix, tilt, sparse))
            teacher_actual = np.asarray(jax.jit(jax.vmap(lambda state: teacher_force(bank, mean_force, state)))(states))
            fit_rows.append(dict(variant=variant, seed=seed, sparse_terms=term_count, threshold_mN=1000*threshold,
                linear_probe_rmse_mN=1000*rmse(linear_test, outputs[2]), sparse_probe_rmse_mN=1000*rmse(sparse_test, outputs[2]),
                linear_rollout_rmse_mN=1000*rmse(linear_actual, truth), sparse_rollout_rmse_mN=1000*rmse(sparse_actual, truth),
                teacher_rollout_rmse_mN=1000*rmse(teacher_actual, truth),
                actor_rollout_thrust_std_mN=1000*float(np.std(truth))))
            if seed == manifest["seeds"][0]:
                take = np.linspace(0, len(truth)-1, 1800, dtype=int)
                sample_plot[variant] = (truth[take], sparse_actual[take], teacher_actual[take])
            rollouts.extend(summarize_rollout(family, frames, bank, variant, seed, "actor"))
            starts = initial_states(frames, bank)

            def sparse_force(state):
                basis, alignment = features(env.observe(state) * env.scale, derivatives)
                return force_from_formula(basis, alignment, jnp.asarray(sparse))

            for case, law in (("sparse", sparse_force), ("teacher", lambda state: teacher_force(bank, mean_force, state))):
                test_policy = make_evaluator(ReplaceThrust(env, law), model, initial_states=starts, record_state=True)
                metrics, result = jax.device_get(test_policy(params))
                if not np.isfinite(metrics).all():
                    raise FloatingPointError(f"Nonfinite {case} metrics for {variant}, seed {seed}")
                rollouts.extend(summarize_rollout(family, result, bank, variant, seed, case))
            print(f"{variant} seed {seed}: {term_count} terms, sparse/teacher gap "
                  f"{fit_rows[-1]['sparse_rollout_rmse_mN']:.2f}/{fit_rows[-1]['teacher_rollout_rmse_mN']:.2f} mN", flush=True)

    summaries = []
    for variant in (B_PRIME, L):
        for case in ("actor", "sparse", "teacher"):
            selected = [r for r in rollouts if (r["variant"], r["controller"]) == (variant, case)]
            summaries.append(dict(variant=variant, controller=case, trials=len(selected), completed=sum(r["completed"] for r in selected),
                mean_position_rmse_cm=float(np.mean([r["position_rmse_cm"] for r in selected])),
                mean_vertical_velocity_rmse_mps=float(np.mean([r["vertical_velocity_rmse_mps"] for r in selected]))))
    write_csv(output / "coefficients.csv", coef_rows)
    write_csv(output / "command_fit.csv", fit_rows)
    write_csv(output / "local_gains.csv", local_gains)
    write_csv(output / "rollout_trials.csv", rollouts)
    write_csv(output / "rollout_summary.csv", summaries)
    plot(output, sample_plot)
    shutil.copy2(__file__, output / Path(__file__).name)
    write_json(output / "manifest.json", dict(status="complete", reference_sha256=digest(archive),
        checkpoint_sha256=hashes, source_sha256=digest(Path(__file__)), seeds=manifest["seeds"],
        fit_data=fit_data, training_reference_sha256=(digest(training_archive) if fit_data == "rollouts" else None),
        probe_sizes=[8192, 2048, 2048], probe_seeds=[74211, 74212, 74213], validation_tolerance_mN=.5,
        scope="Vertical references and near-hover perturbations. All rollout conditions use the original actor torque requests."))


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, default=Path("runs/se23_thrust_identification"))
    parser.add_argument("--fit-data", choices=("rollouts", "probes"), default="rollouts")
    args = parser.parse_args()
    run(args.output, fit_data=args.fit_data)
