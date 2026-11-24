"""Teach and test desired thrust preview on matched vertical references."""

import argparse
from dataclasses import asdict
import hashlib
from pathlib import Path
import shutil
from time import perf_counter

from flax import serialization
import jax
import jax.numpy as jnp
import numpy as np
import optax

from envs.quadrotor_double_thrust import desired_thrust_jet
from envs.quadrotor_smooth import ALLOCATION, DT, GRAVITY, MASS, Reference, State, at, make_bank, make_environment, rotation_exp
from experiments.diagnose_thrust_inputs import MaskedActor
from experiments.quadrotor_comparison import load_policy, write_csv, write_json
from experiments.quadrotor_longer import independent_runners
from rl.quadrotor_ppo import PPOConfig, command, make_evaluator


B_PRIME = "B_prime_se23_wrench"
L = "L_se23_desired_thrust_extension"
SOURCE_RUN = Path("runs/se23_b_prime_2m")
FAMILIES = ("rate", "curvature")
SEEDS = (0, 1, 2)
HOVER = MASS * GRAVITY
DERIVATIVE_CHANNELS = (22, 23)
REWARD = "position_focus_v1"
ARCHITECTURES = {"pilot": (3, 64), "paper": (6, 128)}


def vertical_pairs(seed, count, horizon):
    """Each pair shares its initial position, velocity, attitude and wrench."""
    rng = np.random.default_rng(seed)
    family = np.repeat(np.arange(2), count)
    frequency = rng.uniform(np.where(family == 0, 1.5, 2.6), np.where(family == 0, 3.2, 4.8))
    force_budget = rng.uniform(.12, .26, size=len(family))
    center = rng.uniform(-1., 1., (len(family), 3)) + np.array([0., 0., 2.5])
    family, frequency, force_budget, center = map(lambda x: np.repeat(x, 2, axis=0),
                                                  (family, frequency, force_budget, center))
    sign = np.tile([1., -1.], len(family) // 2)
    w = jnp.asarray(frequency)[:, None]
    angle = w * (DT * jnp.arange(horizon + 1))
    s, s2, c, c2 = jnp.sin(angle), jnp.sin(2 * angle), jnp.cos(angle), jnp.cos(2 * angle)
    rate = (s - .5 * s2, c - c2, -s + 2 * s2, -c + 4 * c2, s - 8 * s2)
    curvature = (1 - c - .25 * (1 - c2), s - .5 * s2, c - c2, -s + 2 * s2, -c + 4 * c2)
    is_rate = jnp.asarray(family[:, None] == 0)
    amplitude = jnp.asarray((sign * force_budget)[:, None]) / (MASS * w**2 * jnp.where(is_rate, 3., 2.))
    jets = [amplitude * w**k * jnp.where(is_rate, a, b) for k, (a, b) in enumerate(zip(rate, curvature))]
    vectors = [jnp.zeros((*z.shape, 3)).at[..., 2].set(z) for z in jets]
    position = vectors[0] + jnp.asarray(center[:, None, :])
    zeros = jnp.zeros_like(jets[0])
    rot = jnp.broadcast_to(jnp.eye(3), (*zeros.shape, 3, 3))
    thrust = HOVER + MASS * jets[2]
    rotor = jnp.repeat(thrust[..., None], 4, axis=-1)
    ref = Reference(position, vectors[1], rot, jnp.zeros_like(position), rotor,
                    *vectors[2:], zeros, zeros, zeros)
    if float(rotor.min()) < 0 or float(rotor.max()) > 1:
        raise ValueError("Infeasible vertical reference")
    return ref, dict(family=family, frequency=frequency, sign=sign, force_budget=force_budget)


def exact_start(bank):
    return State(bank.pos[:, 0], bank.vel[:, 0], bank.rot[:, 0], bank.omega[:, 0],
                 jnp.arange(len(bank.pos)), jnp.zeros(len(bank.pos), dtype=jnp.int32), bank.action[:, 0])


def held_thrust(bank):
    # Integrating m * dv_z/dt = F_d - mg over a vertical reference interval.
    return HOVER + MASS * jnp.diff(bank.vel[..., 2], axis=1) / DT


def validate_pairs(bank, labels):
    thrust_jet = np.asarray(desired_thrust_jet(bank))[:, 0]
    for value in (bank.pos, bank.vel, bank.rot, bank.omega, bank.action):
        np.testing.assert_array_equal(value[::2, 0], value[1::2, 0])
    rate = labels["family"] == 0
    np.testing.assert_allclose(thrust_jet[rate, 1],
        (labels["sign"] * labels["force_budget"] * labels["frequency"])[rate], rtol=1e-6)
    np.testing.assert_allclose(thrust_jet[~rate, 2],
        (1.5 * labels["sign"] * labels["force_budget"] * labels["frequency"]**2)[~rate], rtol=1e-6)
    np.testing.assert_allclose(thrust_jet[rate, 2], 0., atol=1e-6)
    np.testing.assert_allclose(thrust_jet[~rate, 1], 0., atol=1e-6)
    force = np.asarray(held_thrust(bank))
    if not np.all((force > 0) & (force < 1)) or not np.all(force[::2, 0] > force[1::2, 0]):
        raise ValueError("Preview teacher must produce positive, feasible paired gaps")


def teacher_examples(env, vertical, count=8192):
    """Supervise vertical thrust and retain the shared initial actor's torque feedback."""
    keys = jax.random.split(jax.random.PRNGKey(18011), 6)
    track = jax.random.randint(keys[0], (count,), 0, len(vertical.pos))
    step = jax.random.randint(keys[1], (count,), 0, env.horizon)
    ref = jax.vmap(lambda i, t: at(vertical, i, t))(track, step)
    normal = lambda key: jax.random.normal(key, (count, 3))
    state = State(ref.pos + .07 * normal(keys[2]), ref.vel + .12 * normal(keys[3]),
                  ref.rot @ jax.vmap(rotation_exp)(.08 * normal(keys[4])),
                  ref.omega + .05 * normal(keys[5]), track, step, ref.action)
    force = held_thrust(vertical)[track, step]
    feedback = MASS * (3. * (state.pos[:, 2] - ref.pos[:, 2]) + 3. * (state.vel[:, 2] - ref.vel[:, 2]))
    target = jnp.clip((force - feedback) / jnp.maximum(state.rot[:, 2, 2], .85), .02, .98)
    return jax.jit(jax.vmap(env.observe))(state), target


def expand_actor(actor):
    """Embed the pilot actor in the paper architecture with the same outputs."""
    # A 40/60 split preserves the outgoing sum and lets the duplicated
    # units receive different gradients as training proceeds.
    for index in range(3):
        layer = actor[f"actor_{index}"]
        kernel = layer["kernel"]
        if index:
            kernel = jnp.concatenate([.4 * kernel, .6 * kernel], axis=0)
        actor[f"actor_{index}"] = dict(kernel=jnp.tile(kernel, (1, 2)), bias=jnp.tile(layer["bias"], 2))
    eye = jnp.eye(64)
    duplicate_decoder = jnp.concatenate([.4 * eye, .6 * eye], axis=0)
    # LeakyReLU(x) - LeakyReLU(-x) = (1 + negative_slope) x.
    signed_decoder = jnp.concatenate([eye, -eye], axis=0) / 1.01
    for index, decoder in enumerate((duplicate_decoder, signed_decoder, signed_decoder), start=3):
        actor[f"actor_{index}"] = dict(kernel=jnp.concatenate([decoder, -decoder], axis=1), bias=jnp.zeros(128))
    head = actor["actor_out"]
    actor["actor_out"] = dict(kernel=signed_decoder @ head["kernel"], bias=head["bias"])


def transfer_actor(fresh, old, variant):
    """Start B′ and L with the same policy function and fresh critics."""
    params = jax.tree.map(lambda x: x, fresh)
    for key, value in old["params"].items():
        if key.startswith("actor_") or key == "logstd":
            params["params"][key] = jax.tree.map(lambda x: x, value)
    if variant == L:
        kernel = params["params"]["actor_0"]["kernel"]
        params["params"]["actor_0"]["kernel"] = jnp.concatenate([kernel[:22], jnp.zeros_like(kernel[:2]), kernel[22:]])
    if fresh["params"]["actor_0"]["bias"].size == ARCHITECTURES["paper"][1]:
        expand_actor(params["params"])
    return params


def responses(model, params, obs):
    return command(model.apply(params, obs)[0])


def pair_loss(model, params, obs, target_gap, scale):
    thrust = responses(model, params, obs)[:, 0]
    return jnp.mean(((thrust[::2] - thrust[1::2] - target_gap) / scale)**2)


def warm_start(model, params, samples, force, pair_obs, gap, pair_scale, seed, updates):
    torque = responses(model, params, samples)[:, 1:]
    target = jnp.concatenate([force[:, None], torque], axis=1)
    opt = optax.adam(5e-4)

    def loss(p, idx):
        error = responses(model, p, samples[idx]) - target[idx]
        imitation = jnp.mean((error / .1)**2)
        return imitation + .12 * pair_loss(model, p, pair_obs, gap, pair_scale)

    def train(initial, rng):
        def step(carry, _):
            p, opt_state, key = carry
            key, subkey = jax.random.split(key)
            idx = jax.random.randint(subkey, (256,), 0, len(samples))
            value, grad = jax.value_and_grad(loss)(p, idx)
            change, opt_state = opt.update(grad, opt_state, p)
            return (optax.apply_updates(p, change), opt_state, key), value

        (p, _, _), losses = jax.lax.scan(step, (initial, opt.init(initial), rng), None, length=updates)
        return p, losses

    return jax.device_get(jax.jit(train)(params, jax.random.PRNGKey(91000 + seed)))


def branch_rows(env, model, params, ref, labels, variant, seed, phase):
    state = exact_start(ref)
    obs = jax.vmap(env.observe)(state)
    common = np.asarray(obs) if variant == B_PRIME else np.delete(np.asarray(obs), DERIVATIVE_CHANNELS, axis=1)
    np.testing.assert_array_equal(common[::2], common[1::2])
    action = responses(model, params, obs)
    actual, _, _, _, _, _ = jax.vmap(env.transition)(state, action)
    requested = np.asarray(action[:, 0])
    applied = np.asarray(actual.previous_action @ ALLOCATION[0])
    np.testing.assert_allclose(requested, applied, atol=2e-7)
    goal = np.asarray(held_thrust(ref))[:, 0]
    error = np.asarray(actual.vel[:, 2] - ref.vel[:, 1, 2])
    return [dict(variant=variant, seed=seed, phase=phase, pair=i//2, family=FAMILIES[labels["family"][i]],
        frequency_rad_s=labels["frequency"][i], teacher_gap_mN=1000 * (goal[i]-goal[i+1]),
        policy_gap_mN=1000 * (applied[i]-applied[i+1]),
        step_velocity_rmse_mps=float(np.sqrt(np.mean(error[i:i+2]**2)))) for i in range(0, len(obs), 2)]


def lag_ms(actual, desired):
    """Fit the velocity timing offset after 2 s, positive when the vehicle lags."""
    if len(actual) < 70:
        return None
    time = DT * np.arange(1, len(actual) + 1)
    keep = (time >= 2.) & (time <= time[-1] - .35)
    shifts = np.arange(-30, 31) * .01
    error = [np.mean((actual[keep] - np.interp(time[keep] - s, time, desired))**2) for s in shifts]
    return float(1000 * shifts[np.argmin(error)])


def evaluate(env, model, parameters, labels, variant, output):
    count = len(env.bank.pos)
    keys = jnp.repeat(jax.random.split(jax.random.PRNGKey(51001), count // 2), 2, axis=0)
    states, _ = jax.vmap(env.reset)(keys, jnp.arange(count))
    for array in (states.pos, states.vel, states.rot, states.omega):
        np.testing.assert_array_equal(array[::2], array[1::2])
    neutral = jnp.zeros(env.obs_size)
    policy = MaskedActor(model, neutral)
    test = make_evaluator(env, policy, record_state=True, initial_states=states)
    conditions = {"full": ()}
    if variant == L:
        conditions.update(rate_mask=(22,), curvature_mask=(23,), both_mask=(22, 23))
    rows = []
    desired_vel = np.asarray(env.bank.vel[:, 1:, 2])
    for (seed, stage), params in parameters.items():
        for condition, channels in conditions.items():
            if stage != "ppo" and condition != "full":
                continue
            mask = np.zeros(env.obs_size, dtype=bool)
            mask[list(channels)] = True
            metrics, frames = jax.device_get(test((params, mask)))
            if not np.isfinite(metrics).all():
                raise FloatingPointError("Nonfinite evaluation metric")
            if stage == "ppo" and condition in ("full", "both_mask"):
                np.savez_compressed(output / variant / f"seed_{seed}" / f"rollouts_{condition}.npz", **frames)
            for i, value in enumerate(metrics):
                active = frames["active"][:, i]
                actual_v = frames["velocity"][active, i, 2]
                target_v = desired_vel[i, active]
                rows.append(dict(variant=variant, seed=seed, phase=stage, condition=condition, trajectory=i,
                    family=FAMILIES[labels["family"][i]], sign=int(labels["sign"][i]),
                    rmse_cm=float(100 * value[0]), velocity_rmse_mps=float(value[1]),
                    vertical_velocity_rmse_mps=float(np.sqrt(np.mean((actual_v-target_v)**2))),
                    lag_ms=lag_ms(actual_v, target_v), attitude_rmse_rad=float(value[2]),
                    completed=int(not value[7]), survival_s=float(value[8])))
    return rows


def summarize(branches, rollouts):
    branch_summary, rollout_summary = [], []
    for key in dict.fromkeys((r["variant"], r["phase"], r["family"]) for r in branches):
        rows = [r for r in branches if (r["variant"], r["phase"], r["family"]) == key]
        desired = np.array([r["teacher_gap_mN"] for r in rows])
        actual = np.array([r["policy_gap_mN"] for r in rows])
        branch_summary.append(dict(variant=key[0], phase=key[1], family=key[2], trials=len(rows),
            mean_teacher_gap_mN=float(desired.mean()), mean_policy_gap_mN=float(actual.mean()),
            fraction_correct_sign=float(np.mean(actual > 1e-4)), gain=float(np.dot(actual, desired)/np.dot(desired, desired)),
            gap_rmse_mN=float(np.sqrt(np.mean((actual-desired)**2))),
            mean_step_velocity_rmse_mps=float(np.mean([r["step_velocity_rmse_mps"] for r in rows]))))
    for key in dict.fromkeys((r["variant"], r["phase"], r["condition"], r["family"]) for r in rollouts):
        rows = [r for r in rollouts if (r["variant"], r["phase"], r["condition"], r["family"]) == key]
        complete = [r for r in rows if r["completed"]]
        rollout_summary.append(dict(variant=key[0], phase=key[1], condition=key[2], family=key[3],
            trials=len(rows), completed=len(complete),
            mean_position_rmse_cm=float(np.mean([r["rmse_cm"] for r in rows])),
            mean_velocity_rmse_mps=float(np.mean([r["velocity_rmse_mps"] for r in rows])),
            mean_vertical_velocity_rmse_mps=float(np.mean([r["vertical_velocity_rmse_mps"] for r in rows])),
            mean_abs_lag_ms=float(np.mean([abs(r["lag_ms"]) for r in complete])) if complete else None))
    return branch_summary, rollout_summary


def digest(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def run(output, steps, warm_updates, horizon, auxiliary_weight, *, architecture="pilot"):
    layers, width = ARCHITECTURES[architecture]
    config = PPOConfig(total_steps=steps, layers=layers, width=width)
    config.validate()
    if warm_updates < 1 or horizon < 80 or auxiliary_weight < 0:
        raise ValueError("Choose a positive warm start, horizon >= 80 and a nonnegative auxiliary weight")
    vertical, train_labels = vertical_pairs(17201, 16, horizon)
    test, test_labels = vertical_pairs(60249, 12, horizon)
    validate_pairs(vertical, train_labels)
    validate_pairs(test, test_labels)
    _, gentle = make_bank(12345, 16, horizon)
    train = jax.tree.map(lambda a, b: jnp.concatenate([a, b]), vertical, gentle)
    derivative_scale = jnp.sqrt(jnp.mean(desired_thrust_jet(vertical)[..., 1:]**2, axis=(0, 1)))
    output.mkdir(parents=True, exist_ok=False)
    sources = [Path(__file__), Path("envs/quadrotor_smooth.py"), Path("envs/quadrotor_desired_thrust.py"),
               Path("envs/quadrotor_double_thrust.py"), Path("envs/quadrotor_thrust_rate.py"),
               Path("experiments/quadrotor_comparison.py"), Path("experiments/quadrotor_longer.py"),
               Path("experiments/diagnose_thrust_inputs.py"), Path("rl/quadrotor_ppo.py"),
               Path("requirements-quadrotor-lock.txt")]
    (output / "source").mkdir()
    for path in sources:
        shutil.copy2(path, output / "source" / path.name)
    for name, bank, labels in (("train", train, train_labels), ("held_out", test, test_labels)):
        np.savez_compressed(output / f"{name}_references.npz", **bank._asdict(), **labels)
    settings = dict(status="running", config=asdict(config), seeds=SEEDS, reward_profile=REWARD, architecture=architecture,
        dt=DT, horizon=horizon, train_pairs=32, train_gentle_3d_references=16, held_out_pairs=24,
        warm_updates=warm_updates, warm_learning_rate=5e-4, ppo_teacher_weight=auxiliary_weight,
        derivative_scales=np.asarray(derivative_scale).tolist(),
        initial_policy_sha256={seed: digest(SOURCE_RUN / B_PRIME / f"seed_{seed}" / "policy.msgpack") for seed in SEEDS},
        source_sha256={p.name: digest(p) for p in sources})
    write_json(output / "manifest.json", settings)
    gap = jnp.diff(held_thrust(vertical)[:, 0].reshape(-1, 2), axis=1)[:, 0] * -1
    scale = jnp.repeat(jnp.sqrt(jnp.mean(gap.reshape(2, -1)**2, axis=1)), 16)
    branches, rollouts, timing = [], [], []
    print(f"Held-out paired teacher gaps [mN], rate and curvature: "
          f"{np.round(1000 * np.asarray((held_thrust(test)[::2, 0] - held_thrust(test)[1::2, 0]).reshape(2, -1).mean(1)), 2)}", flush=True)
    for variant in (B_PRIME, L):
        env = make_environment(train, variant, reward_profile=REWARD)
        held_out = make_environment(test, variant, reward_profile=REWARD)
        if variant == L:
            for target_env in (env, held_out):
                target_env.scale = target_env.scale.at[jnp.array(DERIVATIVE_CHANNELS)].set(derivative_scale)
        pair_obs = jax.vmap(env.observe)(exact_start(vertical))
        samples, force = teacher_examples(env, vertical)

        def auxiliary(model, params):
            return auxiliary_weight * pair_loss(model, params, pair_obs, gap, scale)

        model, runners, update = independent_runners(env, config, SEEDS, actor_auxiliary_loss=auxiliary)
        warmed, parameters = [], {}
        histories = [[] for _ in SEEDS]
        warm_times = []
        for lane, seed in enumerate(SEEDS):
            path = output / variant / f"seed_{seed}"
            path.mkdir(parents=True)
            old_model, old, _ = load_policy(SOURCE_RUN / B_PRIME / f"seed_{seed}")
            initial = transfer_actor(jax.tree.map(lambda x: x[lane], runners.train_state.params), old, variant)
            sample = samples[:128]
            common = sample if variant == B_PRIME else jnp.concatenate([sample[:, :22], sample[:, 24:]], axis=1)
            np.testing.assert_allclose(responses(model, initial, sample), responses(old_model, old, common), atol=3e-6)
            branches.extend(branch_rows(held_out, model, initial, test, test_labels, variant, seed, "initial"))
            started = perf_counter()
            params, losses = warm_start(model, initial, samples, force, pair_obs, gap, scale, seed, warm_updates)
            warm_times.append(perf_counter() - started)
            if not np.isfinite(losses).all():
                raise FloatingPointError("Nonfinite imitation loss")
            np.save(path / "imitation_loss.npy", losses)
            (path / "warm_policy.msgpack").write_bytes(serialization.to_bytes(params))
            warmed.append(params)
            parameters[seed, "warm"] = params
            branches.extend(branch_rows(held_out, model, params, test, test_labels, variant, seed, "warm"))
            print(f"{variant} seed {seed} warm start, loss {losses[-1]:.4f}", flush=True)
        write_csv(output / "branch_trials.csv", branches)
        runners = runners._replace(train_state=runners.train_state.replace(
            params=jax.tree.map(lambda *x: jnp.stack(x), *warmed)))
        update_times = []
        for index in range(1, config.updates + 1):
            started = perf_counter()
            runners, metrics = jax.block_until_ready(update(runners))
            update_times.append(perf_counter() - started)
            metrics = {name: np.asarray(value) for name, value in metrics.items()}
            if not all(np.isfinite(x).all() for x in metrics.values()):
                raise FloatingPointError("Nonfinite PPO metric")
            for lane, history in enumerate(histories):
                history.append(dict(steps=index * config.batch_size, **{name: float(x[lane]) for name, x in metrics.items()}))
            if index == 1 or index % min(64, max(1, config.updates // 2)) == 0 or index == config.updates:
                for lane, seed in enumerate(SEEDS):
                    write_csv(output / variant / f"seed_{seed}" / "training.csv", histories[lane])
                print(f"{variant} PPO {index * config.batch_size:,}/{steps:,} steps/seed, "
                      f"reward {metrics['step_reward'].mean():.3f}, {sum(update_times):.1f}s", flush=True)
        post_compile = update_times[1:]
        timing.append(dict(variant=variant, architecture=architecture, seeds=len(SEEDS), warm_updates=warm_updates,
            warm_wall_seconds=sum(warm_times), ppo_updates=config.updates, ppo_wall_seconds=sum(update_times),
            first_update_seconds=update_times[0], post_compile_seconds_per_update=(float(np.mean(post_compile))
                if post_compile else None)))
        write_csv(output / "timing.csv", timing)
        for lane, seed in enumerate(SEEDS):
            params = jax.tree.map(lambda x: x[lane], runners.train_state.params)
            path = output / variant / f"seed_{seed}"
            (path / "policy.msgpack").write_bytes(serialization.to_bytes(params))
            write_json(path / "config.json", dict(variant=variant, seed=seed, steps=steps, ppo=asdict(config),
                observation_size=env.obs_size, observation_scale=np.asarray(env.scale).tolist(),
                initial_action_mean=env.initial_action_mean, reward_profile=REWARD))
            parameters[seed, "ppo"] = params
            branches.extend(branch_rows(held_out, model, params, test, test_labels, variant, seed, "ppo"))
        rollouts.extend(evaluate(held_out, model, parameters, test_labels, variant, output))
        write_csv(output / "branch_trials.csv", branches)
        write_csv(output / "rollout_trials.csv", rollouts)
        branch_summary, rollout_summary = summarize(branches, rollouts)
        write_csv(output / "branch_summary.csv", branch_summary)
        write_csv(output / "rollout_summary.csv", rollout_summary)
        print(f"{variant} paired evaluation saved", flush=True)
        jax.clear_caches()
    settings.update(status="complete", reference_sha256={name: digest(output / f"{name}_references.npz")
                                                           for name in ("train", "held_out")})
    write_json(output / "manifest.json", settings)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, default=Path("runs/se23_thrust_preview"))
    parser.add_argument("--steps", type=int, default=1048576)
    parser.add_argument("--warm-updates", type=int, default=1200)
    parser.add_argument("--horizon", type=int, default=240)
    parser.add_argument("--auxiliary-weight", type=float, default=.02)
    parser.add_argument("--architecture", choices=ARCHITECTURES, default="pilot")
    args = parser.parse_args()
    run(args.output, args.steps, args.warm_updates, args.horizon, args.auxiliary_weight, architecture=args.architecture)
