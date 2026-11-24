"""Shared reference and reset distributions for fresh B′ and L training."""

import jax
import jax.numpy as jnp
import numpy as np

from envs.quadrotor_smooth import DT, at, make_bank, reference_at, rotation_exp, sample_trajectory
from experiments.quadrotor_thrust_preview import validate_pairs, vertical_pairs


FAMILIES = ("vertical", "spatial", "yaw", "combined")
YAW_MODES = ("yaw_angle", "yaw_rate", "yaw_angle_and_rate")
TRAIN_BANK_SEED = 56913
TEST_BANK_SEED = 79563
EVALUATION_SEED = 27651


def turning_trajectory(key, combined):
    base, shape, motion, turn, speed = jax.random.split(key, 5)
    trajectory = sample_trajectory(base)
    amplitude = (jax.random.uniform(shape, (3,), minval=jnp.array([1.0, 1.0, .2]),
                                    maxval=jnp.array([2.2, 2.2, .65])) if combined else jnp.zeros(3))
    frequency = jax.random.uniform(motion, (3,), minval=.60, maxval=1.23)
    yaw_amplitude = jax.random.uniform(turn, (), minval=.40, maxval=1.05)
    yaw_frequency = jax.random.uniform(speed, (), minval=.55, maxval=1.45)
    return trajectory._replace(amplitude=amplitude, frequency=frequency,
                               yaw_amplitude=yaw_amplitude, yaw_frequency=yaw_frequency)


def turning_bank(seed, count, horizon, *, combined):
    keys = jax.random.split(jax.random.PRNGKey(seed), count * 4)
    trajectory = jax.vmap(lambda key: turning_trajectory(key, combined))(keys)
    time = DT * jnp.arange(horizon + 1)
    evaluate = jax.jit(jax.vmap(lambda tr: jax.vmap(reference_at, in_axes=(0, None))(time, tr)))
    candidates = evaluate(trajectory)
    rotors = np.asarray(candidates.action)
    # Leave margin for feedback on the turning trajectories.
    valid = ((rotors >= .025) & (rotors <= .975)).all(axis=(1, 2))
    indices = np.flatnonzero(valid)[:count]
    if len(indices) < count:
        raise ValueError("Too few feasible turning references")
    return jax.tree_util.tree_map(lambda value: value[indices], candidates)


def build_bank(seed, per_family, horizon):
    if per_family < 4 or per_family % 4 or horizon < 4:
        raise ValueError("Use multiples of four trajectories per family and at least four time steps")
    vertical, pairs = vertical_pairs(seed, per_family // 4, horizon)
    validate_pairs(vertical, pairs)
    _, spatial = make_bank(seed + 1, per_family, horizon)
    yaw = turning_bank(seed + 2, per_family, horizon, combined=False)
    combined = turning_bank(seed + 3, per_family, horizon, combined=True)
    bank = jax.tree_util.tree_map(lambda *values: jnp.concatenate(values), vertical, spatial, yaw, combined)
    rotors = np.asarray(bank.action)
    if (not all(np.isfinite(np.asarray(value)).all() for value in bank)
            or np.any(rotors < 0.) or np.any(rotors > 1.)):
        raise ValueError("Training or evaluation reference bank is infeasible")
    return bank, np.repeat(FAMILIES, per_family), pairs


def yaw_start(state, ref, key, mode, *, angle_bounds=(10., 45.), rate_bounds=(.2, .95)):
    """Change only relative yaw and body-z rate, keeping the thrust axis fixed."""
    keys = jax.random.split(key, 3)
    sign = jnp.where(jax.random.bernoulli(keys[0], .5, (2,)), 1., -1.)
    angle = sign[0] * jnp.deg2rad(jax.random.uniform(keys[1], (), minval=angle_bounds[0],
                                                     maxval=angle_bounds[1]))
    rate = sign[1] * jax.random.uniform(keys[2], (), minval=rate_bounds[0], maxval=rate_bounds[1])
    angle = jnp.where(mode == 1, 0., angle)
    rate = jnp.where(mode == 0, 0., rate)
    change = rotation_exp(jnp.array([0., 0., angle]))
    omega = change.T @ ref.omega + jnp.array([0., 0., rate])
    return state._replace(pos=ref.pos, vel=ref.vel, rot=ref.rot @ change, omega=omega)


def lateral_start(state, ref, key, *, bounds=(.65, .55, .30, .42)):
    """Add tilted, translating recovery cases in reference-body coordinates."""
    position, velocity, angle, rate = jax.random.uniform(key, (4, 3), minval=-1., maxval=1.)
    change = rotation_exp(bounds[2] * angle)
    position = bounds[0] * position * jnp.array([1., 1., .5])
    velocity = bounds[1] * velocity * jnp.array([1., 1., .5])
    return state._replace(pos=ref.pos + ref.rot @ position, vel=ref.vel + ref.rot @ velocity,
                          rot=ref.rot @ change, omega=change.T @ ref.omega + bounds[3] * rate)


class PracticeEnvironment:
    """Use the existing plant and reward with family-specific training starts."""

    def __init__(self, environment, families):
        self.environment = environment
        self.family = jnp.asarray([FAMILIES.index(name) for name in families])

    def __getattr__(self, name):
        return getattr(self.environment, name)

    def reset(self, key, trajectory=None):
        normal, selection, perturbation, mode_key = jax.random.split(key, 4)
        if trajectory is None:
            trajectory = jax.random.randint(selection, (), 0, len(self.family))
        state, _ = self.environment.reset(normal, trajectory)
        ref = at(self.bank, trajectory, 0)
        mode = jax.random.randint(mode_key, (), 0, len(YAW_MODES))
        group = jnp.maximum(self.family[trajectory] - 1, 0)
        state = jax.lax.switch(
            group, (lambda: state, lambda: yaw_start(state, ref, perturbation, mode),
                    lambda: lateral_start(state, ref, perturbation)),
        )
        return state, self.observe(state)
