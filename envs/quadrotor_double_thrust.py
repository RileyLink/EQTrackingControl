"""I/J: F and Fdot are states; actions are Fddot and three body torques.

Desired thrust derivatives are analytic functions of the reference's acceleration,
jerk and snap. No measured-state differentiation or finite differences are used.
"""

import jax.numpy as jnp
import numpy as np

from envs.quadrotor_smooth import (
    ALLOCATION, DT, E3, GRAVITY, MASS, SmoothQuadrotor, at, observation,
)
from envs.quadrotor_thrust_rate import (
    INVERSE_ALLOCATION, MAX_THRUST, MAX_THRUST_RATE, TORQUE_LIMITS, ThrustState,
)


MAX_THRUST_ACCELERATION = 10.0  # N/s^2; reaches the 1 N/s rate limit in 0.1 s.


def desired_thrust_jet(ref):
    """Return [Fd, Fdot_d, Fddot_d], supporting individual or batched references."""
    q = ref.acc + GRAVITY * E3
    length = jnp.linalg.norm(q, axis=-1)
    qj = jnp.sum(q * ref.jerk, axis=-1)
    ddlength = ((jnp.sum(ref.jerk ** 2 + q * ref.snap, axis=-1) / length)
                - qj ** 2 / length ** 3)
    return MASS * jnp.stack([length, qj / length, ddlength], axis=-1)


def stopping_distance(speed, dt=DT, acceleration=MAX_THRUST_ACCELERATION):
    """Distance to stop a nonnegative speed with piecewise-constant controls.

    Brake at the acceleration limit for whole intervals, then choose a smaller
    constant acceleration to finish at zero speed. This conservative envelope
    leaves an admissible braking action available at every decision time.
    """
    speed = jnp.maximum(speed, 0.)
    intervals = jnp.floor(speed / (acceleration * dt))
    remaining = speed - intervals * acceleration * dt
    return (.5 * acceleration * (intervals * dt) ** 2
            + (intervals + .5) * remaining * dt)


def _speed_from_budget(budget, dt, acceleration):
    """Inverse of w -> dt*w/2 + stopping_distance(max(w,0))."""
    positive = jnp.maximum(budget, 0.)
    intervals = jnp.floor((jnp.sqrt(1 + 8 * positive / (acceleration * dt ** 2)) - 1) / 2)
    used = .5 * acceleration * dt ** 2 * intervals * (intervals + 1)
    remaining_speed = (positive - used) / ((intervals + 1) * dt)
    speed = intervals * acceleration * dt + remaining_speed
    return jnp.where(budget >= 0., speed, 2 * budget / dt)


def acceleration_bounds(thrust, rate, dt=DT, acceleration=MAX_THRUST_ACCELERATION):
    """Admissible constant Fddot preserving rate and discrete stopping envelopes.

    Fnext = F + dt*(rate + rate_next)/2. Requiring its remaining thrust distance
    to exceed stopping_distance(rate_next) gives monotone inequalities in the
    next rate, inverted above. No state is clipped after integration.
    """
    lower_rate = -_speed_from_budget(thrust + .5 * dt * rate, dt, acceleration)
    upper_rate = _speed_from_budget(MAX_THRUST - thrust - .5 * dt * rate, dt, acceleration)
    lower = jnp.maximum(-acceleration, (jnp.maximum(-MAX_THRUST_RATE, lower_rate) - rate) / dt)
    upper = jnp.minimum(acceleration, (jnp.minimum(MAX_THRUST_RATE, upper_rate) - rate) / dt)
    # At a braking boundary the exact feasible interval can be a single point.
    # Float32 roundoff may reverse its endpoints by a few ulps; keep a nonempty
    # interval within the hard acceleration limits without clipping either state.
    lower = jnp.clip(lower, -acceleration, acceleration)
    upper = jnp.maximum(lower, jnp.clip(upper, -acceleration, acceleration))
    return lower, upper


def double_extended_action(thrust, rate, normalized_action, dt=DT,
                           acceleration=MAX_THRUST_ACCELERATION):
    """Integrate the two thrust states and allocate torque over the full ramp."""
    action = jnp.clip(normalized_action, 0., 1.)
    requested = (2 * action[0] - 1) * acceleration
    lower, upper = acceleration_bounds(thrust, rate, dt, acceleration)
    applied = jnp.clip(requested, lower, upper)
    next_thrust = thrust + rate * dt + .5 * applied * dt ** 2
    next_rate = rate + applied * dt
    midpoint = thrust + rate * dt / 2 + applied * dt ** 2 / 8

    # A quadratic ramp can turn inside the interval. Both endpoints alone are
    # insufficient to establish rotor feasibility when the rate changes sign.
    denominator = jnp.where(jnp.abs(applied) > 1e-8, applied, 1.)
    turning_time = jnp.clip(-rate / denominator, 0., dt)
    turning_thrust = thrust + rate * turning_time + .5 * applied * turning_time ** 2
    minimum = jnp.minimum(jnp.minimum(thrust, next_thrust), turning_thrust) / MAX_THRUST
    maximum = jnp.maximum(jnp.maximum(thrust, next_thrust), turning_thrust) / MAX_THRUST
    torque = (2 * action[1:] - 1) * TORQUE_LIMITS
    delta = INVERSE_ALLOCATION @ jnp.concatenate([jnp.zeros(1), torque])
    bounds = jnp.where(delta > 0., (1 - maximum) / jnp.maximum(delta, 1e-8),
                       jnp.where(delta < 0., minimum / jnp.maximum(-delta, 1e-8), jnp.ones(4)))
    torque_scale = jnp.clip(jnp.min(bounds), 0., 1.)
    rotors = midpoint / MAX_THRUST + torque_scale * delta
    return rotors, next_thrust, next_rate, applied, torque_scale


class DoubleThrustExtension(SmoothQuadrotor):
    """Same extended plant for I (full observations) and J (invariant observations)."""

    action_definition = "bounded-collective-thrust-acceleration-and-three-feasible-body-torques"
    initial_action_mean = (.5, .5, .5, .5)

    def __init__(self, bank, symmetry=False, dt=DT,
                 max_thrust_acceleration=MAX_THRUST_ACCELERATION):
        if dt <= 0 or max_thrust_acceleration <= 0:
            raise ValueError("Time step and thrust acceleration limit must be positive")
        super().__init__(bank, symmetry=symmetry, derivatives=False, dt=dt)
        self.core = SmoothQuadrotor(bank, symmetry=symmetry, derivatives=False, dt=dt)
        self.max_thrust_acceleration = max_thrust_acceleration
        self.feature_definition = (("invariant-base" if symmetry else "full-base")
            + " with extended-reference-input [Fddot_d,tau_d] + [F,Fdot,F_d,Fdot_d]")
        self.experiment_metadata = {
            "dynamic_extension": True, "extension_order": 2,
            "symmetry": symmetry, "max_thrust_N": MAX_THRUST,
            "max_thrust_rate_Nps": MAX_THRUST_RATE,
            "max_thrust_acceleration_Nps2": max_thrust_acceleration,
            "body_torque_limits_Nm": [.8, .8, 1.6],
            "actual_rate": "continuous Fdot state, integrated from applied Fddot",
            "reference_input": "analytic [Fddot_d,tau_d], replacing rotor reference observation",
            "extra_features": ["F", "Fdot", "F_d", "Fdot_d"],
            "reference_derivatives": "analytic acceleration/jerk/snap of the desired sinusoidal planner",
            "state_initialization": "F=F_d(0), Fdot=Fdot_d(0); paired physical perturbations",
            "thrust_constraints": "acceleration projection preserving sampled-control stopping envelopes; no state clipping",
            "torque_allocation": "radial feasibility scaling over quadratic ramp, including interior extremum",
            "rigid_body_integration": "existing SO(3) midpoint method with F evaluated at dt/2",
            "reward": "unchanged physical tracking and midpoint rotor-command cost; no added thrust-state penalties",
        }
        input_start = 24 if symmetry else 36
        input_scales = jnp.concatenate([jnp.array([max_thrust_acceleration]), TORQUE_LIMITS])
        scale = self.core.scale.at[input_start:input_start + 4].set(input_scales)
        self.scale = jnp.concatenate([scale, jnp.array([MAX_THRUST, MAX_THRUST_RATE,
                                                       MAX_THRUST, MAX_THRUST_RATE])])
        self.obs_size = len(self.scale)
        jets = np.asarray(desired_thrust_jet(bank))
        if (not np.isfinite(jets).all() or np.any(jets[..., 0] <= 0.)
                or np.any(jets[..., 0] > MAX_THRUST)
                or np.any(np.abs(jets[..., 1]) > MAX_THRUST_RATE)
                or np.any(np.abs(jets[..., 2]) > max_thrust_acceleration)):
            raise ValueError("Desired thrust jet exceeds double-extension actuator limits")
        positive_stop = np.asarray(stopping_distance(jets[..., 1], dt, max_thrust_acceleration))
        negative_stop = np.asarray(stopping_distance(-jets[..., 1], dt, max_thrust_acceleration))
        if np.any(positive_stop > MAX_THRUST - jets[..., 0]) or np.any(negative_stop > jets[..., 0]):
            raise ValueError("Desired thrust states do not allow bounded braking at this time step")

    def observe(self, state):
        ref = at(self.bank, state.trajectory, state.step)
        jet = desired_thrust_jet(ref)
        torque = (ALLOCATION @ ref.action)[1:]
        reference_input = jnp.concatenate([jet[2:3], torque])
        base = observation(state, ref._replace(action=reference_input),
                           symmetry=self.symmetry, derivatives=False)
        extra = jnp.stack([state.thrust, state.thrust_rate, jet[0], jet[1]])
        return jnp.concatenate([base, extra]) / self.scale

    def reset(self, key, trajectory=None):
        state, _ = self.core.reset(key, trajectory)
        ref = at(self.bank, state.trajectory, state.step)
        jet = desired_thrust_jet(ref)
        state = ThrustState(*state, jet[0], jet[1])
        return state, self.observe(state)

    def transition(self, state, action):
        commands, thrust, rate, applied, torque_scale = double_extended_action(
            state.thrust, state.thrust_rate, action, self.dt, self.max_thrust_acceleration)
        next_state, _, reward, failed, truncated, info = self.core.transition(state, commands)
        next_state = next_state._replace(thrust=thrust, thrust_rate=rate)
        info = {**info, "thrust": thrust, "thrust_rate": rate,
                "thrust_acceleration": applied, "torque_scale": torque_scale}
        return next_state, self.observe(next_state), reward, failed, truncated, info
