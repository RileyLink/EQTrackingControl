"""Thrust-rate input ablation and a first-order collective-thrust extension.

F is scalar total thrust in N. G retains rotor-command actions and supplies the
previous interval's causal thrust change. H adds a continuous thrust state with
rate and body-torque actions. Neither observes acceleration/jerk/snap/yaw jets.
"""

from typing import NamedTuple

import jax
import jax.numpy as jnp

from envs.quadrotor_smooth import (
    ALLOCATION, DT, E3, GRAVITY, MASS, SmoothQuadrotor, at, observation,
)


MAX_THRUST = float(jnp.sum(ALLOCATION[0]))  # Four unit commands: 1 N.
MAX_THRUST_RATE = 1.0  # N/s, so a full-range thrust change takes one second.
TORQUE_LIMITS = jnp.array([.8, .8, 1.6])  # Individual attainable axis extrema, Nm.
INVERSE_ALLOCATION = jnp.linalg.inv(ALLOCATION)


class ThrustState(NamedTuple):
    pos: jax.Array
    vel: jax.Array
    rot: jax.Array
    omega: jax.Array
    trajectory: jax.Array
    step: jax.Array
    previous_action: jax.Array
    thrust: jax.Array
    thrust_rate: jax.Array


def desired_thrust_rate(ref):
    """d/dt [m ||a_d + g e3||], evaluated analytically from the planner."""
    specific_force = ref.acc + GRAVITY * E3
    return MASS * (specific_force @ ref.jerk) / jnp.linalg.norm(specific_force)


def extended_action(thrust, normalized_action, dt=DT):
    """Integrate thrust rate and allocate feasible, constant body torque.

    Scale requested torque radially so every rotor remains in [0,1] at both
    endpoints of the linear thrust ramp. This preserves collective thrust and
    avoids independently clipping rotors, which would break the thrust integrator.
    Midpoint rotor commands are supplied to the existing midpoint integrator.
    """
    action = jnp.clip(normalized_action, 0., 1.)
    requested_rate = (2 * action[0] - 1) * MAX_THRUST_RATE
    next_thrust = jnp.clip(thrust + dt * requested_rate, 0., MAX_THRUST)
    applied_rate = (next_thrust - thrust) / dt
    requested_torque = (2 * action[1:] - 1) * TORQUE_LIMITS
    delta = INVERSE_ALLOCATION @ jnp.concatenate([jnp.zeros(1), requested_torque])
    minimum_base = jnp.minimum(thrust, next_thrust) / MAX_THRUST
    maximum_base = jnp.maximum(thrust, next_thrust) / MAX_THRUST
    positive_bound = (1 - maximum_base) / jnp.maximum(delta, 1e-8)
    negative_bound = minimum_base / jnp.maximum(-delta, 1e-8)
    bounds = jnp.where(delta > 0, positive_bound,
                       jnp.where(delta < 0, negative_bound, jnp.ones(4)))
    torque_scale = jnp.clip(jnp.min(bounds), 0., 1.)
    midpoint = (thrust + next_thrust) / (2 * MAX_THRUST)
    rotor_commands = midpoint + torque_scale * delta
    return rotor_commands, next_thrust, applied_rate, torque_scale


class ThrustRateInputs(SmoothQuadrotor):
    """G: baseline invariant features plus desired/actual scalar thrust rates."""

    feature_definition = "invariant-base + desired-Fdot + backward-applied-Fdot"
    action_definition = "four-normalized-rotor-commands"
    experiment_metadata = {"dynamic_extension": False, "rate_units": "N/s",
                           "actual_rate": "backward difference of applied collective thrust; zero at reset",
                           "rate_feature_scales": [1., 1.]}

    def __init__(self, bank, dt=DT):
        super().__init__(bank, symmetry=True, derivatives=False, dt=dt)
        self.core = SmoothQuadrotor(bank, symmetry=True, derivatives=False, dt=dt)
        self.scale = jnp.concatenate([self.core.scale, jnp.ones(2)])  # Both rates in N/s.
        self.obs_size = len(self.scale)

    def observe(self, state):
        ref = at(self.bank, state.trajectory, state.step)
        base = observation(state, ref, symmetry=True, derivatives=False)
        rates = jnp.stack([desired_thrust_rate(ref), state.thrust_rate])
        return jnp.concatenate([base, rates]) / self.scale

    def reset(self, key, trajectory=None):
        state, _ = self.core.reset(key, trajectory)
        thrust = ALLOCATION[0] @ state.previous_action
        state = ThrustState(*state, thrust, jnp.array(0.))
        return state, self.observe(state)

    def transition(self, state, action):
        action = jnp.clip(action, 0., 1.)
        thrust = ALLOCATION[0] @ action
        # Before the new action is chosen, this feature only describes the last
        # completed interval. It is not a future derivative or a ZOH derivative.
        rate = (thrust - ALLOCATION[0] @ state.previous_action) / self.dt
        next_state, _, reward, failed, truncated, info = self.core.transition(state, action)
        next_state = next_state._replace(thrust=thrust, thrust_rate=rate)
        return next_state, self.observe(next_state), reward, failed, truncated, info


class DynamicThrustExtension(ThrustRateInputs):
    """H: dot(F)=nu; four normalized actions represent nu and three body torques."""

    feature_definition = "invariant-base + desired-Fdot + previous-applied-Fdot + thrust-state"
    action_definition = "bounded-collective-thrust-rate-and-three-feasible-body-torques"
    initial_action_mean = (.5, .5, .5, .5)  # Zero rate and torque, retaining initial thrust.
    experiment_metadata = {"dynamic_extension": True, "extension_order": 1,
                           "max_thrust_N": MAX_THRUST, "max_thrust_rate_Nps": MAX_THRUST_RATE,
                           "body_torque_limits_Nm": [.8, .8, 1.6],
                           "torque_allocation": "radial feasibility scaling over entire thrust ramp",
                           "actual_rate": "previous applied rate; zero at reset",
                           "rate_feature_scales": [1., 1.]}

    def __init__(self, bank, dt=DT):
        super().__init__(bank, dt)
        self.scale = jnp.concatenate([self.scale, jnp.ones(1)])  # Thrust in N.
        self.obs_size = len(self.scale)

    def observe(self, state):
        ref = at(self.bank, state.trajectory, state.step)
        base = observation(state, ref, symmetry=True, derivatives=False)
        extra = jnp.stack([desired_thrust_rate(ref), state.thrust_rate, state.thrust])
        return jnp.concatenate([base, extra]) / self.scale

    def transition(self, state, action):
        commands, thrust, rate, torque_scale = extended_action(state.thrust, action, self.dt)
        next_state, _, reward, failed, truncated, info = self.core.transition(state, commands)
        next_state = next_state._replace(thrust=thrust, thrust_rate=rate)
        info = {**info, "thrust_rate": rate, "torque_scale": torque_scale}
        return next_state, self.observe(next_state), reward, failed, truncated, info
