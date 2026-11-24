"""Direct collective thrust and torque actions with optional desired thrust derivatives.

Observation order is [p,v,R,omega,p_d,v_d,R_d,omega_d,F_d,Fdot_d,Fddot_d,tau_d].
The physical state is the original (p,v,R,omega). No actual thrust integrator or
actual thrust-rate state is introduced. Reference derivatives are planner data.
"""

import jax.numpy as jnp
import numpy as np

from envs.quadrotor_smooth import (
    ALLOCATION, DT, GRAVITY, MASS, SE23_TRACKING_ORDER, SE23_TRACKING_SCALE,
    SmoothQuadrotor, at, observation, se23_tracking_features,
)
from envs.quadrotor_double_thrust import desired_thrust_jet
from envs.quadrotor_thrust_rate import INVERSE_ALLOCATION, MAX_THRUST, TORQUE_LIMITS


REFERENCE_RATE_SCALE = 1.0  # N/s; observation normalization, not an actuator limit.
REFERENCE_ACCELERATION_SCALE = 10.0  # N/s^2; normalization only.


def direct_wrench_action(normalized_action):
    """Decode f/tau; preserve f while scaling torque into the rotor-feasible set."""
    action = jnp.clip(normalized_action, 0., 1.)
    thrust = action[0] * MAX_THRUST
    torque = (2 * action[1:] - 1) * TORQUE_LIMITS
    delta = INVERSE_ALLOCATION @ jnp.concatenate([jnp.zeros(1), torque])
    base = thrust / MAX_THRUST
    bounds = jnp.where(delta > 0., (1 - base) / jnp.maximum(delta, 1e-8),
                       jnp.where(delta < 0., base / jnp.maximum(-delta, 1e-8), jnp.ones(4)))
    scale = jnp.clip(jnp.min(bounds), 0., 1.)
    return base + scale * delta, thrust, scale


class DesiredThrustExtension(SmoothQuadrotor):
    """Full or invariant observations with desired wrench and optional thrust derivatives."""

    feature_definition = "[p,v,R,omega,p_d,v_d,R_d,omega_d,F_d,Fdot_d,Fddot_d,tau_d]"
    action_definition = "direct-collective-thrust-and-three-feasible-body-torques"
    initial_action_mean = (MASS * GRAVITY / MAX_THRUST, .5, .5, .5)
    experiment_metadata = {
        "dynamic_extension": False, "extension_order": 0,
        "desired_reference_extension_order": 2, "symmetry": False,
        "actual_state": ["p", "v", "R", "omega"],
        "observation_order": ["p", "v", "R", "omega", "p_d", "v_d", "R_d", "omega_d",
                              "F_d", "Fdot_d", "Fddot_d", "tau_d"],
        "max_thrust_N": MAX_THRUST, "body_torque_limits_Nm": [.8, .8, 1.6],
        "actual_thrust_evolution": "direct held thrust command each interval; no rate or acceleration bounds",
        "reference_thrust_rate_scale_Nps": REFERENCE_RATE_SCALE,
        "reference_thrust_acceleration_scale_Nps2": REFERENCE_ACCELERATION_SCALE,
        "reference_derivatives": "analytic desired acceleration/jerk/snap; no finite differences",
        "torque_allocation": "radial rotor-feasibility scaling at the directly commanded thrust",
        "initial_action_mean": "hover collective thrust and zero body torque",
        "control_law": "direct PPO outputs; no added analytical feedforward",
        "reward": "unchanged physical tracking and applied rotor-command cost",
    }

    def __init__(self, bank, dt=DT, symmetry=False, se23=False, include_thrust_derivatives=True):
        super().__init__(bank, symmetry=symmetry, dt=dt, se23=se23)
        self.core = SmoothQuadrotor(bank, symmetry=symmetry, derivatives=False, dt=dt)
        self.reference_thrust_size = 3 if include_thrust_derivatives else 1
        if se23:
            base_order = SE23_TRACKING_ORDER
            base_scale = SE23_TRACKING_SCALE
        elif symmetry:
            base_order = ["R.T(p_d-p)", "R.T v", "R.T v_d", "vec(R.T R_d)", "omega", "omega_d", "R.T e3"]
            base_scale = jnp.concatenate([self.core.scale[:24], self.core.scale[-3:]])
        else:
            base_order = type(self).experiment_metadata["observation_order"][:8]
            base_scale = self.core.scale[:36]
        thrust_order = ["F_d", "Fdot_d", "Fddot_d"][:self.reference_thrust_size]
        order = [*base_order, *thrust_order, "tau_d"]
        self.experiment_metadata = {**type(self).experiment_metadata, **self.experiment_metadata,
                                    "symmetry": symmetry, "observation_order": order}
        if not include_thrust_derivatives:
            self.experiment_metadata.update(desired_reference_extension_order=0, reference_derivatives="none")
            self.experiment_metadata.pop("reference_thrust_rate_scale_Nps")
            self.experiment_metadata.pop("reference_thrust_acceleration_scale_Nps2")
        self.feature_definition = "[" + ",".join(order) + "]"
        thrust_scales = jnp.array([MAX_THRUST, REFERENCE_RATE_SCALE, REFERENCE_ACCELERATION_SCALE])
        self.scale = jnp.concatenate([base_scale, thrust_scales[:self.reference_thrust_size], TORQUE_LIMITS])
        self.obs_size = len(self.scale)
        jets = np.asarray(desired_thrust_jet(bank)[..., :self.reference_thrust_size])
        if (not np.isfinite(jets).all() or np.any(jets[..., 0] <= 0.)
                or np.any(jets[..., 0] > MAX_THRUST)):
            raise ValueError("Desired thrust reference is non-finite or exceeds the thrust limit")

    def observe(self, state):
        ref = at(self.bank, state.trajectory, state.step)
        if self.se23:
            base = se23_tracking_features(state, ref)
        else:
            original = observation(state, ref, symmetry=self.symmetry, derivatives=False)
            base = (jnp.concatenate([original[:24], original[-3:]])
                    if self.symmetry else original[:36])
        jet = desired_thrust_jet(ref)[:self.reference_thrust_size]
        # Use the same allocation as the physical model for reference body torque.
        torque = (ALLOCATION @ ref.action)[1:]
        return jnp.concatenate([base, jet, torque]) / self.scale

    def reset(self, key, trajectory=None):
        state, _ = self.core.reset(key, trajectory)
        return state, self.observe(state)

    def transition(self, state, action):
        rotors, thrust, torque_scale = direct_wrench_action(action)
        nxt, _, reward, failed, truncated, info = self.core.transition(state, rotors)
        info = {**info, "applied_collective_thrust": thrust, "torque_scale": torque_scale}
        return nxt, self.observe(nxt), reward, failed, truncated, info
