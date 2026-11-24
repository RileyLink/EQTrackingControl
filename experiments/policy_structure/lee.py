"""Lee–Leok–McClamroch position tracking in the simulator's z-up convention."""

from typing import NamedTuple

import jax
import jax.numpy as jnp

from envs.quadrotor_smooth import (
    ALLOCATION, E3, GRAVITY, INERTIA, MASS, hat, normalized_jet, vee,
)
from envs.quadrotor_thrust_rate import MAX_THRUST


class Gains(NamedTuple):
    position: float
    velocity: float
    attitude: float
    rate: float


DEFAULT_GAINS = Gains(4.0 * MASS, 3.2 * MASS, 1.2, 0.65)


class RotationTerms(NamedTuple):
    attitude: jax.Array
    rate: jax.Array
    gyro: jax.Array
    transport: jax.Array
    acceleration: jax.Array

    @property
    def feedforward(self):
        return self.gyro + self.transport + self.acceleration


class LeeTerms(NamedTuple):
    position: jax.Array
    velocity: jax.Array
    body_z: jax.Array
    thrust_ff: jax.Array
    rotation: RotationTerms
    valid_commanded_attitude: jax.Array


def attitude_vector(R, Rc):
    Q = Rc.T @ R
    return vee(Q - Q.T) / 2


def rotation_terms(state, Rc, omega_c, alpha_c):
    transport = state.rot.T @ Rc
    transported_rate = transport @ omega_c
    return RotationTerms(
        attitude_vector(state.rot, Rc), state.omega - transported_rate,
        jnp.cross(state.omega, INERTIA @ state.omega),
        -(INERTIA @ jnp.cross(state.omega, transported_rate)),
        INERTIA @ (transport @ alpha_c),
    )


def planner_alpha(ref):
    torque = (ALLOCATION @ ref.action)[1:]
    return jnp.linalg.solve(INERTIA, torque - jnp.cross(ref.omega, INERTIA @ ref.omega))


def _attitude_jet(force, force_dot, force_ddot, heading, heading_dot, heading_ddot):
    """Same heading and normalized-vector construction as the flatness planner."""
    b3, db3, ddb3 = normalized_jet(force, force_dot, force_ddot)
    b2, db2, ddb2 = normalized_jet(
        jnp.cross(b3, heading),
        jnp.cross(db3, heading) + jnp.cross(b3, heading_dot),
        jnp.cross(ddb3, heading) + 2 * jnp.cross(db3, heading_dot)
        + jnp.cross(b3, heading_ddot),
    )
    b1 = jnp.cross(b2, b3)
    db1 = jnp.cross(db2, b3) + jnp.cross(b2, db3)
    ddb1 = jnp.cross(ddb2, b3) + 2 * jnp.cross(db2, db3) + jnp.cross(b2, ddb3)
    Rc = jnp.stack((b1, b2, b3), axis=1)
    Rc_dot = jnp.stack((db1, db2, db3), axis=1)
    Rc_ddot = jnp.stack((ddb1, ddb2, ddb3), axis=1)
    W = Rc.T @ Rc_dot
    Wdot = Rc.T @ Rc_ddot - W @ W
    return Rc, vee(W - W.T) / 2, vee(Wdot - Wdot.T) / 2


def commanded_attitude(state, ref, force, velocity_error, kp, kv):
    """Continuous command jet using Lee's own thrust and its first derivative.

    Collective clipping enters the acceleration and jerk used by the command
    jet. The simulator subsequently holds the feasible rotor command for dt.
    """
    b3 = state.rot[:, 2]
    b3_dot = (state.rot @ hat(state.omega))[:, 2]
    requested = force @ b3
    thrust = jnp.clip(requested, 0., MAX_THRUST)
    acc = (thrust / MASS) * b3 - GRAVITY * E3
    acc_error = acc - ref.acc
    force_dot = MASS * ref.jerk - kp * velocity_error - kv * acc_error
    thrust_dot = jnp.where((requested > 0) & (requested < MAX_THRUST),
                           force_dot @ b3 + force @ b3_dot, 0.)
    jerk = (thrust_dot * b3 + thrust * b3_dot) / MASS
    force_ddot = MASS * ref.snap - kp * acc_error - kv * (jerk - ref.jerk)
    heading = jnp.array([jnp.cos(ref.yaw), jnp.sin(ref.yaw), 0.])
    heading_dot = ref.yaw_rate * jnp.cross(E3, heading)
    heading_ddot = ref.yaw_acc * jnp.cross(E3, heading) - ref.yaw_rate**2 * heading
    # At the singularities the requested force or heading leaves Rc undefined.
    force_sq = force @ force
    cross = jnp.cross(force, heading)
    valid = (force_sq > 1e-10) & (cross @ cross > 1e-10 * force_sq)
    Rc, omega_c, alpha_c = jax.lax.cond(
        valid,
        lambda: _attitude_jet(force, force_dot, force_ddot, heading, heading_dot, heading_ddot),
        lambda: (ref.rot, ref.omega, planner_alpha(ref)),
    )
    return Rc, omega_c, alpha_c, valid


def lee_terms(state, ref, gains=DEFAULT_GAINS):
    position, velocity = state.pos - ref.pos, state.vel - ref.vel
    body_z = state.rot[:, 2]
    nominal_force = MASS * (ref.acc + GRAVITY * E3)
    force = nominal_force - gains.position * position - gains.velocity * velocity
    Rc, omega_c, alpha_c, valid = commanded_attitude(
        state, ref, force, velocity, gains.position, gains.velocity,
    )
    return LeeTerms(position, velocity, body_z, nominal_force @ body_z,
                    rotation_terms(state, Rc, omega_c, alpha_c), valid)


def lee_wrench(terms, gains=DEFAULT_GAINS):
    """Requested wrench, before actuator limits."""
    thrust = (terms.thrust_ff - gains.position * (terms.position @ terms.body_z)
              - gains.velocity * (terms.velocity @ terms.body_z))
    torque = (terms.rotation.feedforward - gains.attitude * terms.rotation.attitude
              - gains.rate * terms.rotation.rate)
    return jnp.concatenate((thrust[None], torque))
