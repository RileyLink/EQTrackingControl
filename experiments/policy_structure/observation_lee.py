"""Geometric feedback computed from the L actor's unscaled observation."""

from typing import NamedTuple

import jax
import jax.numpy as jnp

from envs.quadrotor_smooth import E3, INERTIA, MASS, hat
from envs.quadrotor_thrust_rate import MAX_THRUST
from experiments.policy_structure.lee import DEFAULT_GAINS, _attitude_jet, rotation_terms


L_VARIANT = "L_se23_desired_thrust_extension"


class Tracking(NamedTuple):
    rot: jax.Array
    pos: jax.Array
    body_vel: jax.Array
    rate_error: jax.Array
    omega_d: jax.Array
    thrust_jet: jax.Array
    moment_d: jax.Array

    @property
    def omega(self):
        return self.rate_error + self.rot.T @ self.omega_d

    @property
    def vel(self):
        return self.rot @ self.body_vel


class Command(NamedTuple):
    thrust_terms: jax.Array
    position_projections: jax.Array
    velocity_projections: jax.Array
    moment_terms: jax.Array
    valid: jax.Array

    @property
    def rotation_errors(self):
        return self.moment_terms[:2].ravel()

    @property
    def feedforward(self):
        return jnp.concatenate((self.thrust_terms[:1], jnp.sum(self.moment_terms[2:], axis=0)))


def unpack(obs):
    return Tracking(obs[:9].reshape(3, 3), obs[9:12], obs[12:15], obs[15:18],
                    obs[18:21], obs[21:24], obs[24:27])


def desired_jets(tracking):
    """Inertial derivatives expressed in the desired frame at this instant."""
    omega = tracking.omega_d
    alpha = jnp.linalg.solve(INERTIA, tracking.moment_d - jnp.cross(omega, INERTIA @ omega))
    W = hat(omega)
    A = hat(alpha) + W @ W
    f, df, ddf = tracking.thrust_jet
    b_dot, b_ddot = W[:, 2], A[:, 2]
    force = jnp.stack((f * E3, df * E3 + f * b_dot, ddf * E3 + 2 * df * b_dot + f * b_ddot))
    e1 = jnp.array([1., 0., 0.], dtype=tracking.rot.dtype)
    heading = jnp.stack((e1, W[:, 0], A[:, 0]))
    return force, heading, alpha


def command_terms(tracking, kp, kv, heading_jet=None):
    force_d, body_heading, alpha_d = desired_jets(tracking)
    desired, desired_dot, desired_ddot = force_d
    heading, heading_dot, heading_ddot = body_heading if heading_jet is None else heading_jet
    velocity = tracking.vel
    body_z = tracking.rot[:, 2]
    body_z_dot = tracking.rot @ jnp.cross(tracking.omega, E3)
    force = desired - kp * tracking.pos - kv * velocity
    requested = force @ body_z
    thrust = jnp.clip(requested, 0., MAX_THRUST)
    acceleration_error = (thrust * body_z - desired) / MASS
    force_dot = desired_dot - kp * velocity - kv * acceleration_error
    thrust_dot = jnp.where((requested > 0.) & (requested < MAX_THRUST),
                           force_dot @ body_z + force @ body_z_dot, 0.)
    jerk_error = (thrust_dot * body_z + thrust * body_z_dot - desired_dot) / MASS
    force_ddot = desired_ddot - kp * acceleration_error - kv * jerk_error

    cross = jnp.cross(force, heading)
    norm_sq = force @ force
    valid = (norm_sq > 1e-10) & (cross @ cross > 1e-10 * norm_sq)
    Rc, omega_c, alpha_c = jax.lax.cond(
        valid,
        lambda: _attitude_jet(force, force_dot, force_ddot, heading, heading_dot, heading_ddot),
        lambda: (jnp.eye(3, dtype=tracking.rot.dtype), tracking.omega_d, alpha_d),
    )
    rotation = rotation_terms(tracking, Rc, omega_c, alpha_c)
    position = tracking.pos * body_z
    velocity_projection = velocity * body_z
    return Command(jnp.array([desired @ body_z, position.sum(), tracking.body_vel[2]]),
                   position, velocity_projection, jnp.stack(rotation), valid)


def request(obs, gains=DEFAULT_GAINS, heading_jet=None):
    kp, kv, kr, kw = gains
    terms = command_terms(unpack(obs), kp, kv, heading_jet)
    thrust = terms.thrust_terms @ jnp.array([1., -kp, -kv])
    moment = jnp.array([-kr, -kw, 1., 1., 1.]) @ terms.moment_terms
    return jnp.concatenate((thrust[None], moment))


def gain_design(terms):
    """Use the same linear scalar/diagonal fitting interface as the original study."""
    translation = -jnp.concatenate((terms.position_projections, terms.velocity_projections))
    return translation, -terms.moment_terms[:2].T, terms.feedforward


def world_heading_in_desired_frame(ref):
    """Reference information used solely to reproduce the old baseline."""
    heading = jnp.array([jnp.cos(ref.yaw), jnp.sin(ref.yaw), 0.])
    turn = jnp.cross(E3, heading)
    world = jnp.stack((heading, ref.yaw_rate * turn,
                       ref.yaw_acc * turn - ref.yaw_rate**2 * heading))
    return world @ ref.rot
