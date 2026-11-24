"""Numerical checks for the frozen-policy structure experiments."""

from pathlib import Path
from types import SimpleNamespace
import unittest

import jax
import jax.numpy as jnp
import numpy as np
from scipy.spatial.transform import Rotation

from envs.quadrotor_smooth import ALLOCATION, GRAVITY, MASS, State, reference_at, sample_trajectory
from experiments.policy_structure.common import Policy, actor_wrench, feasible_wrench
from experiments.policy_structure.jacobians import (
    actor_jacobians, finite_differences, perturb, state_jacobians, tracking_coordinates,
)
from experiments.policy_structure.lee import (
    DEFAULT_GAINS, attitude_vector, commanded_attitude, lee_terms, lee_wrench,
)
from experiments.policy_structure.lee_fit import design, fit, predict
from experiments.policy_structure.sparse_fit import candidate_terms, select_thresholds


def example():
    trajectory = sample_trajectory(jax.random.PRNGKey(713))
    time = 1.3
    ref = reference_at(time, trajectory)
    state = State(ref.pos, ref.vel, ref.rot, ref.omega, jnp.array(0), jnp.array(0), ref.action)
    return trajectory, time, ref, state


class FixedActor:
    """Known observation map with a stable, active radial torque constraint."""

    def apply(self, _, obs):
        bias = jnp.array([.25, .9, .15, -.21])
        weight = jnp.linspace(-.3, .4, 27 * 4).reshape(27, 4)
        return bias + obs @ weight, None, None


class TestPolicyStructure(unittest.TestCase):
    def test_lee_reproduces_reference_wrench_and_command_jet_on_manifold(self):
        _, _, ref, state = example()
        terms = lee_terms(state, ref)
        np.testing.assert_allclose(lee_wrench(terms), ALLOCATION @ ref.action, atol=2e-6)
        np.testing.assert_allclose(terms.rotation.attitude, 0., atol=1e-7)
        np.testing.assert_allclose(terms.rotation.rate, 0., atol=2e-7)
        thrust, moment, ff = design(state, ref, DEFAULT_GAINS)
        np.testing.assert_allclose(thrust, 0., atol=1e-7)
        np.testing.assert_allclose(moment, 0., atol=2e-7)
        np.testing.assert_allclose(ff, ALLOCATION @ ref.action, atol=2e-6)

    def test_coupled_command_jet_matches_time_differencing(self):
        trajectory, time, ref, state = example()
        state = perturb(state, jnp.array([.10, -.08, .13, .15, .03, -.05, .12, -.09, .17, .08, -.11, .07]))
        kp, kv, *_ = DEFAULT_GAINS
        force = MASS * (ref.acc + jnp.array([0., 0., GRAVITY])) - kp * (state.pos - ref.pos) - kv * (state.vel - ref.vel)
        rc, omega_c, alpha_c, valid = commanded_attitude(state, ref, force, state.vel - ref.vel, kp, kv)
        self.assertTrue(valid)
        b3 = np.asarray(state.rot[:, 2])
        b3_dot = np.asarray(state.rot) @ np.cross(np.asarray(state.omega), [0., 0., 1.])
        f = float(force @ b3)
        self.assertTrue(0 < f < 1)
        acc = f * b3 / MASS - [0., 0., GRAVITY]
        f_dot = float((MASS * ref.jerk - kp * (state.vel - ref.vel) - kv * (acc - ref.acc)) @ b3 + force @ b3_dot)
        jerk = (f_dot * b3 + f * b3_dot) / MASS

        def independent_rc(t):
            ref_t = reference_at(time + t, trajectory)
            pos_t = np.asarray(state.pos) + t * state.vel + .5 * t**2 * acc + t**3 * jerk / 6
            vel_t = np.asarray(state.vel) + t * acc + .5 * t**2 * jerk
            force_t = np.asarray(MASS * (ref_t.acc + jnp.array([0., 0., GRAVITY]))
                                 - kp * (pos_t - ref_t.pos) - kv * (vel_t - ref_t.vel), dtype=float)
            z = force_t / np.linalg.norm(force_t)
            y = np.cross(z, [np.cos(ref_t.yaw), np.sin(ref_t.yaw), 0.])
            y /= np.linalg.norm(y)
            return np.column_stack((np.cross(y, z), y, z))

        h = .01
        left, center, right = independent_rc(-h), independent_rc(0.), independent_rc(h)
        np.testing.assert_allclose(rc, center, atol=3e-7)
        rel_left = Rotation.from_matrix(center.T @ left).as_rotvec()
        rel_right = Rotation.from_matrix(center.T @ right).as_rotvec()
        np.testing.assert_allclose(omega_c, (rel_right - rel_left) / (2 * h), atol=2e-4)
        np.testing.assert_allclose(alpha_c, (rel_right + rel_left) / h**2, atol=2e-3)

    def test_lee_thrust_sensitivities_and_fixed_rate_attitude_coordinates(self):
        _, _, ref, state = example()
        state = perturb(state, jnp.array([.05, .08, .17, .10, -.2, .16, .11, -.10, .19, -.05, .11, .19]))
        bank = jax.tree_util.tree_map(lambda x: x[None, None], ref)
        batch = jax.tree_util.tree_map(lambda x: x[None], state)
        data = SimpleNamespace(env=SimpleNamespace(bank=bank))
        function = lambda s: lee_wrench(lee_terms(s, ref))
        jac = state_jacobians(batch, function)
        np.testing.assert_allclose(jac[0, 0, :3], -DEFAULT_GAINS.position * state.rot[:, 2], atol=1e-6)
        np.testing.assert_allclose(jac[0, 0, 3:6], -DEFAULT_GAINS.velocity * state.rot[:, 2], atol=1e-6)
        finite, _ = finite_differences(function, batch, .001)
        np.testing.assert_allclose(jac, finite, atol=4e-4, rtol=.002)
        fixed_rate, error, valid = tracking_coordinates(data, batch, jac)
        self.assertTrue(valid[0])
        eomega = state.omega - state.rot.T @ ref.rot @ ref.omega

        def change_attitude(delta):
            s = perturb(state, jnp.zeros(12).at[6:9].set(delta))
            return s._replace(omega=eomega + s.rot.T @ ref.rot @ ref.omega)

        deltas = np.eye(3) * .001
        states = [change_attitude(sign * delta) for delta in deltas for sign in (1, -1)]
        output = np.stack([np.asarray(function(s)) for s in states]).reshape(3, 2, 4)
        attitude = np.stack([np.asarray(attitude_vector(s.rot, ref.rot)) for s in states]).reshape(3, 2, 3)
        dw = (output[:, 0] - output[:, 1]).T / .002
        de = (attitude[:, 0] - attitude[:, 1]).T / .002
        np.testing.assert_allclose(fixed_rate[0], dw, atol=4e-4, rtol=.002)
        np.testing.assert_allclose(error[0] @ de, dw, atol=4e-4, rtol=.002)

    def test_actor_jacobian_uses_saved_scales_and_actual_actuator_map(self):
        trajectory, time, ref, state = example()
        nxt = reference_at(time + .05, trajectory)
        bank = jax.tree_util.tree_map(lambda x, y: jnp.stack((x, y))[None], ref, nxt)
        scale = np.linspace(.5, 3., 27)
        policy = Policy(Path("."), FixedActor(), None, dict(
            variant="L_se23_desired_thrust_extension", observation_size=27, observation_scale=scale,
        ))
        env = policy.environment(bank, .05)
        np.testing.assert_allclose(env.scale, scale)
        applied = lambda s: actor_wrench(env, policy.controller(env)(s))
        result, _, _, _, _, info = env.transition(state, policy.controller(env)(state))
        self.assertLess(float(info["torque_scale"]), .95)
        np.testing.assert_allclose(applied(state), ALLOCATION @ result.previous_action, atol=1e-7)
        np.testing.assert_allclose(feasible_wrench(applied(state)), applied(state), atol=2e-7)
        batch = jax.tree_util.tree_map(lambda x: x[None], state)
        obs, jac = actor_jacobians(SimpleNamespace(env=env, policy=policy), batch)
        self.assertEqual(obs.shape, (1, 4, 27))
        finite, _ = finite_differences(applied, batch, .002)
        np.testing.assert_allclose(jac, finite, atol=2e-4, rtol=.002)

    def test_scalar_and_diagonal_least_squares_recover_known_gains(self):
        rng = np.random.default_rng(10)
        thrust = rng.normal(size=(256, 6))
        moment = rng.normal(size=(256, 3, 2))
        offset = rng.normal(size=(256, 4))
        for diagonal, a, b in (
            (False, np.repeat([.27, .65], 3), np.array([2.1, .35])),
            (True, np.array([.22, .32, .45, .68, .70, .56]), np.array([[2.1, .35], [2.4, .37], [2.7, .41]])),
        ):
            target = offset + np.column_stack((thrust @ a, (moment * b).sum(axis=-1)))
            fitted_a, fitted_b, diagnostic = fit(thrust, moment, offset, target, diagonal=diagonal)
            np.testing.assert_allclose(fitted_a, a, atol=1e-12)
            np.testing.assert_allclose(fitted_b, b, atol=1e-12)
            self.assertEqual(diagnostic["thrust"]["rank"], 6 if diagonal else 2)
            np.testing.assert_allclose(predict(thrust, moment, offset, fitted_a, fitted_b), target, atol=2e-6)

    def test_stlsq_recovers_sparse_wrench_with_different_feature_scales(self):
        rng = np.random.default_rng(947)
        features = rng.uniform(-1, 1, (700, 6))
        x = np.column_stack((np.ones(len(features)), features * [1., .01, 50., 2., 1., .8],
                             np.zeros(len(features))))
        truth = np.zeros((x.shape[1], 4))
        truth[0, 0], truth[2, 0], truth[3, 1], truth[4, 2], truth[6, 3] = .65, 12., -.004, .09, -.35
        y = x @ truth
        coefficient, _, _, _ = select_thresholds(x[:500], y[:500], x[500:], y[500:], .0001)
        np.testing.assert_array_equal(coefficient != 0, truth != 0)
        np.testing.assert_allclose(coefficient, truth, atol=2e-5, rtol=2e-6)
        np.testing.assert_allclose(x[500:] @ coefficient, y[500:], atol=1e-6)

    def test_sparse_geometric_library_spans_requested_lee_wrench(self):
        _, _, ref, state = example()
        state = perturb(state, jnp.array([.05, .08, .17, .10, -.2, .16, .11, -.10, .19, -.05, .11, .19]))
        (polynomial, geometry, _), requested = candidate_terms(state, ref, DEFAULT_GAINS, True)
        kp, kv, kr, kw = DEFAULT_GAINS
        thrust = (MASS * GRAVITY + geometry["Fd_minus_mg"] + geometry["Fd_projection_change"]
                  - kp * (polynomial["ep_d_z"] + geometry["ep_projection_change"])
                  - kv * (polynomial["ev_d_z"] + geometry["ev_projection_change"]))
        torque = [
            -kr * (polynomial[f"eR_d_{axis}"] + geometry[f"eR_command_change_{axis}"])
            -kw * (polynomial[f"ew_d_{axis}"] + geometry[f"ew_command_change_{axis}"])
            + geometry[f"gyro_{axis}"] + geometry[f"transport_d_{axis}"]
            + geometry[f"transport_command_change_{axis}"] + geometry[f"tau_d_{axis}"]
            + geometry[f"alpha_ff_d_minus_tau_d_{axis}"] + geometry[f"alpha_ff_command_change_{axis}"]
            for axis in "xyz"
        ]
        np.testing.assert_allclose([thrust, *torque], requested, atol=4e-7)


if __name__ == "__main__":
    unittest.main()
