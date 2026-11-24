"""Physical consistency, symmetry and PPO-boundary checks for the comparison."""

import unittest

import jax
import jax.numpy as jnp
import numpy as np

from envs.quadrotor_double_thrust import desired_thrust_jet
from envs.quadrotor_smooth import (
    ALLOCATION, E3, GRAVITY, INERTIA, MASS, VARIANTS, SmoothQuadrotor,
    actual_derivative_features, advance, at, flat_jet, hat, make_bank, observation, reference_at,
    make_environment, rotation_exp, sample_trajectory, transform_bank, transform_state,
)
from rl.quadrotor_ppo import (PPOConfig, advantages_and_returns, command,
                              log_probability, make_evaluator, make_learner)


class QuadrotorComparisonTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.trajectories, cls.bank = make_bank(12345, 8, 100)

    def test_analytic_derivatives_match_autodiff(self):
        tr = sample_trajectory(jax.random.PRNGKey(4))
        t = jnp.array(1.7)
        jet = flat_jet(t, tr)
        derivative = lambda time: flat_jet(time, tr)[0]
        for order in range(1, 5):
            derivative = jax.jacfwd(derivative)
            np.testing.assert_allclose(derivative(t), jet[order], rtol=2e-5, atol=2e-6)
        yaw = lambda time: flat_jet(time, tr)[5]
        np.testing.assert_allclose(jax.jacfwd(yaw)(t), jet[6], atol=1e-6)
        np.testing.assert_allclose(jax.jacfwd(jax.jacfwd(yaw))(t), jet[7], atol=1e-6)

    def test_flatness_matches_continuous_dynamics(self):
        tr = sample_trajectory(jax.random.PRNGKey(9))
        t = jnp.array(2.3)
        ref = reference_at(t, tr)
        dR = jax.jacfwd(lambda time: reference_at(time, tr).rot)(t)
        domega = jax.jacfwd(lambda time: reference_at(time, tr).omega)(t)
        wrench = ALLOCATION @ ref.action
        acc = ref.rot @ (wrench[0] / MASS * E3) - GRAVITY * E3
        alpha = jnp.linalg.solve(INERTIA, wrench[1:] - jnp.cross(ref.omega, INERTIA @ ref.omega))
        np.testing.assert_allclose(ref.rot.T @ ref.rot, jnp.eye(3), atol=1e-6)
        self.assertAlmostEqual(float(jnp.linalg.det(ref.rot)), 1., places=5)
        np.testing.assert_allclose(acc, ref.acc, atol=3e-6)
        np.testing.assert_allclose(dR, ref.rot @ hat(ref.omega), atol=2e-6)
        np.testing.assert_allclose(alpha, domega, atol=2e-6)

    def test_reference_feasibility_and_nontrivial_yaw(self):
        _, bank = make_bank(65432, 32, 400, frequency_scale=1.3)
        self.assertTrue(np.isfinite(np.asarray(bank.action)).all())
        self.assertGreater(float(bank.action.min()), 0.)
        self.assertLess(float(bank.action.max()), 1.)
        self.assertGreater(float(jnp.max(jnp.abs(bank.jerk))), .01)
        self.assertGreater(float(jnp.max(jnp.abs(bank.snap))), .01)
        self.assertGreater(float(jnp.max(jnp.abs(bank.yaw_rate))), .01)

    def test_all_observation_shapes_and_invariance(self):
        transformed = transform_bank(self.bank, 1.2, jnp.array([5., -4., 2.]))
        from experiments.quadrotor_comparison import OBSERVATION_SIZES, LABELS
        sizes = [40, 31, 54, 45, 69, 60, 33, 34, 44, 35, 42, 33, 25, 27, 25]
        self.assertEqual(len(sizes), len(VARIANTS))
        for (name, config), size in zip(VARIANTS.items(), sizes):
            with self.subTest(variant=name):
                env = make_environment(self.bank, name)
                state, obs = env.reset(jax.random.PRNGKey(10), jnp.array(2))
                changed_state = transform_state(state, 1.2, jnp.array([5., -4., 2.]))
                changed_env = make_environment(transformed, name)
                changed_obs = changed_env.observe(changed_state)
                self.assertEqual(obs.shape, (size,))
                self.assertEqual(OBSERVATION_SIZES[name], size)
                self.assertIn(f"(obs {size}, act 4)", LABELS[name])
                self.assertTrue(np.isfinite(obs).all())
                if config.symmetry:
                    np.testing.assert_allclose(obs, changed_obs, atol=2e-6)
                else:
                    self.assertGreater(float(jnp.max(jnp.abs(obs - changed_obs))), 1.)

    def test_se23_errors_and_paired_transitions_under_arbitrary_rotation(self):
        Q = rotation_exp(jnp.array([.7, -.4, .5]))
        shift = jnp.array([2., -3., .4])
        boost = jnp.array([1.2, -.6, .3])
        # When rotating trajectories while retaining world gravity, add the
        # common acceleration required to keep both motions feasible.
        frame_acc = GRAVITY * (Q @ E3 - E3)
        times = jnp.arange(self.bank.pos.shape[1])[:, None] * .05
        rotate = lambda x: jnp.einsum("ij,...j->...i", Q, x)
        changed_bank = self.bank._replace(
            pos=rotate(self.bank.pos) + shift + times * boost + .5 * times ** 2 * frame_acc,
            vel=rotate(self.bank.vel) + boost + times * frame_acc,
            rot=Q @ self.bank.rot, acc=rotate(self.bank.acc) + frame_acc,
            jerk=rotate(self.bank.jerk), snap=rotate(self.bank.snap),
        )
        for variant, baseline in [("B_se23", "B_symmetry"),
                                  ("L_se23_desired_thrust_extension", "L_symmetry_desired_thrust_extension")]:
            with self.subTest(variant=variant):
                env, transformed, original = [make_environment(bank, name, reward_profile="position_focus_v1")
                    for bank, name in [(self.bank, variant), (changed_bank, variant), (self.bank, baseline)]]
                key = jax.random.PRNGKey(22)
                state, obs = env.reset(key, jnp.array(1))
                changed_state, changed_obs = transformed.reset(key, jnp.array(1))
                ref = at(self.bank, 1, 0)
                relative_rot = ref.rot.T @ state.rot
                np.testing.assert_allclose(obs[:9], relative_rot.ravel(), atol=1e-6)
                np.testing.assert_allclose(2 * obs[9:12], ref.rot.T @ (state.pos - ref.pos), atol=1e-6)
                nu, nu_d = state.rot.T @ state.vel, ref.rot.T @ ref.vel
                np.testing.assert_allclose(obs[12:15], nu - relative_rot.T @ nu_d, atol=1e-6)
                np.testing.assert_allclose(obs[15:18], state.omega - relative_rot.T @ ref.omega, atol=1e-6)
                np.testing.assert_array_equal(obs[18:21], ref.omega)
                reference_input = (ref.action if variant == "B_se23" else
                                   jnp.concatenate([desired_thrust_jet(ref), (ALLOCATION @ ref.action)[1:]]))
                np.testing.assert_allclose((obs * env.scale)[21:], reference_input, atol=1e-6)
                np.testing.assert_allclose(obs, changed_obs, atol=2e-6)

                action = jnp.array([.66, .49, .52, .51])
                result = env.transition(state, action)
                paired = transformed.transition(changed_state, action)
                reference = original.transition(state, action)
                for actual, expected in zip(result[0], reference[0]):
                    np.testing.assert_array_equal(actual, expected)
                np.testing.assert_array_equal(result[2:5], reference[2:5])
                np.testing.assert_allclose(result[1], paired[1], atol=3e-6)
                np.testing.assert_allclose(result[2], paired[2], atol=3e-6)
                np.testing.assert_array_equal(result[3:5], paired[3:5])

    def test_dynamics_reward_and_termination_are_symmetric(self):
        yaw, translation = 2.8, jnp.array([5., -4., 2.])
        env = SmoothQuadrotor(self.bank, True, True, actual_derivatives=True)
        transformed_env = SmoothQuadrotor(transform_bank(self.bank, yaw, translation), True, True, actual_derivatives=True)
        state, _ = env.reset(jax.random.PRNGKey(10), jnp.array(1))
        transformed_state = transform_state(state, yaw, translation)
        action = jnp.array([.61, .64, .65, .68])
        first = env.transition(state, action)
        second = transformed_env.transition(transformed_state, action)
        expected = transform_state(first[0], yaw, translation)
        for a, b in zip(expected, second[0]):
            np.testing.assert_allclose(a, b, atol=3e-6)
        np.testing.assert_allclose(first[1], second[1], atol=3e-6)
        np.testing.assert_allclose(first[2], second[2], atol=3e-6)
        self.assertEqual(bool(first[3]), bool(second[3]))

    def test_midpoint_tracks_feasible_reference_without_drift(self):
        # Feedforward from an exact initial state tests the physical reference map
        # independently of PPO learning, including its motor allocation conventions.
        env = SmoothQuadrotor(self.bank)
        ref = at(self.bank, 0, 0)
        state, _ = env.reset(jax.random.PRNGKey(0), jnp.array(0))
        state = state._replace(pos=ref.pos, vel=ref.vel, rot=ref.rot, omega=ref.omega)
        tr = jax.tree_util.tree_map(lambda x: x[0], self.trajectories)
        def step(state, index):
            midpoint_ref = reference_at((index + .5) * env.dt, tr)
            state = advance(state, midpoint_ref.action, env.dt)
            return state, None
        final, _ = jax.jit(lambda s: jax.lax.scan(step, s, jnp.arange(100)))(state)
        expected = at(self.bank, 0, 100)
        self.assertLess(float(jnp.linalg.norm(final.pos - expected.pos)), .01)
        np.testing.assert_allclose(final.rot.T @ final.rot, jnp.eye(3), atol=2e-5)

    def test_terminal_reward_precedes_reset(self):
        env = SmoothQuadrotor(self.bank)
        state, _ = env.reset(jax.random.PRNGKey(11), jnp.array(0))
        state = state._replace(pos=state.pos + jnp.array([10., 0., 0.]))
        terminal, obs, reward, failed, _, info = env.transition(state, at(self.bank, 0, 0).action)
        self.assertTrue(bool(failed))
        self.assertLess(float(reward), -25.)
        self.assertGreater(float(jnp.linalg.norm(terminal.pos - at(self.bank, 0, 1).pos)), 5.)
        self.assertEqual(int(terminal.step), 1)

    def test_time_limits_bootstrap_and_stop_gae(self):
        # At t=0 a time limit bootstraps V(terminal_obs)=10 but must not
        # accumulate the following reset episode's advantage at t=1.
        r = jnp.array([[1.], [100.]])
        v = jnp.array([[2.], [3.]])
        nv = jnp.array([[10.], [20.]])
        terminated = jnp.array([[False], [True]])
        truncated = jnp.array([[True], [False]])
        adv, target = advantages_and_returns(r, v, nv, terminated, truncated, .9, .95)
        np.testing.assert_allclose(adv[:, 0], [8., 97.], atol=1e-6)
        np.testing.assert_allclose(target[:, 0], [10., 100.], atol=1e-6)

    def test_bounded_actions_and_finite_likelihoods(self):
        latent = jnp.array([[-5., -.2, .5, 5.]])
        actions = command(latent)
        self.assertTrue(bool(jnp.all((actions >= 0) & (actions <= 1))))
        self.assertTrue(bool(jnp.all(jnp.isfinite(log_probability(latent, jnp.zeros_like(latent), jnp.zeros(4))))))

    def test_actual_derivatives_match_held_input_dynamics(self):
        env = SmoothQuadrotor(self.bank)
        state, _ = env.reset(jax.random.PRNGKey(8), jnp.array(0))
        state = state._replace(previous_action=jnp.array([.61, .67, .66, .63]))
        actual = actual_derivative_features(state)
        wrench = ALLOCATION @ state.previous_action
        def flow(R, w):
            return R @ hat(w), jnp.linalg.solve(INERTIA, wrench[1:] - jnp.cross(w, INERTIA @ w))
        def acceleration(R, w):
            return wrench[0] / MASS * R[:, 2] - GRAVITY * E3
        def jerk(R, w):
            return jax.jvp(acceleration, (R, w), flow(R, w))[1]
        def yaw(R, w):
            return jnp.arctan2(-R[0, 1], R[1, 1])
        def yaw_rate(R, w):
            return jax.jvp(yaw, (R, w), flow(R, w))[1]
        point = (state.rot, state.omega)
        np.testing.assert_allclose(actual.acc, acceleration(*point), atol=2e-6)
        np.testing.assert_allclose(actual.jerk, jerk(*point), atol=2e-6)
        np.testing.assert_allclose(actual.snap, jax.jvp(jerk, point, flow(*point))[1], atol=2e-6)
        np.testing.assert_allclose(actual.yaw_rate, yaw_rate(*point), atol=2e-6)
        np.testing.assert_allclose(actual.yaw_acc, jax.jvp(yaw_rate, point, flow(*point))[1], atol=2e-6)
        self.assertEqual(float(actual.heading_valid), 1.)
        # Derivatives do not inspect a future action. They change with the last
        # applied command; next-observation construction must use the new hold.
        next_state, *_ = env.transition(state, jnp.array([.7, .6, .6, .7]))
        np.testing.assert_allclose(next_state.previous_action, [.7, .6, .6, .7], atol=1e-6)
        self.assertGreater(float(jnp.linalg.norm(actual.acc - actual_derivative_features(
            state._replace(previous_action=jnp.ones(4) * .9)).acc)), 1.)

    def test_actual_heading_matches_reference_and_handles_singularities(self):
        env = SmoothQuadrotor(self.bank)
        state, _ = env.reset(jax.random.PRNGKey(8), jnp.array(0))
        ref = at(self.bank, 0, 0)
        actual = actual_derivative_features(state._replace(rot=ref.rot, omega=ref.omega))
        np.testing.assert_allclose(actual.heading, [jnp.cos(ref.yaw), jnp.sin(ref.yaw), 0.], atol=2e-6)
        pole = state._replace(rot=rotation_exp(jnp.array([jnp.pi / 2, 0., 0.])))
        masked = actual_derivative_features(pole)
        self.assertEqual(float(masked.heading_valid), 0.)
        self.assertTrue(all(np.isfinite(np.asarray(x)).all() for x in masked))
        np.testing.assert_array_equal(masked.heading, np.zeros(3))
        self.assertEqual(float(masked.yaw_rate), 0.)

    def test_jitted_update_and_evaluation_checkpoint_roundtrip(self):
        from flax import serialization
        # One real PPO update and complete evaluation scan catch wiring failures
        # that analytic dynamics/observation tests cannot exercise.
        bank = jax.tree_util.tree_map(lambda x: x[:2, :9], self.bank)
        env = SmoothQuadrotor(bank, symmetry=True, derivatives=True, actual_derivatives=True)
        config = PPOConfig(total_steps=16, num_envs=2, rollout_steps=8,
                           epochs=1, minibatches=2, layers=1, width=8)
        model, runner, update = make_learner(env, config, 0)
        runner, metrics = jax.block_until_ready(update(runner))
        self.assertEqual(int(runner.train_state.step), 2)
        self.assertTrue(all(np.isfinite(np.asarray(x)).all() for x in metrics.values()))
        restored = serialization.from_bytes(runner, serialization.to_bytes(runner))
        evaluator = make_evaluator(env, model)
        results, frames = jax.block_until_ready(evaluator(restored.train_state.params))
        self.assertEqual(results.shape, (2, 9))
        self.assertEqual(frames["position"].shape, (8, 2, 3))
        self.assertTrue(np.isfinite(np.asarray(results)).all())


if __name__ == "__main__":
    unittest.main()
