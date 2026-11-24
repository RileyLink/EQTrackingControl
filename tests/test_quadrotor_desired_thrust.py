"""Desired-only extension observations, direct f/tau actuation and PPO wiring."""

import unittest

from flax import serialization
import jax
import jax.numpy as jnp
import numpy as np

from envs.quadrotor_smooth import (
    ALLOCATION, MASS, GRAVITY, E3, at, make_bank, make_environment, transform_bank, transform_state,
)
from envs.quadrotor_double_thrust import desired_thrust_jet
from envs.quadrotor_desired_thrust import DesiredThrustExtension, direct_wrench_action
from envs.quadrotor_thrust_rate import TORQUE_LIMITS
from rl.quadrotor_ppo import PPOConfig, command, make_evaluator, make_learner


class DesiredThrustTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        _, cls.bank = make_bank(12345, 4, 100)

    def test_exact_observation_order_and_no_actual_thrust_history(self):
        env = make_environment(self.bank, "K_full_desired_thrust_extension")
        state, obs = env.reset(jax.random.PRNGKey(4), jnp.array(0))
        ref = at(self.bank, 0, 0)
        expected = jnp.concatenate([state.pos, state.vel, state.rot.ravel(), state.omega,
            ref.pos, ref.vel, ref.rot.ravel(), ref.omega,
            desired_thrust_jet(ref), (ALLOCATION @ ref.action)[1:]])
        self.assertEqual(obs.shape, (42,))
        np.testing.assert_allclose(obs * env.scale, expected, atol=1e-6)
        self.assertFalse(hasattr(state, "thrust"))
        self.assertFalse(hasattr(state, "thrust_rate"))
        np.testing.assert_array_equal(env.observe(state), env.observe(
            state._replace(previous_action=jnp.zeros(4))))

    def test_wrench_allocation_preserves_force_and_feasible_torque(self):
        actions = jax.random.uniform(jax.random.PRNGKey(7), (200, 4))
        rotors, force, scale = jax.vmap(direct_wrench_action)(actions)
        wrench = rotors @ ALLOCATION.T
        np.testing.assert_allclose(wrench[:, 0], actions[:, 0], atol=2e-7)
        np.testing.assert_allclose(wrench[:, 1:], scale[:, None] * (2 * actions[:, 1:] - 1) * TORQUE_LIMITS, atol=3e-7)
        self.assertTrue(bool(jnp.all((rotors >= -2e-7) & (rotors <= 1 + 2e-7))))
        for thrust in [0., 1.]:
            rotors, force, scale = direct_wrench_action(jnp.array([thrust, 1., 1., 1.]))
            np.testing.assert_allclose(rotors, jnp.ones(4) * thrust, atol=2e-7)
            self.assertEqual(float(scale), 0.)

    def test_symmetry_observations_and_same_physics_as_full_case(self):
        full = DesiredThrustExtension(self.bank)
        sym = make_environment(self.bank, "L_symmetry_desired_thrust_extension")
        state, _ = full.reset(jax.random.PRNGKey(2), jnp.array(1))
        paired, obs = sym.reset(jax.random.PRNGKey(2), jnp.array(1))
        self.assertEqual(obs.shape, (33,))
        ref = at(self.bank, 1, 0)
        expected = jnp.concatenate([state.rot.T @ (ref.pos-state.pos),
            state.rot.T @ state.vel, state.rot.T @ ref.vel, (state.rot.T @ ref.rot).ravel(),
            state.omega, ref.omega, state.rot.T @ E3, desired_thrust_jet(ref),
            (ALLOCATION @ ref.action)[1:]])
        np.testing.assert_allclose(obs * sym.scale, expected, atol=1e-6)
        for a, b in zip(state, paired): np.testing.assert_array_equal(a, b)
        yaw, translation = 1.2, jnp.array([5., -4., 2.])
        transformed = DesiredThrustExtension(transform_bank(self.bank, yaw, translation), symmetry=True)
        changed = transform_state(state, yaw, translation)
        np.testing.assert_allclose(obs, transformed.observe(changed), atol=3e-6)
        action = jnp.array([.63, .45, .6, .4])
        a, b, c = full.transition(state, action), sym.transition(paired, action), transformed.transition(changed, action)
        for x, y in zip(a[0], b[0]): np.testing.assert_array_equal(x, y)
        for x, y in zip(transform_state(b[0], yaw, translation), c[0]):
            np.testing.assert_allclose(x, y, atol=3e-6)
        np.testing.assert_allclose(b[1], c[1], atol=3e-6)
        np.testing.assert_allclose(a[2], b[2], atol=1e-7)
        np.testing.assert_allclose(b[2], c[2], atol=3e-6)
        self.assertFalse(full.experiment_metadata["symmetry"])
        self.assertTrue(sym.experiment_metadata["symmetry"])

    def test_reference_wrench_recovers_original_reference_rotors(self):
        ref = at(self.bank, 0, 0)
        force = desired_thrust_jet(ref)[0]
        torque = (ALLOCATION @ ref.action)[1:]
        action = jnp.concatenate([force[None], .5 * (torque / TORQUE_LIMITS + 1)])
        rotors, _, scale = direct_wrench_action(action)
        np.testing.assert_allclose(rotors, ref.action, atol=3e-7)
        self.assertAlmostEqual(float(scale), 1., places=6)

    def test_b_prime_is_l_with_only_current_desired_wrench(self):
        profile = "position_focus_v1"
        baseline = make_environment(self.bank, "B_prime_se23_wrench", reward_profile=profile)
        extended = make_environment(self.bank, "L_se23_desired_thrust_extension", reward_profile=profile)
        state, obs = baseline.reset(jax.random.PRNGKey(8), jnp.array(1))
        full_obs = extended.observe(state)
        np.testing.assert_array_equal(obs, jnp.concatenate([full_obs[:22], full_obs[24:]]))
        self.assertEqual(baseline.initial_action_mean, extended.initial_action_mean)
        self.assertEqual(baseline.experiment_metadata["desired_reference_extension_order"], 0)
        action = jnp.array([.63, .95, .87, .23])
        first, second = baseline.transition(state, action), extended.transition(state, action)
        for actual, expected in zip(first[0], second[0]):
            np.testing.assert_array_equal(actual, expected)
        np.testing.assert_array_equal(first[2:5], second[2:5])

    def test_physics_matches_core_and_force_changes_without_integrator(self):
        env = DesiredThrustExtension(self.bank)
        state, _ = env.reset(jax.random.PRNGKey(8), jnp.array(0))
        for force in [.1, .95]:
            action = jnp.array([force, .51, .49, .5])
            rotors, _, _ = direct_wrench_action(action)
            expected = env.core.transition(state, rotors)
            result = env.transition(state, action)
            for a, b in zip(result[0], expected[0]):
                np.testing.assert_allclose(a, b, atol=1e-7)
            np.testing.assert_allclose(result[2], expected[2], atol=1e-7)
            np.testing.assert_allclose(ALLOCATION[0] @ result[0].previous_action, force, atol=2e-7)
            state = result[0]
        self.assertFalse(hasattr(state, "thrust_rate"))

    def test_real_ppo_update_checkpoint_and_path_export(self):
        bank = jax.tree_util.tree_map(lambda x: x[:2, :9], self.bank)
        cfg = PPOConfig(total_steps=16, num_envs=2, rollout_steps=8, epochs=1,
                        minibatches=2, layers=1, width=8)
        for symmetry in [False, True]:
            env = DesiredThrustExtension(bank, symmetry=symmetry)
            model, runner, update = make_learner(env, cfg, 0)
            mean, _, _ = model.apply(runner.train_state.params, jnp.zeros(env.obs_size))
            np.testing.assert_allclose(command(mean), [MASS * GRAVITY, .5, .5, .5], atol=2e-7)
            updated, metrics = jax.block_until_ready(update(runner))
            self.assertTrue(all(np.isfinite(np.asarray(value)).all() for value in metrics.values()))
            restored = serialization.from_bytes(updated, serialization.to_bytes(updated))
            np.testing.assert_array_equal(restored.obs, updated.obs)
            evaluator = make_evaluator(env, model, record_state=True)
            results, frames = jax.block_until_ready(evaluator(restored.train_state.params))
            self.assertTrue(np.isfinite(np.asarray(results)).all())
            self.assertNotIn("thrust_rate", frames)
            self.assertEqual(frames["position"].shape, (8, 2, 3))


if __name__ == "__main__":
    unittest.main()
