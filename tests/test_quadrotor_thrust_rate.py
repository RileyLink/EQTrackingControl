"""Causality, physical feasibility and integrator checks for G/H."""

import unittest

from flax import serialization
import jax
import jax.numpy as jnp
import numpy as np

from envs.quadrotor_smooth import (
    ALLOCATION, at, make_bank, reference_at, sample_trajectory,
    transform_bank, transform_state,
)
from envs.quadrotor_thrust_rate import (
    DynamicThrustExtension, ThrustRateInputs, desired_thrust_rate, extended_action,
)
from rl.quadrotor_ppo import PPOConfig, command, make_learner


class ThrustRateTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.tr, cls.bank = make_bank(12345, 4, 100)

    def test_reference_rate_matches_automatic_derivative(self):
        tr = sample_trajectory(jax.random.PRNGKey(18))
        t = jnp.array(2.3)
        exact = jax.jacfwd(lambda time: ALLOCATION[0] @ reference_at(time, tr).action)(t)
        np.testing.assert_allclose(desired_thrust_rate(reference_at(t, tr)), exact, atol=2e-6)
        self.assertGreater(abs(float(exact)), 1e-5)

    def test_input_case_has_same_physics_as_baseline_and_causal_rates(self):
        env = ThrustRateInputs(self.bank)
        state, obs = env.reset(jax.random.PRNGKey(0), jnp.array(1))
        baseline, base_obs = env.core.reset(jax.random.PRNGKey(0), jnp.array(1))
        self.assertEqual(obs.shape, (33,))
        self.assertEqual(float(state.thrust_rate), 0.)
        np.testing.assert_array_equal(obs[:31], base_obs)
        action = jnp.array([.6, .64, .66, .68])
        before = env.observe(state)
        nxt, _, reward, failed, truncated, _ = env.transition(state, action)
        expected, _, expected_reward, expected_failed, expected_truncated, _ = env.core.transition(baseline, action)
        for a, b in zip(nxt[:7], expected):
            np.testing.assert_allclose(a, b, atol=1e-6)
        np.testing.assert_allclose(reward, expected_reward, atol=1e-6)
        self.assertEqual(bool(failed), bool(expected_failed))
        self.assertEqual(bool(truncated), bool(expected_truncated))
        expected_rate = (ALLOCATION[0] @ (action - state.previous_action)) / env.dt
        np.testing.assert_allclose(nxt.thrust_rate, expected_rate, atol=2e-6)
        np.testing.assert_array_equal(env.observe(state), before)

    def test_rate_integrator_and_rotor_feasibility_over_ramp(self):
        keys = jax.random.split(jax.random.PRNGKey(4), 2)
        thrusts = jax.random.uniform(keys[0], (200,), minval=0., maxval=1.)
        actions = jax.random.uniform(keys[1], (200, 4))
        rotors, ends, rates, scales = jax.vmap(extended_action)(thrusts, actions)
        np.testing.assert_allclose(ends, thrusts + .05 * rates, atol=1e-6)
        self.assertTrue(bool(jnp.all(jnp.abs(rates) <= 1.00001)))
        self.assertTrue(bool(jnp.all((scales >= 0) & (scales <= 1))))
        midpoint = (thrusts + ends) / 2
        np.testing.assert_allclose(rotors @ ALLOCATION[0], midpoint, atol=2e-6)
        # Commands vary linearly with F, with constant torque; verify endpoints.
        for base in [thrusts, ends]:
            endpoint_rotors = rotors + (base - midpoint)[:, None]
            self.assertTrue(bool(jnp.all(endpoint_rotors >= -2e-6)))
            self.assertTrue(bool(jnp.all(endpoint_rotors <= 1 + 2e-6)))
        _, end, rate, _ = extended_action(jnp.array(.99), jnp.ones(4))
        self.assertAlmostEqual(float(end), 1., places=6)
        self.assertLess(float(rate), 1.)

    def test_dynamic_state_uses_rate_and_previous_rate_observation(self):
        env = DynamicThrustExtension(self.bank)
        state, obs = env.reset(jax.random.PRNGKey(7), jnp.array(0))
        self.assertEqual(obs.shape, (34,))
        action = jnp.array([.75, .5, .5, .5])  # +0.5 N/s and zero torque.
        nxt, nxt_obs, *_ = env.transition(state, action)
        self.assertAlmostEqual(float(nxt.thrust), float(state.thrust) + .025, places=6)
        self.assertAlmostEqual(float(nxt.thrust_rate), .5, places=5)
        np.testing.assert_allclose(nxt_obs[-2:], [nxt.thrust_rate, nxt.thrust], atol=1e-6)
        np.testing.assert_allclose(ALLOCATION[0] @ nxt.previous_action, (state.thrust + nxt.thrust) / 2, atol=1e-6)

    def test_both_cases_preserve_yaw_translation_symmetry(self):
        yaw, translation = 1.2, jnp.array([5., -4., 2.])
        bank2 = transform_bank(self.bank, yaw, translation)
        for cls in [ThrustRateInputs, DynamicThrustExtension]:
            with self.subTest(environment=cls.__name__):
                env, transformed = cls(self.bank), cls(bank2)
                state, _ = env.reset(jax.random.PRNGKey(9), jnp.array(0))
                changed = transform_state(state, yaw, translation)
                np.testing.assert_allclose(env.observe(state), transformed.observe(changed), atol=3e-6)
                action = jnp.array([.6, .4, .7, .5])
                first, second = env.transition(state, action), transformed.transition(changed, action)
                expected = transform_state(first[0], yaw, translation)
                for a, b in zip(expected, second[0]):
                    np.testing.assert_allclose(a, b, atol=3e-6)
                np.testing.assert_allclose(first[1], second[1], atol=3e-6)
                np.testing.assert_allclose(first[2], second[2], atol=3e-6)

    def test_extended_action_initialization_and_real_ppo_update(self):
        env = DynamicThrustExtension(jax.tree_util.tree_map(lambda x: x[:2, :9], self.bank))
        cfg = PPOConfig(total_steps=16, num_envs=2, rollout_steps=8, epochs=1,
                        minibatches=2, layers=1, width=8)
        model, runner, update = make_learner(env, cfg, 0)
        mean, _, _ = model.apply(runner.train_state.params, jnp.zeros(env.obs_size))
        np.testing.assert_allclose(command(mean), [.5] * 4, atol=1e-6)
        updated, metrics = jax.block_until_ready(update(runner))
        self.assertTrue(all(np.isfinite(np.asarray(x)).all() for x in metrics.values()))
        restored = serialization.from_bytes(updated, serialization.to_bytes(updated))
        np.testing.assert_array_equal(restored.env_state.thrust, updated.env_state.thrust)
        self.assertEqual(int(updated.train_state.step), 2)


if __name__ == "__main__":
    unittest.main()
