"""Reward reweighting, legacy compatibility and consistent critic reward units."""

import unittest

import jax
import jax.numpy as jnp
import numpy as np

from envs.quadrotor_smooth import (
    MASS, GRAVITY, REWARD_PROFILES, POSITION_FOCUS_VARIANTS, WRENCH_VARIANTS, RewardSettings,
    make_bank, make_environment, transform_bank, transform_state,
)
from rl.quadrotor_ppo import PPOConfig, make_learner


class RewardProfileTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        _, bank = make_bank(12345, 2, 8)
        cls.bank = bank
        cls.hover = bank._replace(
            pos=jnp.zeros_like(bank.pos), vel=jnp.zeros_like(bank.vel),
            rot=jnp.broadcast_to(jnp.eye(3), bank.rot.shape), omega=jnp.zeros_like(bank.omega),
            action=jnp.full_like(bank.action, MASS * GRAVITY), acc=jnp.zeros_like(bank.acc),
            jerk=jnp.zeros_like(bank.jerk), snap=jnp.zeros_like(bank.snap),
            yaw=jnp.zeros_like(bank.yaw), yaw_rate=jnp.zeros_like(bank.yaw_rate), yaw_acc=jnp.zeros_like(bank.yaw_acc))

    def test_position_penalty_is_50_times_larger_at_small_errors(self):
        for variant in POSITION_FOCUS_VARIANTS:
            for error in [.05, .3]:
                for profile, expected in [("legacy", -.02 * error), ("position_focus_v1", -error)]:
                    env = make_environment(self.hover, variant, reward_profile=profile)
                    state, _ = env.reset(jax.random.PRNGKey(0), jnp.array(0))
                    state = state._replace(pos=jnp.array([error, 0., 0.]), vel=jnp.zeros(3),
                                           rot=jnp.eye(3), omega=jnp.zeros(3))
                    action = (jnp.array([MASS * GRAVITY, .5, .5, .5]) if variant in WRENCH_VARIANTS
                              else jnp.full(4, MASS * GRAVITY))
                    result = env.transition(state, action)
                    self.assertFalse(bool(result[3]))
                    np.testing.assert_allclose(result[2], expected, atol=2e-6)

    def test_reward_profile_does_not_change_dynamics_or_observations(self):
        for variant in POSITION_FOCUS_VARIANTS:
            old = make_environment(self.bank, variant)
            new = make_environment(self.bank, variant, reward_profile="position_focus_v1")
            a, obs_a = old.reset(jax.random.PRNGKey(4), jnp.array(0))
            b, obs_b = new.reset(jax.random.PRNGKey(4), jnp.array(0))
            np.testing.assert_array_equal(obs_a, obs_b)
            action = jnp.array([.6, .45, .55, .5])
            first, second = old.transition(a, action), new.transition(b, action)
            for x, y in zip(first[0], second[0]): np.testing.assert_array_equal(x, y)
            np.testing.assert_array_equal(first[1], second[1])
            self.assertEqual(bool(first[3]), bool(second[3]))
            info = second[5]
            difference = -.98 * jnp.sqrt(info["position_sq"]) - .099 * jnp.sqrt(info["velocity_sq"])
            np.testing.assert_allclose(second[2] - first[2], difference, atol=3e-6)
            if hasattr(new, "core"):
                self.assertEqual(new.reward_settings, new.core.reward_settings)

    def test_failure_cost_is_applied_before_reset_and_exceeds_valid_discounted_cost(self):
        env = make_environment(self.hover, "B_symmetry", reward_profile="position_focus_v1")
        state, _ = env.reset(jax.random.PRNGKey(2), jnp.array(0))
        state = state._replace(pos=jnp.array([6., 0., 0.]), vel=jnp.zeros(3), rot=jnp.eye(3), omega=jnp.zeros(3))
        nxt, _, reward, failed, _, info = env.transition(state, jnp.full(4, MASS * GRAVITY))
        self.assertTrue(bool(failed))
        np.testing.assert_allclose(reward, -1506., atol=2e-4)
        self.assertGreater(float(nxt.pos[0]), 5.)
        self.assertEqual(float(info["failure_cost"]), 1500.)
        q = env.reward_settings
        max_valid_cost = 5*q.position + 8*q.velocity + np.pi/2*q.attitude + 10*q.omega + 2*q.rotor_command
        self.assertGreater(q.failure_penalty, max_valid_cost / (1 - .995))

    def test_new_reward_preserves_yaw_translation_symmetry(self):
        yaw, translation = 1.2, jnp.array([5., -4., 2.])
        for variant in ["B_symmetry", "L_symmetry_desired_thrust_extension"]:
            env = make_environment(self.bank, variant, reward_profile="position_focus_v1")
            other = make_environment(transform_bank(self.bank, yaw, translation), variant, reward_profile="position_focus_v1")
            state, _ = env.reset(jax.random.PRNGKey(3), jnp.array(0))
            action = jnp.array([.64, .5, .52, .48])
            first = env.transition(state, action)
            second = other.transition(transform_state(state, yaw, translation), action)
            np.testing.assert_allclose(first[1], second[1], atol=3e-6)
            np.testing.assert_allclose(first[2], second[2], atol=3e-6)

    def test_critic_scaling_matches_explicitly_scaled_reward_with_raw_logs(self):
        env = make_environment(self.bank, "B_symmetry", reward_profile="position_focus_v1")
        scaled = make_environment(self.bank, "B_symmetry", reward_profile="position_focus_v1")
        q = env.reward_settings
        scaled.reward_settings = RewardSettings(*(x / q.learning_scale for x in q[:-1]), learning_scale=1.)
        scaled.learning_reward_scale = 1.
        cfg = PPOConfig(total_steps=16, num_envs=2, rollout_steps=8, epochs=1, minibatches=2, layers=1, width=8)
        _, first, update = make_learner(env, cfg, 0)
        _, second, other_update = make_learner(scaled, cfg, 0)
        first, metrics = jax.block_until_ready(update(first))
        second, other_metrics = jax.block_until_ready(other_update(second))
        for a, b in zip(jax.tree_util.tree_leaves(first.train_state.params), jax.tree_util.tree_leaves(second.train_state.params)):
            np.testing.assert_allclose(a, b, atol=2e-5, rtol=2e-5)
        np.testing.assert_allclose(metrics["learning_step_reward"], metrics["step_reward"] / 50, atol=1e-7)
        np.testing.assert_allclose(metrics["learning_step_reward"], other_metrics["step_reward"], atol=1e-7)
        cost = sum(metrics["mean_" + name] for name in ["position_cost", "velocity_cost", "attitude_cost",
                   "omega_cost", "rotor_command_cost", "failure_cost"])
        np.testing.assert_allclose(metrics["step_reward"], -cost, atol=1e-6)

    def test_real_updates_for_only_the_four_supported_profiles(self):
        cfg = PPOConfig(total_steps=16, num_envs=2, rollout_steps=8, epochs=1, minibatches=2, layers=1, width=8)
        for variant in POSITION_FOCUS_VARIANTS:
            env = make_environment(self.bank, variant, reward_profile="position_focus_v1")
            _, state, update = make_learner(env, cfg, 1)
            state, metrics = jax.block_until_ready(update(state))
            self.assertEqual(int(state.train_state.step), 2)
            self.assertTrue(all(np.isfinite(np.asarray(value)).all() for value in metrics.values()))
        with self.assertRaisesRegex(ValueError, "restricted"):
            make_environment(self.bank, "C_full_derivatives", reward_profile="position_focus_v1")
        with self.assertRaisesRegex(ValueError, "Unknown"):
            make_environment(self.bank, "A_full", reward_profile="missing")


if __name__ == "__main__":
    unittest.main()
