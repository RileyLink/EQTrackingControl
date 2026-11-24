import unittest

import jax
import jax.numpy as jnp
import numpy as np

from envs.quadrotor_desired_thrust import DesiredThrustExtension
from rl.quadrotor_ppo import QuadActorCritic
from rl.quadrotor_reppo import (
    QuadActor, ReppoConfig, constrained_actor_loss, expected_q, forward_kl, hl_gauss_targets,
    soft_td_lambda, squashed_log_probability,
)


class ReppoTests(unittest.TestCase):
    def test_actor_matches_ppo_at_initialization_for_both_observations(self):
        initial = DesiredThrustExtension.initial_action_mean
        ppo, actor = QuadActorCritic(3, 16, initial), QuadActor(3, 16, initial)
        key = jax.random.PRNGKey(19)
        for width in (25, 27):
            obs = jnp.linspace(-1., 1., width)
            mean, logstd, _ = ppo.apply(ppo.init(key, obs), obs)
            reppo_mean, reppo_logstd = actor.apply(actor.init(key, obs), obs)
            np.testing.assert_array_equal(reppo_mean, mean)
            np.testing.assert_array_equal(reppo_logstd, logstd)

    def test_squashed_log_density_has_stable_jacobian(self):
        latent = np.array([-.5, .25, .4, .75])
        mean, logstd = np.zeros(4), np.zeros(4)
        gaussian = (-.5 * latent**2 - .5 * np.log(2 * np.pi)).sum()
        expected = gaussian - np.log1p(-np.tanh(latent)**2).sum()
        self.assertAlmostEqual(float(squashed_log_probability(latent, mean, logstd)), expected, places=5)
        saturated = jnp.array([-50., 50., -35., 35.])
        logp, grad = jax.value_and_grad(squashed_log_probability)(saturated, mean, logstd)
        self.assertTrue(np.isfinite(logp))
        self.assertTrue(np.isfinite(grad).all())

    def test_forward_kl_uses_behavior_to_current_direction(self):
        old_mean, mean = jnp.array([.3, 0.]), jnp.array([0., .1])
        old_std, std = np.array([.2, .5]), np.array([.4, .25])
        old_logstd, logstd = jnp.log(old_std), jnp.log(std)
        expected = (np.log(std / old_std)
                    + (old_std**2 + (np.asarray(old_mean) - mean)**2) / (2 * std**2) - .5).sum()
        self.assertAlmostEqual(float(forward_kl(old_mean, old_logstd, mean, logstd)), float(expected), places=6)
        self.assertEqual(float(forward_kl(mean, logstd, mean, logstd)), 0.)
        self.assertNotAlmostEqual(float(forward_kl(mean, logstd, old_mean, old_logstd)), float(expected))

    def test_td_lambda_separates_failure_timeout_and_rollout_boundary(self):
        targets = soft_td_lambda(
            jnp.array([1., 2., 3., 4., 5.]), jnp.array([10., jnp.nan, 30., 40., 50.]),
            jnp.array([-1., jnp.nan, -3., -4., -5.]), jnp.array([False, True, False, False, False]),
            jnp.array([False, True, False, True, False]), gamma=.5, lam=.5, alpha=2.,
        )
        np.testing.assert_allclose(targets, [5., 2., 20.5, 28., 35.])

    def test_hl_gauss_targets_normalize_and_clip_support(self):
        centers = jnp.linspace(-2., 2., 5)
        probs = hl_gauss_targets(jnp.array([-10., -2., 0., 2., 10.]), centers)
        np.testing.assert_allclose(probs.sum(axis=-1), np.ones(5), atol=1e-6)
        np.testing.assert_array_equal(probs[0], probs[1])
        np.testing.assert_array_equal(probs[-1], probs[-2])
        np.testing.assert_allclose(probs[2], probs[2, ::-1], atol=1e-6)
        decoded = expected_q(jnp.log(probs), centers)
        self.assertAlmostEqual(float(decoded[2]), 0., places=6)
        self.assertLess(float(decoded[0]), -1.)
        self.assertGreater(float(decoded[-1]), 1.)

    def test_actor_gradient_and_dual_updates_follow_the_kl_gate(self):
        config = ReppoConfig(4096)
        log_alpha, log_beta = jnp.log(.01), jnp.log(.02)

        def objective(q, kl, alpha, beta):
            return constrained_actor_loss(q, jnp.array([3., 3.]), kl, alpha, beta, config)[0]

        grad_q, grad_kl, grad_alpha, grad_beta = jax.grad(objective, (0, 1, 2, 3))(
            jnp.ones(2), jnp.array([.01, .3]), log_alpha, log_beta)
        np.testing.assert_allclose(grad_q, [-.5, 0.])
        np.testing.assert_allclose(grad_kl, [0., .01], atol=1e-6)
        self.assertLess(float(grad_alpha), 0.)
        self.assertLess(float(grad_beta), 0.)


if __name__ == "__main__":
    unittest.main()
