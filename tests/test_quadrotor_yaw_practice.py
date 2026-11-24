"""Checks on experimental controls for the independent B′ and L run."""

import unittest

import jax
import jax.numpy as jnp
import numpy as np

from envs.quadrotor_smooth import at, make_environment
from experiments.quadrotor_thrust_preview import B_PRIME, L, REWARD
from experiments.quadrotor_yaw_practice import FAMILIES, PracticeEnvironment, build_bank, yaw_start


class TestYawPractice(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.bank, cls.families, _ = build_bank(56913, 4, 16)
        cls.b = make_environment(cls.bank, B_PRIME, reward_profile=REWARD)
        cls.l = make_environment(cls.bank, L, reward_profile=REWARD)

    def test_paired_policy_observations_and_plant(self):
        b, l = (PracticeEnvironment(env, self.families) for env in (self.b, self.l))
        keys = jax.random.split(jax.random.PRNGKey(731), len(self.families))
        ids = jnp.arange(len(self.families))
        state_b, obs_b = jax.jit(jax.vmap(b.reset))(keys, ids)
        state_l, obs_l = jax.jit(jax.vmap(l.reset))(keys, ids)
        for left, right in zip(state_b, state_l):
            np.testing.assert_array_equal(left, right)
        self.assertEqual(obs_b.shape[-1], 25)
        self.assertEqual(obs_l.shape[-1], 27)
        np.testing.assert_allclose(obs_b, jnp.concatenate((obs_l[:, :22], obs_l[:, 24:]), axis=-1))
        u = jax.random.uniform(jax.random.PRNGKey(469), (len(ids), 4))
        next_b, _, reward_b, failed_b, _, _ = jax.jit(jax.vmap(b.transition))(state_b, u)
        next_l, _, reward_l, failed_l, _, _ = jax.jit(jax.vmap(l.transition))(state_l, u)
        for left, right in zip(next_b, next_l):
            np.testing.assert_array_equal(left, right)
        np.testing.assert_array_equal(reward_b, reward_l)
        np.testing.assert_array_equal(failed_b, failed_l)

    def test_angle_and_rate_starts_are_isolated(self):
        index = int(np.flatnonzero(self.families == "yaw")[0])
        key = jax.random.PRNGKey(443)
        state, _ = self.l.reset(key, index)
        ref = at(self.bank, index, 0)
        for mode in range(3):
            start = yaw_start(state, ref, key, mode)
            np.testing.assert_array_equal(start.pos, ref.pos)
            np.testing.assert_array_equal(start.vel, ref.vel)
            np.testing.assert_allclose(start.rot[:, 2], ref.rot[:, 2], atol=1e-7)
            relative = ref.rot.T @ start.rot
            error = start.omega - relative.T @ ref.omega
            angle = np.rad2deg(np.arctan2(relative[1, 0], relative[0, 0]))
            self.assertTrue(abs(angle) < .001 if mode == 1 else 10. <= abs(angle) <= 45.)
            self.assertTrue(abs(error[2]) < 1e-6 if mode == 0 else .2 <= abs(error[2]) <= .95)

    def test_reference_categories(self):
        for family in FAMILIES:
            self.assertEqual(np.count_nonzero(self.families == family), 4)
        yaw = self.families == "yaw"
        self.assertTrue(np.any(np.asarray(self.bank.yaw_rate)[yaw] != 0))
        self.assertTrue(np.all(np.asarray(self.bank.vel)[yaw] == 0))
        combined = self.families == "combined"
        self.assertTrue(np.any(np.abs(np.asarray(self.bank.acc)[combined, :, :2]) > .25))
        self.assertTrue(np.all((self.bank.action >= 0.) & (self.bank.action <= 1.)))


if __name__ == "__main__":
    unittest.main()
