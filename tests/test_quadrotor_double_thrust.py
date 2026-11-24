"""Analytic reference derivatives, true double integration and actuator feasibility."""

import unittest

from flax import serialization
import jax
import jax.numpy as jnp
import numpy as np

from envs.quadrotor_smooth import (
    ALLOCATION, E3, GRAVITY, MASS, at, flat_jet, make_bank, reference_at,
    sample_trajectory, transform_bank, transform_state,
)
from envs.quadrotor_double_thrust import (
    DoubleThrustExtension, acceleration_bounds, desired_thrust_jet,
    double_extended_action, stopping_distance,
)
from rl.quadrotor_ppo import PPOConfig, command, make_evaluator, make_learner


class DoubleThrustTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.trajectories, cls.bank = make_bank(12345, 4, 100)

    def test_reference_thrust_jet_matches_automatic_differentiation(self):
        tr = sample_trajectory(jax.random.PRNGKey(18))
        thrust = lambda t: MASS * jnp.linalg.norm(flat_jet(t, tr)[2] + GRAVITY * E3)
        for time in [.3, 2.3, 4.1]:
            t = jnp.array(time)
            ref = reference_at(t, tr)
            exact = jnp.stack([thrust(t), jax.jacfwd(thrust)(t), jax.jacfwd(jax.jacfwd(thrust))(t)])
            np.testing.assert_allclose(desired_thrust_jet(ref), exact, atol=2e-7, rtol=3e-5)
            self.assertGreater(abs(float(exact[2])), 1e-5)

    def test_constant_acceleration_integrates_both_states_and_true_midpoint(self):
        rotors, force, rate, applied, _ = double_extended_action(
            jnp.array(.6), jnp.array(.2), jnp.array([.6, .5, .5, .5]))
        self.assertAlmostEqual(float(applied), 2., places=5)
        self.assertAlmostEqual(float(force), .6125, places=6)
        self.assertAlmostEqual(float(rate), .3, places=6)
        self.assertAlmostEqual(float(ALLOCATION[0] @ rotors), .605625, places=6)
        self.assertGreater(abs(float(ALLOCATION[0] @ rotors - (.6 + force) / 2)), .0005)

    def test_constraints_hold_under_long_random_control_sequences(self):
        # Includes states on both braking envelopes and both rate limits.
        speed = jnp.linspace(-1., 1., 129)
        low = stopping_distance(-speed)
        high = 1 - stopping_distance(speed)
        fraction = jnp.linspace(0., 1., 129)
        thrust = low + fraction * (high - low)
        actions = jax.random.uniform(jax.random.PRNGKey(12), (200, 129, 4))
        def step(state, action):
            f, v = state
            lower, upper = jax.vmap(acceleration_bounds)(f, v)
            rotors, nf, nv, applied, scale = jax.vmap(double_extended_action)(f, v, action)
            # Reconstruct each constant-torque rotor trajectory at many times.
            times = jnp.linspace(0., .05, 21)
            curve = f[:, None] + v[:, None] * times + .5 * applied[:, None] * times ** 2
            mid = f + v * .025 + applied * .05 ** 2 / 8
            commands = rotors[:, None, :] + (curve - mid[:, None])[:, :, None]
            diagnostics = jnp.array([commands.min(), commands.max(), nf.min(), nf.max(),
                jnp.abs(nv).max(), jnp.abs(applied).max(), (lower - upper).max(),
                (stopping_distance(nv) - (1 - nf)).max(),
                (stopping_distance(-nv) - nf).max(), scale.min()])
            return (nf, nv), diagnostics
        _, results = jax.jit(lambda: jax.lax.scan(step, (thrust, speed), actions))()
        data = np.asarray(results)
        self.assertTrue(np.isfinite(data).all())
        self.assertGreaterEqual(data[:, 0].min(), -3e-6)
        self.assertLessEqual(data[:, 1].max(), 1 + 3e-6)
        self.assertGreaterEqual(data[:, 2].min(), -3e-6)
        self.assertLessEqual(data[:, 3].max(), 1 + 3e-6)
        self.assertLessEqual(data[:, 4].max(), 1 + 3e-6)
        self.assertLessEqual(data[:, 5].max(), 10 + 3e-5)
        self.assertLessEqual(data[:, 6:9].max(), 3e-6)

    def test_torque_allocation_includes_interior_thrust_extremum(self):
        rotors, end, rate, applied, scale = double_extended_action(
            jnp.array(.99), jnp.array(.2), jnp.array([0., 1., 1., 1.]))
        self.assertAlmostEqual(float(applied), -10., places=5)
        turn = -.2 / float(applied)
        self.assertGreater(turn, 0.)
        self.assertLess(turn, .05)
        peak = .99 + .2 * turn + .5 * float(applied) * turn ** 2
        mid = .99 + .2 * .025 + float(applied) * .05 ** 2 / 8
        self.assertGreater(peak, max(.99, float(end)))
        self.assertAlmostEqual(float((rotors + peak - mid).max()), 1., places=5)
        self.assertLess(float((rotors + max(.99, float(end)) - mid).max()), .999)
        self.assertLess(float(scale), 1.)

    def test_observations_contain_extended_reference_and_true_rate_state(self):
        for symmetry, size in [(False, 44), (True, 35)]:
            env = DoubleThrustExtension(self.bank, symmetry=symmetry)
            state, obs = env.reset(jax.random.PRNGKey(4), jnp.array(0))
            ref = at(self.bank, 0, 0)
            jet = desired_thrust_jet(ref)
            np.testing.assert_allclose([state.thrust, state.thrust_rate], jet[:2], atol=1e-7)
            self.assertEqual(obs.shape, (size,))
            np.testing.assert_allclose(obs[-4:], [jet[0], jet[1], jet[0], jet[1]], atol=1e-7)
            start = 24 if symmetry else 36
            expected_input = jnp.concatenate([jet[2:3], (ALLOCATION @ ref.action)[1:]])
            np.testing.assert_allclose(obs[start:start + 4] * env.scale[start:start + 4], expected_input, atol=1e-7)
            action = jnp.array([.6, .5, .5, .5])
            nxt, nxt_obs, _, _, _, info = env.transition(state, action)
            nu = info["thrust_acceleration"]
            np.testing.assert_allclose(nxt.thrust_rate, state.thrust_rate + env.dt * nu, atol=1e-7)
            np.testing.assert_allclose(nxt.thrust, state.thrust + state.thrust_rate * env.dt + .5 * nu * env.dt ** 2, atol=1e-7)
            np.testing.assert_allclose(nxt_obs[-4:-2], [nxt.thrust, nxt.thrust_rate], atol=1e-7)

    def test_same_plant_for_full_and_symmetry_and_equivariant_transitions(self):
        yaw, translation = 1.2, jnp.array([5., -4., 2.])
        bank2 = transform_bank(self.bank, yaw, translation)
        full, sym = [DoubleThrustExtension(self.bank, symmetry=s) for s in [False, True]]
        state, _ = full.reset(jax.random.PRNGKey(9), jnp.array(0))
        sym_state, _ = sym.reset(jax.random.PRNGKey(9), jnp.array(0))
        for a, b in zip(state, sym_state):
            np.testing.assert_array_equal(a, b)
        action = jnp.array([.6, .4, .7, .5])
        first = full.transition(state, action)
        second = sym.transition(state, action)
        for a, b in zip(first[0], second[0]):
            np.testing.assert_array_equal(a, b)
        np.testing.assert_array_equal(first[2], second[2])
        changed = transform_state(state, yaw, translation)
        for env in [full, sym]:
            transformed = DoubleThrustExtension(bank2, symmetry=env.symmetry)
            if env.symmetry:
                np.testing.assert_allclose(env.observe(state), transformed.observe(changed), atol=3e-6)
            else:
                self.assertGreater(float(jnp.max(jnp.abs(env.observe(state) - transformed.observe(changed)))), 1.)
            result = transformed.transition(changed, action)
            expected = transform_state(env.transition(state, action)[0], yaw, translation)
            for a, b in zip(expected, result[0]):
                np.testing.assert_allclose(a, b, atol=3e-6)
            np.testing.assert_allclose(first[2], result[2], atol=3e-6)

    def test_rejects_infeasible_desired_thrust_derivatives(self):
        invalid = self.bank._replace(snap=jnp.ones_like(self.bank.snap) * 1e5)
        with self.assertRaisesRegex(ValueError, "thrust jet"):
            DoubleThrustExtension(invalid)

    def test_real_ppo_updates_serialization_and_later_evaluation(self):
        bank = jax.tree_util.tree_map(lambda x: x[:2, :9], self.bank)
        cfg = PPOConfig(total_steps=16, num_envs=2, rollout_steps=8, epochs=1,
                        minibatches=2, layers=1, width=8)
        for symmetry in [False, True]:
            env = DoubleThrustExtension(bank, symmetry=symmetry)
            model, runner, update = make_learner(env, cfg, 0)
            mean, _, _ = model.apply(runner.train_state.params, jnp.zeros(env.obs_size))
            np.testing.assert_allclose(command(mean), [.5] * 4, atol=1e-6)
            updated, metrics = jax.block_until_ready(update(runner))
            self.assertTrue(all(np.isfinite(np.asarray(x)).all() for x in metrics.values()))
            restored = serialization.from_bytes(updated, serialization.to_bytes(updated))
            np.testing.assert_array_equal(restored.env_state.thrust_rate, updated.env_state.thrust_rate)
            self.assertEqual(int(updated.train_state.step), 2)
            evaluator = make_evaluator(env, model)
            results, _ = jax.block_until_ready(evaluator(restored.train_state.params))
            self.assertEqual(results.shape, (2, 9))
            self.assertTrue(np.isfinite(np.asarray(results)).all())


if __name__ == "__main__":
    unittest.main()
