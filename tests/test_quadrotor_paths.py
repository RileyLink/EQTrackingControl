"""Time alignment and failure handling for saved-path evaluation exports."""

from types import SimpleNamespace
import unittest

import jax
import jax.numpy as jnp
import numpy as np

from envs.quadrotor_double_thrust import DoubleThrustExtension
from envs.quadrotor_smooth import ALLOCATION, make_bank
from experiments.quadrotor_paths import selected_path
from rl.quadrotor_ppo import PPOConfig, make_evaluator, make_learner


class PathExportTests(unittest.TestCase):
    def test_selected_path_stops_at_failure_and_keeps_initial_point(self):
        bank = SimpleNamespace(pos=np.zeros((1, 4, 3)))
        frames = dict(initial_position=np.array([[.1, 0., 0.]]),
                      position=np.array([[[.2, 0., 0.]], [[.3, 0., 0.]], [[999., 0., 0.]]]),
                      active=np.array([[True], [True], [False]]),
                      failed=np.array([[False], [True], [False]]))
        path = selected_path(frames, bank, 0, .05)
        np.testing.assert_allclose(path["time"], [0., .05, .1])
        np.testing.assert_allclose(path["error"], [.1, .2, .3])
        self.assertTrue(path["failed"])
        self.assertEqual(path["position"].shape, (3, 3))

    def test_rich_recording_preserves_metrics_and_records_true_thrust_states(self):
        _, bank = make_bank(54321, 2, 8)
        env = DoubleThrustExtension(bank, symmetry=True)
        config = PPOConfig(total_steps=16, num_envs=2, rollout_steps=8,
                           epochs=1, minibatches=2, layers=1, width=8)
        model, runner, _ = make_learner(env, config, 0)
        plain = make_evaluator(env, model)
        rich = make_evaluator(env, model, record_state=True)
        plain_metrics, plain_frames = jax.block_until_ready(plain(runner.train_state.params))
        metrics, frames = jax.block_until_ready(rich(runner.train_state.params))
        np.testing.assert_allclose(plain_metrics, metrics, atol=1e-6)
        for name, values in plain_frames.items():
            np.testing.assert_allclose(values, frames[name], atol=1e-6)
        self.assertEqual(frames["rotation"].shape, (8, 2, 3, 3))
        self.assertEqual(frames["initial_position"].shape, (2, 3))
        f = np.concatenate([np.asarray(frames["initial_thrust"])[None], np.asarray(frames["thrust"])])
        rate = np.concatenate([np.asarray(frames["initial_thrust_rate"])[None], np.asarray(frames["thrust_rate"])])
        active = np.asarray(frames["active"])
        np.testing.assert_allclose(np.diff(f, axis=0)[active],
                                   (.5 * env.dt * (rate[1:] + rate[:-1]))[active], atol=1e-7)
        midpoint = f[:-1] + .5 * env.dt * rate[:-1] + env.dt * np.diff(rate, axis=0) / 8
        actual_midpoint = np.asarray(frames["rotor_command"]) @ np.asarray(ALLOCATION[0])
        np.testing.assert_allclose(actual_midpoint[active], midpoint[active], atol=1e-7)


if __name__ == "__main__":
    unittest.main()
