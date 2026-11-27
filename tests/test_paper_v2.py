"""Check actual reward contributions, activation behavior and campaign invariants."""

from contextlib import redirect_stdout
import io
import unittest
from types import SimpleNamespace
from unittest.mock import patch

import jax
import jax.numpy as jnp
import numpy as np

from rl.cluster_jobs import JOBS, job_config
from rl.eval_vis import environment_kwargs
from rl.models import clipped_relu
from rl.preflight import check_device, check_rewards
from rl.train_policy import make_environment


class PaperV2Tests(unittest.TestCase):
    def test_production_preflight_rejects_cpu_and_multiple_visible_gpus(self):
        for devices in ([SimpleNamespace(platform="cpu")],
                        [SimpleNamespace(platform="gpu"), SimpleNamespace(platform="gpu")]):
            with patch("jax.devices", return_value=devices):
                with self.assertRaisesRegex(ValueError, "exactly one visible GPU"):
                    check_device(require_gpu=True)

    def test_effective_bonus_penalty_and_weights_for_all_modes(self):
        with redirect_stdout(io.StringIO()):
            for job in JOBS:
                config = job_config(job.index)
                result = check_rewards(config)
                self.assertEqual(result["observation_inputs"], job.inputs)
                self.assertAlmostEqual(result["perfect_tracking_reward"], 1.)
                self.assertAlmostEqual(result["failure_penalty_difference"], -25.)
                self.assertEqual(result["effective"]["reward_q_rot"], .6 if job.robot == "quadrotor" else 10.)

    def test_attitude_coefficient_changes_the_actual_reward(self):
        with redirect_stdout(io.StringIO()):
            for index in (0, 4):
                config = job_config(index)
                env = make_environment(config)
                zero = make_environment(config, reward_q_rot=0.)
                state, _ = env.reset(jax.random.PRNGKey(4))
                theta = .2
                rotation = jnp.array([[jnp.cos(theta), -jnp.sin(theta), 0.],
                                      [jnp.sin(theta), jnp.cos(theta), 0.], [0., 0., 1.]])
                state = state.replace(rotm=rotation, ref_rotm=jnp.eye(3))
                action = jnp.zeros(env.action_space().shape)
                angle = float(env._get_matrix_norm(env._compute_logm(rotation)))
                difference = float(env._get_reward(state, action, action) - zero._get_reward(state, action, action))
                self.assertAlmostEqual(difference, -config["REWARD_Q_ROTM"] * angle, places=5)
                self.assertEqual(environment_kwargs(config)["reward_q_rot"], config["REWARD_Q_ROTM"])
                with self.assertRaisesRegex(TypeError, "reward_q_rot"):
                    make_environment(config, reward_q_rotm=123.)

    def test_clipped_relu_values_and_interior_gradients(self):
        inputs = jnp.array([-2., -.5, .25, .75, 2.])
        np.testing.assert_array_equal(clipped_relu(inputs), [0., 0., .25, .75, 1.])
        np.testing.assert_array_equal(jax.vmap(jax.grad(clipped_relu))(inputs), [0., 0., 1., 1., 0.])
        self.assertEqual(job_config(0)["OUT_ACTIVATION"], "clipped_relu")
        self.assertEqual(job_config(4)["OUT_ACTIVATION"], "hard_tanh")


if __name__ == "__main__":
    unittest.main()
