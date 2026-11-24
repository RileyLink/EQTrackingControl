"""Exercise actual checkpoint loading, controlled rollouts, and the ROS file contract."""

from contextlib import redirect_stdout
import io
import json
import os
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

import jax
import jax.numpy as jnp
import numpy as np
import orbax.checkpoint as ocp

import envs.registry as registry
from rl import eval_vis
from rl.models import ActorCritic


def constant_policy(robot, mode, action):
    task = "random_lissajous" if robot == "particle" else "random_walk"
    env = registry.get_env_by_name(
        f"{robot}_{task}", equivariant=mode, symmetry_type=2,
        terminate_on_error=False, max_time=float("inf"),
    )
    model = ActorCritic(len(action), num_layers=1, num_nodes=4, out_activation="hard_tanh")
    params = model.init(jax.random.PRNGKey(0), jnp.zeros(env.observation_space().shape))["params"]
    params = jax.tree_util.tree_map(jnp.zeros_like, params)
    params["Dense_1"]["bias"] = jnp.array(action)
    return eval_vis.Policy("fixture", Path("fixture"), {"EQUIVARIANT": mode}, env, model, params)


class EvalVisTests(unittest.TestCase):
    def setUp(self):
        # Orbax 0.6.4's throughput logger can divide by a zero-duration Windows
        # clock tick for tiny fixtures. Disable only that third-party diagnostic.
        diagnostic = patch("orbax.checkpoint._src.handlers.base_pytree_checkpoint_handler._log_io_per_sec_metric")
        diagnostic.start()
        self.addCleanup(diagnostic.stop)

    def save_fixture(self, root, robot, mode):
        # Controlled, untrained weights: every observation map produces the same command.
        width = {"quadrotor": 4, "astrobee": 6, "particle": 3}[robot]
        policy = constant_policy(robot, mode, np.full(width, .2))
        params = jax.tree_util.tree_map(lambda value: jnp.stack([value, value]), policy.params)
        params["Dense_1"]["bias"] = jnp.stack([jnp.full(width, .1), jnp.full(width, -.1)])
        folder = root / f"{robot}_{mode}"
        folder.mkdir()
        config = dict(env_name=f"{robot}_random_walk", EQUIVARIANT=mode, SYMMETRY_TYPE=2,
                      NUM_LAYERS=1, NUM_NODES=4, ACTIVATION="leaky_relu", OUT_ACTIVATION="relu",
                      TERMINATION_BOUND=.001)
        (folder / "config.txt").write_text(repr(config))
        ocp.PyTreeCheckpointer().save(str(folder / "model_final"), [{"params": {"params": params}}])
        return folder

    def test_all_four_modes_restore_and_export_for_both_robots(self):
        for robot, mode_ids in ((robot, eval_vis.MODE_IDS[robot]) for robot in ("quadrotor", "astrobee")):
            with self.subTest(robot=robot), tempfile.TemporaryDirectory() as temp, redirect_stdout(io.StringIO()):
                root = Path(temp)
                requests = {label: self.save_fixture(root, robot, mode) for label, mode in mode_ids.items()}
                output = root / "exports"
                argv = ["--env", robot, "--reference", "random_walk", "--num-timesteps", "12",
                        "--train-seed", "1", "--rviz-pair", "corrected_strong", "weak",
                        "--output-dir", str(output)]
                for label, path in requests.items():
                    # Both accepted path forms must reconstruct the saved model correctly.
                    if label == "weak":
                        path = path / "model_final"
                    argv += ["--checkpoint", f"{label}={path}"]
                metrics = eval_vis.main(argv)
                manifest = json.loads((output / "manifest.json").read_text())
                self.assertEqual(manifest["exported_steps"], 12)
                self.assertEqual(manifest["rviz_pair"], ["corrected_strong", "weak"])
                self.assertFalse(manifest["automatic_resets"])
                baseline = np.load(output / "full_pos.npy")
                reference = np.load(output / "ref_pos.npy")
                expected_rmse = np.sqrt(np.mean(np.sum((baseline.astype(float) - reference) ** 2, axis=1)))
                for label, mode in mode_ids.items():
                    self.assertEqual(manifest["policies"][label]["mode"], mode)
                    np.testing.assert_array_equal(np.load(output / f"{label}_pos.npy"), baseline)
                    with np.load(output / f"{label}_rollout.npz") as archive:
                        np.testing.assert_array_equal(archive["ref_pos"], reference)
                        np.testing.assert_allclose(archive["time"], np.arange(1, 13) * .05, atol=1e-7)
                        # Seed 1 has negative biases: relu must produce zero, not hard_tanh(-.1).
                        np.testing.assert_array_equal(archive["actions"], np.zeros_like(archive["actions"]))
                        self.assertTrue(archive["bound_violation"].any())
                    quaternion = np.load(output / f"{label}_quat.npy")
                    self.assertEqual(quaternion.shape, (12, 4))
                    np.testing.assert_allclose(np.linalg.norm(quaternion, axis=1), 1.)
                for slot, label in (("baseline", "corrected_strong"), ("equivariant", "weak")):
                    for quantity in ("pos", "quat"):
                        np.testing.assert_array_equal(np.load(output / f"{slot}_{quantity}.npy"),
                                                      np.load(output / f"{label}_{quantity}.npy"))
                for item in metrics:
                    self.assertAlmostEqual(item["position_rmse_m"], expected_rmse)
                self.assertEqual(np.load(output / "ref_quat.npy").shape, (12, 4))
                self.assertEqual(len((output / "metrics.csv").read_text().splitlines()), 5)

    def test_astrobee_reference_is_independent_of_actual_angular_velocity_and_actions(self):
        with redirect_stdout(io.StringIO()):
            first = constant_policy("astrobee", 0, [0., 0., 0., .4, .1, 0.])
            second = constant_policy("astrobee", 2, [0., 0., 0., -.4, -.1, 0.])
        state, _ = first.env.reset(jax.random.PRNGKey(3))
        rng = jax.random.PRNGKey(4)
        tape = eval_vis.make_reference(first.env, state, rng, 8)
        changed = state.replace(omega=state.omega + jnp.array([2., -3., 1.]))
        other_tape = eval_vis.make_reference(first.env, changed, rng, 8)
        for key in tape:
            np.testing.assert_array_equal(tape[key], other_tape[key])
        a, b = (eval_vis.rollout(policy, state, tape, rng) for policy in (first, second))
        self.assertGreater(np.linalg.norm(a["omega"] - b["omega"]), .1)
        for key in tape:
            np.testing.assert_array_equal(a[key], b[key])

    def test_quadrotor_lissajous_uses_one_explicit_bank_index(self):
        with tempfile.TemporaryDirectory() as temp, redirect_stdout(io.StringIO()):
            previous = Path.cwd()
            try:
                os.chdir(temp)
                states = np.zeros((2, 6, 18), dtype=np.float32)
                states[:, :, 6:15] = np.eye(3).ravel()
                states[1, :, 0] = np.arange(6)
                np.save("ref_state_list.npy", states)
                np.save("ref_u_list.npy", np.full((2, 6, 4), .6573, dtype=np.float32))
                env = registry.get_env_by_name("quadrotor_random_lissajous", equivariant=0,
                                               terminate_on_error=False, max_time=float("inf"))
                initial = eval_vis.initial_state(env, "quadrotor", "lissajous", 7, 1, 5)
                tape = eval_vis.make_reference(env, initial, jax.random.PRNGKey(9), 5)
                np.testing.assert_array_equal(tape["ref_pos"][:, 0], np.arange(5))
                self.assertNotIn("ref_state_list", tape)
                self.assertNotIn("ref_u_list", tape)
                with self.assertRaisesRegex(ValueError, "exceeds"):
                    eval_vis.initial_state(env, "quadrotor", "lissajous", 7, 1, 7)
            finally:
                os.chdir(previous)

    def test_incorrect_checkpoint_label_is_rejected(self):
        with tempfile.TemporaryDirectory() as temp, redirect_stdout(io.StringIO()):
            full = self.save_fixture(Path(temp), "quadrotor", 0)
            with self.assertRaisesRegex(ValueError, "expected 2"):
                eval_vis.load_policies("quadrotor", "random_walk", {"weak": full}, 0)
        with self.assertRaisesRegex(ValueError, "outside"):
            eval_vis.select_seed_params({"log_std": np.zeros((2, 4))}, 2)

    def test_original_particle_exports_match_its_ros_reader(self):
        with tempfile.TemporaryDirectory() as temp, redirect_stdout(io.StringIO()):
            root = Path(temp)
            full = self.save_fixture(root, "particle", 0)
            strong = self.save_fixture(root, "particle", 3)
            output = root / "exports"
            eval_vis.main(["--env", "particle", "--checkpoint", f"full={full}",
                           "--checkpoint", f"strong={strong}", "--num-timesteps", "4",
                           "--output-dir", str(output)])
            np.testing.assert_array_equal(np.load(output / "anim_ref.npy"), np.load(output / "ref_pos.npy"))
            np.testing.assert_array_equal(np.load(output / "anim_baseline.npy"), np.load(output / "full_pos.npy"))
            np.testing.assert_array_equal(np.load(output / "anim_eq.npy"), np.load(output / "strong_pos.npy"))

    def test_numerical_failure_truncates_every_mode_to_the_same_prefix(self):
        data = {"pos": np.ones((5, 3)), "time": np.arange(5.)}
        bad = {key: value.copy() for key, value in data.items()}
        bad["pos"][3, 0] = np.nan
        result, failures = eval_vis.finite_rollouts({"full": data, "weak": bad})
        self.assertEqual(failures, {"full": None, "weak": 4})
        self.assertEqual(len(result["full"]["pos"]), 3)
        self.assertEqual(len(result["weak"]["pos"]), 3)


if __name__ == "__main__":
    unittest.main()
