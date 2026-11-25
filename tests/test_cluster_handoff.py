"""Train all eight smoke jobs, resume one, transfer them, and evaluate the return."""

from contextlib import redirect_stdout
import io
import json
from pathlib import Path
import sys
import tarfile
import tempfile
import unittest
from unittest.mock import patch

import jax
import numpy as np

from rl import cluster_train
from rl.cluster_artifacts import package_runs, receive_archive, verify_run
from rl.cluster_jobs import JOBS


class ClusterHandoffTests(unittest.TestCase):
    def setUp(self):
        diagnostic = patch("orbax.checkpoint._src.handlers.base_pytree_checkpoint_handler._log_io_per_sec_metric")
        diagnostic.start()
        self.addCleanup(diagnostic.stop)

    def test_eight_smoke_jobs_resume_archive_receive_and_eval(self):
        with tempfile.TemporaryDirectory(prefix="cluster-handoff-") as temporary, redirect_stdout(io.StringIO()):
            root = Path(temporary)
            runs = root / "runs"
            common = ["--output-root", str(runs), "--smoke", "--allow-cpu", "--allow-dirty",
                      "--updates-per-chunk", "1"]
            # Simulate a preemption after the first committed PPO update.
            cluster_train.main(["train", "--task-index", "0", *common, "--stop-after-chunks", "1"])
            progress_path = runs / JOBS[0].name(0) / "progress.json"
            self.assertFalse(json.loads(progress_path.read_text())["completed"])
            self.assertEqual(json.loads(progress_path.read_text())["completed_updates"], 1)
            with self.assertRaisesRegex(ValueError, "incomplete"):
                package_runs(runs, root / "incomplete.tar.gz", allow_smoke=True)
            for job in JOBS:
                self.assertEqual(cluster_train.main(["train", "--task-index", str(job.index), *common]), 0)
                folder, success = verify_run(runs, job, 0, allow_smoke=True)
                self.assertEqual(success["actual_timesteps"], 64)
                self.assertEqual(success["observation_inputs"], job.inputs)
                with np.load(folder / "training_data.npz") as curves:
                    self.assertEqual(curves["return_mean"].shape, (1, 2))
                    np.testing.assert_array_equal(curves["x_vals_rew"], [32, 64])
                sys.stderr.write(f"  smoke job {job.index} saved and verified\n")
                sys.stderr.flush()
                jax.clear_caches()
            self.assertEqual(cluster_train.main(["status", "--output-root", str(runs), "--verify"]), 0)
            # The production packaging command must reject these tiny test policies.
            with self.assertRaisesRegex(ValueError, "Smoke-test"):
                package_runs(runs, root / "not-production.tar.gz")
            archive = root / "return.tar.gz"
            manifest = package_runs(runs, archive, allow_smoke=True)
            self.assertEqual(manifest["tasks"], list(range(8)))
            self.assertTrue(any("/model_final/d/" in name for name in manifest["files"]))
            received = root / "received"
            receive_archive(archive, received)
            for job in JOBS:
                verify_run(received, job, 0, allow_smoke=True)
            for robot in ("quadrotor", "astrobee"):
                output = root / f"eval_{robot}"
                self.assertEqual(cluster_train.main([
                    "export", "--output-root", str(received), "--env", robot,
                    "--num-timesteps", "4", "--output-dir", str(output),
                ]), 0)
                self.assertEqual(np.load(output / "baseline_pos.npy").shape, (4, 3))
                self.assertEqual(np.load(output / "equivariant_quat.npy").shape, (4, 4))
                self.assertEqual(len((output / "metrics.csv").read_text().splitlines()), 5)
                jax.clear_caches()
            # A truncated tensor chunk must never be mistaken for a valid checkpoint.
            first = JOBS[0]
            folder, success = verify_run(received, first, 0, allow_smoke=True)
            tensor_name = next(name for name in success["files"] if name.startswith("model_final/d/"))
            tensor = folder / tensor_name
            tensor.write_bytes(b"truncated")
            with self.assertRaisesRegex(ValueError, "Missing or changed"):
                verify_run(received, first, 0, allow_smoke=True)

    def test_receive_rejects_path_traversal_even_with_a_matching_archive_hash(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            archive = root / "unsafe.tar.gz"
            with tarfile.open(archive, "w:gz") as bundle:
                for name, payload in (("CAMPAIGN.json", b"{}"), ("../outside.txt", b"bad")):
                    member = tarfile.TarInfo(name)
                    member.size = len(payload)
                    bundle.addfile(member, io.BytesIO(payload))
            checksum = cluster_train.file_digest(archive)
            with self.assertRaisesRegex(ValueError, "Unsafe"):
                receive_archive(archive, root / "received", checksum)
            self.assertFalse((root / "outside.txt").exists())
            self.assertFalse((root / "received").exists())


if __name__ == "__main__":
    unittest.main()
