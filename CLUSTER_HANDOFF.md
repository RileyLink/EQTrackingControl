# Run the eight symmetry policies and return the trained checkpoints

This guide is for the friend with cluster access. Riley does not need a cluster account to use the returned policies. Use a Linux NVIDIA GPU node and Python 3.11. ROS is only needed later for playback, not for training or returning the models.

## What to train

Each row is a separate policy trained from scratch. All use six hidden layers of 128 units, one training seed (`0`), `leaky_relu` hidden activations, and supported `hard_tanh` outputs. No Lee controller, transferred policy, derivative inputs, or derivative warm-up is used.

| Array task | Robot | Label | Observation mode | Inputs | Requested training steps |
| --- | --- | --- | --- | --- | --- |
| 0 | Quadrotor | `full` | 0 | 40 | 1,000,000,000 |
| 1 | Quadrotor | `strong` (original) | 1 | 31 | 1,000,000,000 |
| 2 | Quadrotor | `corrected_strong` | 3 | 31 | 1,000,000,000 |
| 3 | Quadrotor | `weak` | 2 | 25 | 1,000,000,000 |
| 4 | Astrobee | `full` | 0 | 42 | 400,000,000 |
| 5 | Astrobee | `strong` (original) | 1 | 30 | 400,000,000 |
| 6 | Astrobee | `corrected_strong` | 6 | 30 | 400,000,000 |
| 7 | Astrobee | `weak` | 2 | 27 | 400,000,000 |

Training uses the current random-walk environments and standard trainer PPO/reward defaults, with the larger network and budgets above. Quadrotor uses 64 parallel environments; Astrobee uses 32. Both use rollout length 512, five PPO epochs, four minibatches, learning rate 0.0003, and the existing learning-rate schedule. All Astrobee jobs use body-force `symmetry_type=2`. The four jobs for a given robot differ only in observation mode and experiment name. Exact settings are in [rl/cluster_jobs.py](rl/cluster_jobs.py) and each saved `config.txt`.

Budgets round down to complete PPO batches. The actual transition count is recorded in `SUCCESS.json`; these are training budgets, not guarantees of convergence. Keep the same seed/configurations for all eight jobs. Multiple independent seeds would require additional campaigns.

## 1. Clone and install

Use the symmetry branch, which is separate from the thrust-derivative experiments:

```bash
git clone --branch codex/quadrotor-weak-symmetry https://github.com/RileyLink/EQTrackingControl.git
cd EQTrackingControl
git rev-parse HEAD

# Load your site's Python 3.11 module first if necessary.
python3.11 -m venv .venv-cluster
source .venv-cluster/bin/activate
python -m pip install --upgrade pip
python -m pip install -r requirements-cluster-gpu.txt
python -m pip install -e .
python -m pip check
python -m rl.cluster_train plan
mkdir -p slurm_outs
```

The GPU requirements retain JAX/JAXlib 0.4.30 and add its matching CUDA 12 plugin. They use pip-provided CUDA libraries; the node still needs a compatible NVIDIA driver. Check with `nvidia-smi` inside the allocation. This is an older pinned research stack: on GPU hardware requiring a newer CUDA/JAX stack, stop at the smoke test and report the incompatibility rather than silently upgrading packages. GPU installation details: [JAX documentation](https://docs.jax.dev/en/latest/installation.html), [JAX 0.4.30 package metadata](https://pypi.org/pypi/jax/0.4.30/json).

Keep this checkout unchanged while the campaign runs. Each run records its source commit, package versions, device, configuration, and encoder version. Production training refuses source edits, and a resume requires the original source commit. Generated virtual-environment/egg-info installation files are excluded from this source check. If the repository is private, request GitHub read access from Riley or use a permitted fork; cluster credentials do not need to be shared.

## 2. Check GPU access and run the eight short smoke jobs

The commands below assume SLURM. Add your site's `--partition=...` and `--account=...` options to `srun`/`sbatch` if required. They are deliberately not hard-coded in the repository. Run training on allocated compute nodes.

First check the environment in a GPU allocation:

```bash
srun --gpus=1 --cpus-per-task=4 --mem=8G --time=00:10:00 \
  .venv-cluster/bin/python -m rl.cluster_train check
```

The check must show a GPU. The runner also enforces GPU availability before starting a production task. Then submit the smoke array from the repository root:

```bash
export SYMMETRY_RUN_KIND=smoke
export SYMMETRY_OUTPUT_ROOT="$PWD/cluster_runs/smoke_seed0"
sbatch --time=00:30:00 cluster/train_symmetry_array.sbatch
```

The supplied array runs tasks 0–7, at most four concurrently, with one GPU and four CPU cores per job. Smoke jobs retain the six-layer/128-unit network but use only 64 transitions and small PPO batches. They test observation shapes, GPU operations, saving, and the eight-job workflow. Their weights are not useful trained controllers.

Monitor and verify after all smoke jobs finish:

```bash
squeue -u "$USER"
python -m rl.cluster_train status --output-root "$SYMMETRY_OUTPUT_ROOT" --verify
```

Verification must report `8/8 complete`. Logs are in `slurm_outs/symmetry-<array_job_id>_<task_id>.out` and `.err`. If a job fails, send Riley the relevant logs and environment/device information. Do not launch the long campaign until the smoke campaign passes.

## 3. Submit the full training campaign

Choose a shared, persistent output directory visible from every compute node. The repository-local default works if the checkout is on shared storage; a site-provided project/scratch directory may be preferable. Do not use node-local temporary storage for the only copy of checkpoints.

```bash
export SYMMETRY_RUN_KIND=full
export SYMMETRY_OUTPUT_ROOT="$PWD/cluster_runs/full_seed0"
sbatch cluster/train_symmetry_array.sbatch
```

This starts the eight full-budget policies. The default job request is 1 GPU, 4 CPU cores, 32 GB host memory, and 48 hours per task. Adjust limits to your site's partition. The resource request is a limit, not a predicted runtime. To limit concurrency to two GPUs, use:

```bash
sbatch --array=0-7%2 cluster/train_symmetry_array.sbatch
```

Use `%8` if eight simultaneous GPU allocations are available. Each policy uses one GPU; the concurrency comes from independent array jobs. Optional environment variables are `SYMMETRY_VENV` (default `.venv-cluster`), `SYMMETRY_SEED` (default `0`), and `SYMMETRY_UPDATES_PER_CHUNK` (default `128`).

For a cluster without SLURM, obtain one GPU through its scheduler and run the equivalent commands sequentially inside that allocation:

```bash
for task in 0 1 2 3 4 5 6 7; do
  python -m rl.cluster_train train --task-index "$task" \
    --output-root "$SYMMETRY_OUTPUT_ROOT" --updates-per-chunk 128
done
```

Neither training task needs the external quadrotor Lissajous `.npy` banks. The random-walk references are generated internally.

## 4. Monitor and resume

```bash
python -m rl.cluster_train status --output-root "$SYMMETRY_OUTPUT_ROOT"
```

The runner logs progress after each chunk. It saves the full runner state: policy/critic weights, Adam state, simulator state, counters, and random keys. It retains two recent intermediate checkpoints plus the final checkpoint. Metrics retain five scalars per PPO update rather than every simulator-step log, avoiding the previous approximately 17 GB logging cost at one billion transitions. The PPO loss, dynamics, reward, and observation implementations are unchanged by this logging option; the original `rl.train_policy` command keeps its original logging behavior.

If a job is interrupted, resubmit the same task with the same output root and source checkout. It resumes automatically. For example, resume task 3:

```bash
sbatch --array=3 cluster/train_symmetry_array.sbatch
```

Resubmitting the entire array also works: completed tasks exit successfully and incomplete tasks resume. SLURM's advance signal requests a checkpointed stop at the next chunk boundary; a forced kill may lose the current unsaved chunk. The script does not assume your site's automatic requeue policy. For tighter checkpoint intervals set `SYMMETRY_UPDATES_PER_CHUNK=32` before submission. [SLURM submission/signal documentation](https://slurm.schedmd.com/sbatch.html).

## 5. Package what Riley needs

After all eight full-budget jobs finish:

```bash
python -m rl.cluster_train status --output-root "$SYMMETRY_OUTPUT_ROOT" --verify
mkdir -p cluster_artifacts
python -m rl.cluster_train package --output-root "$SYMMETRY_OUTPUT_ROOT" \
  --archive cluster_artifacts/symmetry-policies-seed0.tar.gz
```

The package command refuses incomplete runs, smoke-test policies, changed/missing files, mismatched configurations, or mixed source commits. It includes all eight final checkpoints and validates checksums. If a filename already exists, choose a new archive filename.

**Send Riley both files:**

```text
symmetry-policies-seed0.tar.gz
symmetry-policies-seed0.tar.gz.sha256
```

Transfer them through a shared institutional folder, Google Drive/Dropbox, an accessible download link, or a GitHub Release attachment. Copy the files to a location Riley can access; a path on a cluster Riley cannot access is not sufficient. The archive includes:

- `model_final/` for every policy, including every Orbax tensor-data file and hidden metadata file;
- `config.txt`, needed to reconstruct each observation mode and neural network;
- source commit, encoder version, package/device metadata, completion markers, and file checksums;
- compact training curves (`training_data.npz` and `training_curve.png`).

Do not send just `_METADATA`, `_sharding`, or a screenshot of training curves. The weight data, often under `model_final/d/`, are essential. The package command collects them automatically. Intermediate checkpoints and the virtual environment are not needed for Riley's evaluation. Keep the originals on the cluster until Riley verifies import.

### Optional pull request

A PR is useful for the run report and any code fixes. It does not replace transferring the trained weights. The repository ignores checkpoint/data directories.

Fork the repository if necessary, create a report branch from this symmetry branch, and add a small report under `reports/<campaign-name>/`. Use [the report template](cluster/REPORT_TEMPLATE.md) to record the source commit, GPU, eight completion statuses, archive download link, and archive SHA-256. Submit the PR against `RileyLink/EQTrackingControl`, base branch `codex/quadrotor-weak-symmetry`. Keep the weight archive in the separate download location. Riley must be able to download it.

## 6. Riley: receive and evaluate the returned models

Place the archive and its `.sha256` file together. In the same source revision and project Python environment:

```bash
python -m rl.cluster_train receive --archive /path/to/symmetry-policies-seed0.tar.gz \
  --destination checkpoints/friend_symmetry_seed0
```

On Windows, use the downloaded archive's quoted Windows path. `receive` verifies the outer SHA-256, every member checksum, all eight configurations, and the complete checkpoint data before installing anything. Choose a destination that does not already exist. No cluster account, GPU, or ROS installation is required to import the archive. The CUDA-trained weights can be restored on the laptop's CPU.

Export each robot's four policies on common references:

```bash
python -m rl.cluster_train export --output-root checkpoints/friend_symmetry_seed0 --env quadrotor
python -m rl.cluster_train export --output-root checkpoints/friend_symmetry_seed0 --env astrobee
```

These commands use the previously implemented four-mode `eval_vis` exporter, internally generated random-walk references, and corrected-strong/weak RViz slots. Add `--eval-seed 1` to test another initialization/reference, `--num-timesteps 2000` to specify duration, or `--rviz-pair full weak` to select another pair. Every mode is saved separately. See [the exporter guide](rl/README.md#ros-rollout-export-eval_vispy) and [ROS playback guide](ros2_ws/src/trajectory_visualization/README.md).

The handoff workflow is validated locally with real CPU smoke training, interruption/resume, archive transfer, and evaluation. Actual NVIDIA/SLURM execution must be verified by the friend's GPU smoke campaign before committing cluster time to the full runs.
