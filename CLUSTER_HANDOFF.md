# Corrected eight-policy campaign v2, including NVIDIA B200

Use branch `codex/paper-rewards-b200`. V2 restores the **last upstream SLURM script's requested settings**, wires the attitude coefficient into the reward, and defines quadrotor clipped ReLU. It is a corrected implementation of those presets, not a verified exact paper reproduction. No Lee controller, derivative inputs, transferred policy, or warm-up is used.

Local checks and their limits are recorded in [cluster/VALIDATION.md](cluster/VALIDATION.md).

Policies start from scratch. Within each robot, only observation mode and experiment name differ. Resume continues the same policy/optimizer/RNG/environment state; it does not transfer between modes. Keep v1 directories/checkpoints separate.

## Exact jobs and settings

| Task | Robot | Mode | Inputs | Hidden layers | Units/layer | Requested steps |
| --- | --- | --- | --- | --- | --- | --- |
| 0 | Quadrotor | Full 0 | 40 | 6 | 128 | 1 billion |
| 1 | Quadrotor | Original strong 1 | 31 | 6 | 128 | 1 billion |
| 2 | Quadrotor | Corrected strong 3 | 31 | 6 | 128 | 1 billion |
| 3 | Quadrotor | Weak 2 | 25 | 6 | 128 | 1 billion |
| 4 | Astrobee | Full 0 | 42 | 6 | 256 | 400 million |
| 5 | Astrobee | Original strong 1 | 30 | 6 | 256 | 400 million |
| 6 | Astrobee | Corrected strong 6 | 30 | 6 | 256 | 400 million |
| 7 | Astrobee | Weak 2 | 27 | 6 | 256 | 400 million |

| Parameter | Quadrotor | Astrobee |
| --- | --- | --- |
| Position weight | 0.02 | 0.05 |
| Velocity weight | 0.001 | 0.001 |
| Effective attitude weight | 0.6 | 10 |
| Angular-rate weight | 0.0001 | 0.05 |
| Action-error weight | 0.0005 | 0.001 |
| Reach bonus | Enabled | Enabled |
| Error penalty | -25 | -25 |
| Termination bound | 10 | 25 |
| Actor output | clipped ReLU, min(max(z,0),1) | hard_tanh, signed commands |
| Parallel environments / rollout steps | 64 / 512 | 32 / 512 |

Hidden activation is leaky ReLU. PPO and discrete dynamics retain existing behavior. The original reward/done timing is retained: penalty is evaluated on the crossing state; done/reset follows a step later. The attitude argument is now `reward_q_rot`; misspelled `reward_q_rotm` raises an error. Saved config retains the key `REWARD_Q_ROTM`.

The archived Astrobee config differs from the SLURM script (position weight 0.01 and 128 units). V2 follows the script's 0.05 and 256. Clipped ReLU is defined explicitly because the released script referenced an unavailable Flax symbol; the historical implementation could not be verified. Astrobee keeps the script's hard_tanh because its force/moment commands are signed.

## Why the B200 failed before

The returned smoke log failed before PPO training while compiling a PRNG operation:

```text
ptxas fatal: Program with .target 'sm_90a' cannot be compiled to future architecture
```

V1 pinned JAX/jaxlib/CUDA plugin 0.4.30, predating Blackwell support. V2 pins JAX 0.6.2 and matching plugins plus CUDA 12.8 libraries/compiler. Old logs also reported driver CUDA 12.8 versus ptxas 12.9.86, disabling parallel compilation. Pinning CUDA 12.8 avoids that particular compiler drift.

The B200 host needs a compatible driver (R570+ for native CUDA 12.8). Python packages cannot upgrade the host driver. Use the pip-provided CUDA stack in a fresh environment without old CUDA modules overriding its libraries. Accounts/partitions/GPU quotas are site-specific.

References: [NVIDIA Blackwell compatibility](https://docs.nvidia.com/cuda/archive/12.8.0/blackwell-compatibility-guide/index.html), [CUDA 12.8 release notes](https://docs.nvidia.com/cuda/archive/12.8.0/cuda-toolkit-release-notes/index.html), [JAX installation](https://docs.jax.dev/en/latest/installation.html).

## Pull and install

```bash
git clone --branch codex/paper-rewards-b200 https://github.com/RileyLink/EQTrackingControl.git EQTrackingControl-paper-v2
cd EQTrackingControl-paper-v2
bash cluster/setup_paper_v2.sh
```

For the offline update ZIP, use the existing v1 clone (it must contain commit `de2b98970ee59326da42b0d5d340e242be250aa6`). Extract the ZIP elsewhere, then from that old clone run:

```bash
git fetch /path/to/paper-v2-update.bundle codex/paper-rewards-b200:codex/paper-rewards-b200
git worktree add ../EQTrackingControl-paper-v2 codex/paper-rewards-b200
cd ../EQTrackingControl-paper-v2
bash cluster/setup_paper_v2.sh
```

The small incremental Git bundle contains new source commits and requires the original v1 commit; it is not a standalone clone. A fresh clone can use GitHub instead. No user's trained weights or virtual environment are included.

Python 3.11 is default. For 3.12, use `SYMMETRY_PYTHON=python3.12 bash cluster/setup_paper_v2.sh`. The helper uses `uv venv --seed` when available, otherwise `python -m venv` (requires the OS venv module). Keep the old environment unchanged. Set `SYMMETRY_VENV=/absolute/path` consistently to use a custom path.

Use scratch storage with roughly 10 GB free for the environment and installation temporaries; the CUDA libraries are large. Setup disables pip's download cache to avoid retaining a second copy. Check any per-user filesystem quota as well as free space.

For CPU evaluation, install `requirements-paper-v2-cpu.txt`. Legacy `requirements-cluster-gpu.txt` / `requirements-weak-symmetry.txt` are retained for old artifacts and are not the v2 B200 recipe.

## Preflight and smoke array

Use `sinfo -o '%P %G'` to identify available partitions/GPU types. Replace `YOUR_B200_PARTITION` below and supply your site's account if required. If the site needs a typed GPU request, pass its advertised type with `--gpus-per-task=TYPE:1`.

Before training, the runner compiles and executes PRNG, matmul, matrix exponential, and inverse kernels. Each policy requires exactly one visible GPU. GPU kind, driver/package versions, and effective reward checks are saved in `run_metadata.json`. It verifies +1 reward at perfect tracking and a -25 failure-penalty difference. Production refuses CPU fallback. Merely listing a GPU is insufficient.

```bash
mkdir -p slurm_outs
sbatch --partition=YOUR_B200_PARTITION --export=ALL,SYMMETRY_RUN_KIND=smoke cluster/train_symmetry_array.sbatch
```

After completion:

```bash
.venv-paper-v2/bin/python -m rl.cluster_train status --output-root cluster_runs/paper_v2_smoke_seed0 --verify
```

All eight must complete. Smoke validates execution/serialization/transfer, not learned tracking. For detailed preflight inside a one-GPU allocation, run `.venv-paper-v2/bin/python -m rl.cluster_train check`.

## Eight policies in parallel

```bash
sbatch --partition=YOUR_B200_PARTITION --export=ALL,SYMMETRY_RUN_KIND=full cluster/train_symmetry_array.sbatch
```

The array is `0-7%8`, one GPU per task. Eight simultaneous policies need eight GPUs and sufficient quota. Each policy remains single-GPU PPO. Concurrency reduces campaign elapsed time, not one policy's training time; queue availability can still delay jobs.

Alternatively request one node with eight GPU-bound workers:

```bash
sbatch --partition=YOUR_B200_PARTITION --export=ALL,SYMMETRY_RUN_KIND=full cluster/train_eight_on_node.sbatch
```

Do not submit both alternatives for the same output root. Preserve `CUDA_VISIBLE_DEVICES`; it isolates workers. Each process verifies that JAX sees exactly one GPU.

Jobs use bounded logging, resumable 128-update chunks, and a persistent compilation cache. Environment batch sizes/PPO are not inflated for B200 utilization. Actual B200 speed needs measurement; small simulators/networks can be limited by kernel overhead rather than peak matrix throughput.

For an optional early inspection, use `--array=0,1,4,5` and `--export=ALL,SYMMETRY_RUN_KIND=full,SYMMETRY_STOP_AFTER_CHUNKS=8`. This stops at exact full-run prefixes. Resume without the stop setting to continue unchanged configurations.

## Resume, verify, and return

Resubmit the same command/output root after interruption. Never use v1 run roots. Config fingerprints/source-commit checks reject mixing campaigns.

```bash
.venv-paper-v2/bin/python -m rl.cluster_train status --output-root cluster_runs/paper_v2_full_seed0 --verify
mkdir -p cluster_artifacts
.venv-paper-v2/bin/python -m rl.cluster_train package \
  --output-root cluster_runs/paper_v2_full_seed0 \
  --archive cluster_artifacts/paper-v2-seed0.tar.gz
```

Return the `.tar.gz`, `.sha256` sidecar, and smoke/full `slurm_outs`. Retain originals until import is verified. The archive includes all eight final tensor checkpoints, configs, curves, source/runtime/effective-reward metadata, and checksums. A PR containing only plots does not transfer weights.

Riley imports/evaluates using the modern CPU profile:

```bash
python -m rl.cluster_train receive --archive /path/to/paper-v2-seed0.tar.gz --destination checkpoints/paper_v2_seed0
python -m rl.cluster_train export --output-root checkpoints/paper_v2_seed0 --env quadrotor --num-timesteps 300
python -m rl.cluster_train export --output-root checkpoints/paper_v2_seed0 --env astrobee --num-timesteps 300
```

Exports include trajectories, actual actions, errors, and ROS poses for all modes. Assess learned quality from rollouts as well as returns. No B200 hardware run is claimed until the GPU preflight/smoke jobs pass on the friend's cluster.
