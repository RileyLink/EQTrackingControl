#!/usr/bin/env bash
#SBATCH --job-name=symmetry-paper-v2
#SBATCH --array=0-7%8
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gpus-per-task=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=32G
#SBATCH --time=2-00:00:00
#SBATCH --signal=USR1@120
#SBATCH --output=slurm_outs/paper-v2-%A_%a.out
#SBATCH --error=slurm_outs/paper-v2-%A_%a.err

set -euo pipefail
cd "${SLURM_SUBMIT_DIR:?Submit from the repository root}"
# Each independently scheduled policy receives its own GPU. Supply your site's
# B200 partition/account as sbatch arguments; do not clear CUDA_VISIBLE_DEVICES.
srun --kill-on-bad-exit=1 --gpu-bind=single:1 bash cluster/run_paper_v2_task.sh
