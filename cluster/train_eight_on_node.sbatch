#!/usr/bin/env bash
#SBATCH --job-name=symmetry-eight-v2
#SBATCH --nodes=1
#SBATCH --ntasks=8
#SBATCH --gpus-per-task=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=256G
#SBATCH --time=2-00:00:00
#SBATCH --signal=USR1@120
#SBATCH --output=slurm_outs/paper-v2-node-%j.out
#SBATCH --error=slurm_outs/paper-v2-node-%j.err

set -euo pipefail
cd "${SLURM_SUBMIT_DIR:?Submit from the repository root}"
# Alternative to the array: one allocation must provide eight GPUs on one node.
# SLURM_PROCID 0..7 selects the policy; GPU binding isolates each process.
srun --label --kill-on-bad-exit=1 --gpu-bind=single:1 bash cluster/run_paper_v2_task.sh
