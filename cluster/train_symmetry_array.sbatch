#!/usr/bin/env bash
#SBATCH --job-name=symmetry-eight
#SBATCH --array=0-7%4
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gpus-per-task=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=32G
#SBATCH --time=2-00:00:00
#SBATCH --signal=USR1@120
#SBATCH --output=slurm_outs/symmetry-%A_%a.out
#SBATCH --error=slurm_outs/symmetry-%A_%a.err

set -euo pipefail

# Submit from the repository root. Account/partition/GPU model are site-specific
# and can be supplied as sbatch flags without editing this file.
cd "${SLURM_SUBMIT_DIR:?Submit with sbatch from the repository root}"
SYMMETRY_VENV="${SYMMETRY_VENV:-$PWD/.venv-cluster}"
SYMMETRY_SEED="${SYMMETRY_SEED:-0}"
SYMMETRY_RUN_KIND="${SYMMETRY_RUN_KIND:-full}"
SYMMETRY_OUTPUT_ROOT="${SYMMETRY_OUTPUT_ROOT:-$PWD/cluster_runs/${SYMMETRY_RUN_KIND}_seed${SYMMETRY_SEED}}"

if [[ ! -x "$SYMMETRY_VENV/bin/python" ]]; then
    echo "Missing cluster virtual environment: $SYMMETRY_VENV" >&2
    exit 2
fi
if [[ "$SYMMETRY_RUN_KIND" != full && "$SYMMETRY_RUN_KIND" != smoke ]]; then
    echo "SYMMETRY_RUN_KIND must be full or smoke" >&2
    exit 2
fi

export MPLBACKEND=Agg
export PYTHONUNBUFFERED=1
export OMP_NUM_THREADS="${SLURM_CPUS_PER_TASK:-4}"
extra=()
if [[ "$SYMMETRY_RUN_KIND" == smoke ]]; then
    extra+=(--smoke)
fi

# SLURM signals the srun job step; Python checkpoints at the next chunk boundary.
# A resubmission of the same task/output root resumes automatically.
srun --kill-on-bad-exit=1 "$SYMMETRY_VENV/bin/python" -m rl.cluster_train train \
    --task-index "${SLURM_ARRAY_TASK_ID:?Missing array task ID}" \
    --output-root "$SYMMETRY_OUTPUT_ROOT" --seed "$SYMMETRY_SEED" \
    --updates-per-chunk "${SYMMETRY_UPDATES_PER_CHUNK:-128}" "${extra[@]}"
