#!/usr/bin/env bash
set -euo pipefail
cd "${SLURM_SUBMIT_DIR:?Submit from the repository root}"
SYMMETRY_VENV="${SYMMETRY_VENV:-$PWD/.venv-paper-v2}"
SYMMETRY_SEED="${SYMMETRY_SEED:-0}"
SYMMETRY_RUN_KIND="${SYMMETRY_RUN_KIND:-full}"
SYMMETRY_OUTPUT_ROOT="${SYMMETRY_OUTPUT_ROOT:-$PWD/cluster_runs/paper_v2_${SYMMETRY_RUN_KIND}_seed${SYMMETRY_SEED}}"
task="${SLURM_ARRAY_TASK_ID:-${SLURM_PROCID:?Missing array task or process index}}"
if [[ ! -x "$SYMMETRY_VENV/bin/python" ]]; then
    echo "Missing environment: $SYMMETRY_VENV. Run bash cluster/setup_paper_v2.sh" >&2
    exit 2
fi
if [[ "$SYMMETRY_RUN_KIND" != full && "$SYMMETRY_RUN_KIND" != smoke ]]; then
    echo "SYMMETRY_RUN_KIND must be full or smoke" >&2
    exit 2
fi
export MPLBACKEND=Agg PYTHONUNBUFFERED=1 JAX_PLATFORMS=cuda
export OMP_NUM_THREADS="${SLURM_CPUS_PER_TASK:-4}"
export JAX_COMPILATION_CACHE_DIR="${JAX_COMPILATION_CACHE_DIR:-$PWD/cluster_runs/jax_cache_paper_v2}"
mkdir -p "$JAX_COMPILATION_CACHE_DIR"
extra=()
if [[ "$SYMMETRY_RUN_KIND" == smoke ]]; then extra+=(--smoke); fi
if [[ "${SYMMETRY_STOP_AFTER_CHUNKS:-0}" != 0 ]]; then
    extra+=(--stop-after-chunks "$SYMMETRY_STOP_AFTER_CHUNKS")
fi
echo "Task $task on $(hostname); visible GPU ${CUDA_VISIBLE_DEVICES:-unset}; output $SYMMETRY_OUTPUT_ROOT"
exec "$SYMMETRY_VENV/bin/python" -m rl.cluster_train train \
    --task-index "$task" --output-root "$SYMMETRY_OUTPUT_ROOT" --seed "$SYMMETRY_SEED" \
    --updates-per-chunk "${SYMMETRY_UPDATES_PER_CHUNK:-128}" "${extra[@]}"
