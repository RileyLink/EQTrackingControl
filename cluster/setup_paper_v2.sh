#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "${BASH_SOURCE[0]}")/.."
SYMMETRY_VENV="${SYMMETRY_VENV:-$PWD/.venv-paper-v2}"
SYMMETRY_PYTHON="${SYMMETRY_PYTHON:-python3.11}"
if [[ ! -x "$SYMMETRY_VENV/bin/python" ]]; then
    if command -v uv >/dev/null 2>&1; then
        uv venv --python "$SYMMETRY_PYTHON" --seed "$SYMMETRY_VENV"
    else
        "$SYMMETRY_PYTHON" -m venv "$SYMMETRY_VENV"
    fi
fi
"$SYMMETRY_VENV/bin/python" -c 'import sys; assert sys.version_info[:2] in ((3,11),(3,12)), "Use Python 3.11 or 3.12"'
"$SYMMETRY_VENV/bin/python" -m pip install --no-cache-dir -r requirements-paper-v2-b200.txt
"$SYMMETRY_VENV/bin/python" -m pip install --no-cache-dir --no-deps -e .
"$SYMMETRY_VENV/bin/python" -m pip check
mkdir -p slurm_outs cluster_artifacts
"$SYMMETRY_VENV/bin/python" -m rl.cluster_train plan
echo "Setup complete. Run the GPU smoke array inside a B200 allocation next."
