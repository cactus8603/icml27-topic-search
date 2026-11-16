#!/usr/bin/env bash
# [codex] Run only through the shared-GPU queue after a committed study freeze.
set -euo pipefail
here=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)
repo=$(git -C "$here" rev-parse --show-toplevel)
ref=${1:?Supply the full frozen commit}
out=${2:?Supply an isolated run directory}
python=${PYTHON:-$repo/dl/retrieval3090/envs/retrieval3090-torch214/bin/python}
export TZ=Asia/Taipei TMPDIR="$repo/hf_home/tmp" HF_HOME="$repo/hf_home"
export HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 PYTHONDONTWRITEBYTECODE=1 PYTHONUNBUFFERED=1
export OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1
"$python" -B "$here/runner.py" --stage preflight --plan "$here/plan.json" --out "$out"
"$python" -B "$here/runner.py" --stage smoke --plan "$here/plan.json" --out "$out" --freeze-commit "$ref"
"$python" -B "$here/runner.py" --stage all --plan "$here/plan.json" --out "$out" --freeze-commit "$ref"
