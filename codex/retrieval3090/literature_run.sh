#!/usr/bin/env bash
# [codex] Run under resources.py queue; all stages share the admitted GPU.
set -euo pipefail
here=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)
repo=$(git -C "$here" rev-parse --show-toplevel)
runtime=${RETRIEVAL_ROOT:-$repo/dl/retrieval3090}
python=${PYTHON:-$runtime/envs/retrieval3090-torch214/bin/python}
data="$runtime/data/literature-development/panels"
run=${1:?Supply an isolated run directory}
plan="$here/literature_plan.json"
export TZ=Asia/Taipei TMPDIR="$repo/hf_home/tmp" HF_HOME="$repo/hf_home"
export HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 PYTHONDONTWRITEBYTECODE=1 PYTHONUNBUFFERED=1
export OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1

"$python" -B "$here/literature_dev.py" --stage smoke --data "$data" --out "$run/smoke" --plan "$plan"
"$python" -B "$here/literature_dev.py" --stage index --data "$data" --out "$run/main" --plan "$plan"
"$python" -B "$here/literature_dev.py" --stage analyze --data "$data" --out "$run/main" --plan "$plan"
"$python" -B "$here/literature_dev.py" --stage cross50 --data "$data" --out "$run/main" --plan "$plan"
"$python" -B "$here/literature_dev.py" --stage analyze-cross --data "$data" --out "$run/main" --plan "$plan"
