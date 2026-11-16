#!/usr/bin/env bash
# [codex] Native MCMR only; root queues after independent review and freeze.
set -euo pipefail
here=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)
repo=$(git -C "$here" rev-parse --show-toplevel)
ref=${1:?Supply the full frozen commit}
plan=${2:?Supply the frozen native MCMR plan}
model=${3:?Supply the model key}
out=${4:?Supply an isolated native run directory}
family=$(python3 -c 'import json,sys; print(json.load(open(sys.argv[1]))["runtime_family"])' "$plan")
if [[ "$family" == qwen3 ]]; then
  python="$repo/dl/retrieval3090/envs/retrieval3090-torch214/bin/python"
  export PYTHONPATH="$repo/dl/retrieval-matrix/deps/core-transformers457:$repo/dl/retrieval-matrix/deps/core-p0"
elif [[ "$family" == qwen2 ]]; then
  python="$repo/dl/retrieval3090/envs/retrieval3090/bin/python"
  export PYTHONPATH="$repo/dl/retrieval-matrix/deps/qwen2-transformers451:$repo/dl/retrieval-matrix/deps/qwen2-flash:$repo/dl/retrieval-matrix/deps/core-p0"
else
  printf 'Unsupported frozen runtime family: %s\n' "$family" >&2
  exit 2
fi
export TZ=Asia/Taipei TMPDIR="$repo/hf_home/tmp" HF_HOME="$repo/hf_home"
export HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 PYTHONDONTWRITEBYTECODE=1 PYTHONUNBUFFERED=1
export TOKENIZERS_PARALLELISM=false OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1
"$python" -B "$repo/codex/retrieval_matrix/check_dependencies.py" --plan "$plan"
"$python" -B "$here/runner.py" --stage smoke --plan "$plan" --model "$model" \
  --out "$out/smoke" --freeze-commit "$ref"
"$python" -B "$here/runner.py" --stage index --plan "$plan" --model "$model" \
  --out "$out/mcmr" --smoke-out "$out/smoke" --freeze-commit "$ref"
