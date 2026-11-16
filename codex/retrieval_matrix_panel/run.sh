#!/usr/bin/env bash
# [codex] P2 must retain the exact qualified P0 environment/model specification.
set -euo pipefail
here=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)
repo=$(git -C "$here" rev-parse --show-toplevel)
ref=${1:?full frozen commit}
plan=${2:?qualified frozen P2 plan}
model=${3:?qualified model key}
family=${4:?core or qwen2}
out=${5:?isolated output directory}
if [ "$family" = core ]; then
  python="$repo/dl/retrieval3090/envs/retrieval3090-torch214/bin/python"
  export PYTHONPATH="$repo/dl/retrieval-matrix/deps/core-transformers457:$repo/dl/retrieval-matrix/deps/core-p0"
elif [ "$family" = qwen2 ]; then
  python="$repo/dl/retrieval3090/envs/retrieval3090/bin/python"
  export PYTHONPATH="$repo/dl/retrieval-matrix/deps/qwen2-transformers451:$repo/dl/retrieval-matrix/deps/qwen2-flash:$repo/dl/retrieval-matrix/deps/core-p0"
else
  exit 2
fi
export TZ=Asia/Taipei TMPDIR="$repo/hf_home/tmp" HF_HOME="$repo/hf_home"
export HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 PYTHONDONTWRITEBYTECODE=1 PYTHONUNBUFFERED=1
export TOKENIZERS_PARALLELISM=false OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1
"$python" -B "$repo/codex/retrieval_matrix/check_dependencies.py" --plan "$plan"
"$python" -B "$here/runner.py" --stage smoke --plan "$plan" --model "$model" --dataset conjunction-dev \
  --out "$out/smoke" --freeze-commit "$ref"
for dataset in conjunction-dev negation-dev cola-gqa cola-paco; do
  "$python" -B "$here/runner.py" --stage index --plan "$plan" --model "$model" --dataset "$dataset" \
    --out "$out/$dataset" --smoke-out "$out/smoke" --freeze-commit "$ref"
done
