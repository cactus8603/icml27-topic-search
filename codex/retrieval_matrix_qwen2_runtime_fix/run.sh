#!/usr/bin/env bash
# [codex] New isolated runtime; original frozen source and model settings intact.
set -euo pipefail
here=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)
repo=$(git -C "$here" rev-parse --show-toplevel)
mode=${1:?embedding or mcmr}
ref=${2:?new full frozen commit}
plan=${3:?new independently reviewed runtime-fix plan}
model=${4:?gme2 or vlm2vec2}
out=${5:?new isolated output directory}
python="$repo/dl/retrieval3090/envs/retrieval3090/bin/python"
export TZ=Asia/Taipei TMPDIR="$repo/hf_home/tmp" HF_HOME="$repo/hf_home"
export PYTHONPATH="$repo/dl/retrieval-matrix-qwen2-runtime-fix/deps:$repo/dl/retrieval-matrix/deps/qwen2-transformers451:$repo/dl/retrieval-matrix/deps/qwen2-flash:$repo/dl/retrieval-matrix/deps/core-p0:$repo"
export HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 PYTHONDONTWRITEBYTECODE=1 PYTHONUNBUFFERED=1
export TOKENIZERS_PARALLELISM=false OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1
"$python" -B "$here/check_dependencies.py" --plan "$plan"
if [[ "$mode" == embedding ]]; then
  "$python" -B "$here/runner.py" --mode "$mode" --stage smoke --plan "$plan" --model "$model" --dataset coco \
    --out "$out/smoke" --freeze-commit "$ref"
  for dataset in coco negbench flickr; do
    "$python" -B "$here/runner.py" --mode "$mode" --stage index --plan "$plan" --model "$model" --dataset "$dataset" \
      --out "$out/$dataset" --smoke-out "$out/smoke" --freeze-commit "$ref"
  done
elif [[ "$mode" == mcmr ]]; then
  "$python" -B "$here/runner.py" --mode "$mode" --stage smoke --plan "$plan" --model "$model" \
    --out "$out/smoke" --freeze-commit "$ref"
  "$python" -B "$here/runner.py" --mode "$mode" --stage index --plan "$plan" --model "$model" \
    --out "$out/mcmr" --smoke-out "$out/smoke" --freeze-commit "$ref"
else
  printf 'Unknown runtime repair mode: %s\n' "$mode" >&2
  exit 2
fi
