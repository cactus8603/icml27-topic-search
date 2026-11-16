#!/usr/bin/env bash
# [codex] Independent queued Qwen2 execution. No changes to CORE's runtime.
set -euo pipefail
here=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)
repo=$(git -C "$here" rev-parse --show-toplevel)
ref=${1:?Supply full frozen commit}
plan=${2:?Supply frozen Qwen2 plan}
model=${3:?Supply gme2 or vlm2vec2}
out=${4:?Supply isolated output directory}
python="$repo/dl/retrieval3090/envs/retrieval3090/bin/python"
export TZ=Asia/Taipei TMPDIR="$repo/hf_home/tmp" HF_HOME="$repo/hf_home"
export PYTHONPATH="$repo/dl/retrieval-matrix/deps/qwen2-transformers451:$repo/dl/retrieval-matrix/deps/qwen2-flash:$repo/dl/retrieval-matrix/deps/core-p0"
export HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 PYTHONDONTWRITEBYTECODE=1 PYTHONUNBUFFERED=1
export TOKENIZERS_PARALLELISM=false OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1
"$python" -B "$repo/codex/retrieval_matrix/check_dependencies.py" --plan "$plan"
"$python" -B "$here/runner.py" --stage smoke --plan "$plan" --model "$model" --dataset coco \
  --out "$out/smoke" --freeze-commit "$ref"
for dataset in coco negbench flickr; do
  "$python" -B "$here/runner.py" --stage index --plan "$plan" --model "$model" --dataset "$dataset" \
    --out "$out/$dataset" --smoke-out "$out/smoke" --freeze-commit "$ref"
done
