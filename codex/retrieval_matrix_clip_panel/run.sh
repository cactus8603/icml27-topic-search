#!/usr/bin/env bash
# [codex] Qualified SigLIP2 only, preserving its native P0 environment.
set -euo pipefail
here=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)
repo=$(git -C "$here" rev-parse --show-toplevel)
ref=${1:?full frozen commit}
plan=${2:?qualified frozen CLIP P2 plan}
model=${3:?qualified model key}
out=${4:?isolated output directory}
python="$repo/dl/retrieval3090/envs/retrieval3090/bin/python"
export PYTHONPATH="$repo/dl/retrieval-matrix-extra/deps:$repo/dl/retrieval-matrix/deps/qwen2-transformers451:$repo/dl/retrieval-matrix/deps/qwen2-flash:$repo/dl/retrieval-matrix/deps/core-p0"
export TZ=Asia/Taipei TMPDIR="$repo/hf_home/tmp" HF_HOME="$repo/hf_home"
export HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 PYTHONDONTWRITEBYTECODE=1 PYTHONUNBUFFERED=1
export TOKENIZERS_PARALLELISM=false OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1
"$python" -B "$here/runner.py" --stage smoke --plan "$plan" --model "$model" --dataset conjunction-dev \
  --out "$out/smoke" --freeze-commit "$ref"
for dataset in conjunction-dev negation-dev cola-gqa cola-paco; do
  "$python" -B "$here/runner.py" --stage index --plan "$plan" --model "$model" --dataset "$dataset" \
    --out "$out/$dataset" --smoke-out "$out/smoke" --freeze-commit "$ref"
done
