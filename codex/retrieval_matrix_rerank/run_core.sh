#!/usr/bin/env bash
# [codex] Exact released CORE MCQ scorer; schedule only after review/freeze.
set -euo pipefail
here=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)
repo=$(git -C "$here" rev-parse --show-toplevel)
ref=${1:?full frozen commit}
plan=${2:?frozen reviewed plan}
model=${3:?core2 or core8}
out=${4:?isolated output directory}
export HF_HOME="$repo/hf_home" TMPDIR="$repo/hf_home/tmp" TZ=Asia/Taipei
export HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 PYTHONDONTWRITEBYTECODE=1
export IMAGE_MAX_TOKEN_NUM=1200 GENERATIVE_RERANKER_POSITIVE_TOKEN=yes GENERATIVE_RERANKER_NEGATIVE_TOKEN=no
export TOKENIZERS_PARALLELISM=false OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1
python="$repo/dl/retrieval-matrix-rerank/env/bin/python"
"$python" -B "$here/runner.py" --stage smoke --plan "$plan" --model "$model" --out "$out/smoke" --freeze-commit "$ref"
"$python" -B "$here/runner.py" --stage score --plan "$plan" --model "$model" --out "$out/main" --smoke-out "$out/smoke" --freeze-commit "$ref"
