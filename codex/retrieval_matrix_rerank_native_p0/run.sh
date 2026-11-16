#!/usr/bin/env bash
# [codex] Review/freeze/qualification/central gates are enforced in runner.
set -euo pipefail
here=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)
repo=$(git -C "$here" rev-parse --show-toplevel)
commit=${1:?full immutable source commit}
plan=${2:?reviewed frozen native MCQ plan}
variant=${3:?exact approved variant}
out=${4:?isolated output directory}
export HF_HOME="$repo/hf_home" HF_MODULES_CACHE="$repo/hf_home/modules" TMPDIR="$repo/hf_home/tmp" TZ=Asia/Taipei
export HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 PYTHONDONTWRITEBYTECODE=1
export IMAGE_MAX_TOKEN_NUM=1200 GENERATIVE_RERANKER_POSITIVE_TOKEN=yes GENERATIVE_RERANKER_NEGATIVE_TOKEN=no
export TOKENIZERS_PARALLELISM=false OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1
runtime="$repo/dl/retrieval-matrix-rerank/env/bin/python"
"$runtime" -B "$here/runner.py" --stage smoke --plan "$plan" --variant "$variant" --out "$out/smoke" --freeze-commit "$commit"
"$runtime" -B "$here/runner.py" --stage score --plan "$plan" --variant "$variant" --out "$out/scores" --smoke-out "$out/smoke" --freeze-commit "$commit"
