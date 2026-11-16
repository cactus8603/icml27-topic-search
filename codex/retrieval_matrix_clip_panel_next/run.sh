#!/usr/bin/env bash
# [codex] Preserve each qualified parent runtime; only root dispatches GPU work.
set -euo pipefail
if [[ $# -ne 4 ]]; then echo "Usage: run.sh FULL_REF PLAN MODEL OUT" >&2; exit 2; fi
here=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)
repo=$(git -C "$here" rev-parse --show-toplevel)
ref=${1:?full frozen source commit}
plan=${2:?real qualified frozen P2 plan}
model=${3:?qualified exact model}
out=${4:?new isolated output}
export HF_HOME="$repo/hf_home" TMPDIR="$repo/hf_home/tmp" TZ=Asia/Taipei HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1
export PYTHONDONTWRITEBYTECODE=1 TOKENIZERS_PARALLELISM=false OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1
# Both constant-allowlisted repaired parents have these exact dependency roots.
export PYTHONPATH="$repo/dl/retrieval-matrix-qwen2-runtime-fix/deps:$repo/dl/retrieval-matrix-extra/deps:$repo/dl/retrieval-matrix/deps/qwen2-transformers451:$repo/dl/retrieval-matrix/deps/qwen2-flash:$repo/dl/retrieval-matrix/deps/core-p0:$repo"
runtime="$repo/dl/retrieval3090/envs/retrieval3090/bin/python"
"$runtime" -B "$here/runner.py" --stage smoke --plan "$plan" --model "$model" --dataset conjunction-dev --out "$out/smoke" --freeze-commit "$ref"
for dataset in conjunction-dev negation-dev cola-gqa cola-paco; do
  "$runtime" -B "$here/runner.py" --stage index --plan "$plan" --model "$model" --dataset "$dataset" --out "$out/$dataset" --smoke-out "$out/smoke" --freeze-commit "$ref"
done
