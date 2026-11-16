#!/usr/bin/env bash
# [codex] Explicit CPU stages; never submit GPU work or call a producer index stage.
set -euo pipefail
here=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)
repo=$(git -C "$here" rev-parse --show-toplevel)
export CUDA_VISIBLE_DEVICES='' HF_HOME="$repo/hf_home" HF_MODULES_CACHE="$repo/hf_home/modules"
export TMPDIR="$repo/hf_home/tmp" HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 PYTHONDONTWRITEBYTECODE=1
export PYTHONPATH="$repo" TZ=Asia/Taipei OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1
exec "$repo/dl/retrieval-matrix-rerank/env/bin/python" -B "$here/launcher.py" "$@"
