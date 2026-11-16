#!/usr/bin/env bash
# [codex] Explicit missing dependency overlay and private original paired producer.
set -euo pipefail
here=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)
repo=$(git -C "$here" rev-parse --show-toplevel)
export HF_HOME="$repo/hf_home" TMPDIR="$repo/hf_home/tmp" TZ=Asia/Taipei HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1
export PYTHONDONTWRITEBYTECODE=1 OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 TOKENIZERS_PARALLELISM=false
export PYTHONPATH="$repo/dl/retrieval-matrix-qwen2-runtime-fix/deps:$repo/dl/retrieval-matrix-extra/deps:$repo/dl/retrieval-matrix/deps/qwen2-transformers451:$repo/dl/retrieval-matrix/deps/qwen2-flash:$repo/dl/retrieval-matrix/deps/core-p0:$repo"
exec "$repo/dl/retrieval3090/envs/retrieval3090/bin/python" -B "$here/runner.py" "$@"
