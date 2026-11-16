#!/usr/bin/env bash
# [codex] Metadata-only driver; worker selects the exact frozen CPU environment.
set -euo pipefail
here=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)
repo=$(git -C "$here" rev-parse --show-toplevel)
export CUDA_VISIBLE_DEVICES='' PYTHONDONTWRITEBYTECODE=1
export OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1
export PYTHONPATH="$repo" TMPDIR="$repo/hf_home/tmp"
exec "$repo/dl/retrieval3090/envs/retrieval3090/bin/python" -B "$here/launcher.py" "$@"
