#!/usr/bin/env bash
# [codex] CPU audit/analyze stay separate; root alone dispatches GPU stages.
set -euo pipefail
TASK_ROOT=$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)
export PYTHONDONTWRITEBYTECODE=1 HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1
export HF_HOME="$TASK_ROOT/hf_home" TMPDIR="$TASK_ROOT/hf_home/tmp"
export OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1
export PYTHONPATH="$TASK_ROOT/dl/retrieval-matrix-extra/deps:$TASK_ROOT/dl/retrieval-matrix/deps/qwen2-transformers451:$TASK_ROOT/dl/retrieval-matrix/deps/qwen2-flash:$TASK_ROOT/dl/retrieval-matrix/deps/core-p0:$TASK_ROOT"
exec "$TASK_ROOT/dl/retrieval3090/envs/retrieval3090/bin/python" -B "$TASK_ROOT/codex/retrieval_matrix_paired_p0/runner.py" "$@"
