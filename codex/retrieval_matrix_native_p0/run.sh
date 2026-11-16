#!/usr/bin/env bash
# [codex] Root owns all GPU dispatch. audit/analyze are separate CPU stages.
set -euo pipefail
TASK_ROOT=$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)
export PYTHONDONTWRITEBYTECODE=1
export HF_HOME="$TASK_ROOT/hf_home"
export HF_HUB_OFFLINE=1
export TRANSFORMERS_OFFLINE=1
export TMPDIR="$TASK_ROOT/hf_home/tmp"
export PYTHONPATH="$TASK_ROOT/dl/retrieval-matrix-extra/deps:$TASK_ROOT/dl/retrieval-matrix/deps/qwen2-transformers451:$TASK_ROOT/dl/retrieval-matrix/deps/qwen2-flash:$TASK_ROOT/dl/retrieval-matrix/deps/core-p0:$TASK_ROOT"
exec "$TASK_ROOT/dl/retrieval3090/envs/retrieval3090/bin/python" -B "$TASK_ROOT/codex/retrieval_matrix_native_p0/runner.py" "$@"
