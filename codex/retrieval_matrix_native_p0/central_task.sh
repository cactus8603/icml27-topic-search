#!/usr/bin/env bash
# [codex] Central dispatcher positional protocol; GPU producer only.
set -euo pipefail
if [[ $# -ne 4 || $3 != siglip2 ]]; then
  echo "Usage: bash central_task.sh FULL_REF PLAN siglip2 OUT" >&2
  exit 2
fi
TASK_ROOT=$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)
TASK_REF=$1
TASK_PLAN=$2
TASK_OUT=$4
bash "$TASK_ROOT/codex/retrieval_matrix_native_p0/run.sh" smoke --plan "$TASK_PLAN" --out "$TASK_OUT" --freeze-commit "$TASK_REF"
for TASK_BENCH in coco flickr; do
  bash "$TASK_ROOT/codex/retrieval_matrix_native_p0/run.sh" index --plan "$TASK_PLAN" --out "$TASK_OUT" --freeze-commit "$TASK_REF" --benchmark "$TASK_BENCH"
done
