#!/usr/bin/env bash
# [codex] Draft only: not scheduled; final sources/protocol require root freeze.
set -euo pipefail
if [[ $# -ne 4 || ! $3 =~ ^clic_(b32|l14|clips_l14)$ ]]; then
  echo "Usage: bash central_task.sh FULL_REF PLAN MODEL OUT" >&2
  exit 2
fi
TASK_ROOT=$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)
TASK_REF=$1 TASK_PLAN=$2 TASK_MODEL=$3 TASK_OUT=$4
"$TASK_ROOT/dl/retrieval3090/envs/retrieval3090/bin/python" -B -c 'import json,sys; assert json.load(open(sys.argv[1]))["model_key"] == sys.argv[2]' "$TASK_PLAN" "$TASK_MODEL"
bash "$TASK_ROOT/codex/retrieval_matrix_paired_p0/run.sh" smoke --plan "$TASK_PLAN" --out "$TASK_OUT" --freeze-commit "$TASK_REF"
for TASK_BENCH in sugarcrepe sugarcrepe_pp; do
  bash "$TASK_ROOT/codex/retrieval_matrix_paired_p0/run.sh" index --plan "$TASK_PLAN" --out "$TASK_OUT" --benchmark "$TASK_BENCH" --freeze-commit "$TASK_REF"
done
