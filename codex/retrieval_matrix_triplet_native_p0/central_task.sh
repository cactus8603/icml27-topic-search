#!/usr/bin/env bash
# [codex] Four-argument review interface; source gaps intentionally refuse GPU.
set -euo pipefail
if [[ $# -ne 4 ]]; then echo 'Usage: central_task.sh FULL_REF PLAN MODEL OUT' >&2; exit 2; fi
here=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)
repo=$(git -C "$here" rev-parse --show-toplevel)
export PYTHONDONTWRITEBYTECODE=1
python3 -B "$here/runner.py" --freeze-commit "$1" --plan "$2" --model "$3" --out "$4"
