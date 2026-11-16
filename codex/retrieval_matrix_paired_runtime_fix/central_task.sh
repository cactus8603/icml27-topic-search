#!/usr/bin/env bash
# [codex] Same central four-argument API, isolated new paired outputs.
set -euo pipefail
if [[ $# -ne 4 ]]; then echo "Usage: central_task.sh FULL_REF PLAN MODEL OUT" >&2; exit 2; fi
case $3 in clic_b32|clic_l14|clic_clips_l14) ;; *) exit 2 ;; esac
here=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)
ref=$1
plan=$2
model=$3
out=$4
python3 - "$plan" "$model" <<'PY'
import json,sys
if json.load(open(sys.argv[1]))['model_key']!=sys.argv[2]:raise ValueError('Central model differs from the exact paired plan')
PY
bash "$here/run.sh" smoke --plan "$plan" --out "$out" --freeze-commit "$ref"
for benchmark in sugarcrepe sugarcrepe_pp; do
  bash "$here/run.sh" index --plan "$plan" --out "$out" --benchmark "$benchmark" --freeze-commit "$ref"
done
