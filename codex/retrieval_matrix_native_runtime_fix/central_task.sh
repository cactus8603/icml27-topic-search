#!/usr/bin/env bash
# [codex] Keep the central dispatcher's four positional arguments.
set -euo pipefail
if [[ $# -ne 4 || $3 != siglip2 ]]; then
  echo "Usage: bash central_task.sh FULL_REF PLAN siglip2 OUT" >&2
  exit 2
fi
here=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)
ref=$1
plan=$2
out=$4
bash "$here/run.sh" smoke --plan "$plan" --out "$out" --freeze-commit "$ref"
for benchmark in coco flickr; do
  bash "$here/run.sh" index --plan "$plan" --out "$out" --freeze-commit "$ref" --benchmark "$benchmark"
done
