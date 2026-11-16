#!/usr/bin/env bash
# [codex] Preserve the central dispatcher's fixed four-positional-argument API.
set -euo pipefail
here=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)
ref=${1:?new full frozen commit}
plan=${2:?new independently reviewed runtime-fix plan}
model=${3:?gme2 or vlm2vec2}
out=${4:?new isolated output directory}
mode=$(python3 - "$plan" <<'PY'
import json,sys
plan=json.load(open(sys.argv[1]))
mapping={'P0-Qwen2':'embedding','MCMR-native-MMEB-v2':'mcmr'}
mode=mapping[plan['phase']]
if plan['runtime_fix']['mode']!=mode:
    raise ValueError('Phase differs from this independently frozen runtime mode')
print(mode)
PY
)
exec bash "$here/run.sh" "$mode" "$ref" "$plan" "$model" "$out"
