#!/usr/bin/env bash
# [codex] Local development only; original frozen files are never edited.
set -euo pipefail
here=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)
repo=$(git -C "$here" rev-parse --show-toplevel)
runtime=${RETRIEVAL_ROOT:-$repo/dl/retrieval3090}
python=${PYTHON:-$runtime/envs/retrieval3090/bin/python}
entry=${1:?Use test, preflight, smoke, cache, cache2, analyze, analyze2, or tests}
shift
SOURCE=2daedf522c6d64b860277657d3067b61da3a0f2b
FREEZE=569b6c053f0f06006b7eab01ab64b815cd136702
export TZ=Asia/Taipei TMPDIR="$repo/hf_home/tmp" HF_HOME="$repo/hf_home"
export HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 PYTHONUNBUFFERED=1
export OMP_NUM_THREADS=${OMP_NUM_THREADS:-4}
export PYTHONDONTWRITEBYTECODE=1
mkdir -p "$TMPDIR"
umask 077
bundle=$(mktemp -d "$TMPDIR/retrieval3090.XXXXXXXX")
trap 'rm -r --one-file-system -- "$bundle"' EXIT
topic=killtests/compositional-retrieval
stages=codex/experiments
git -C "$repo" archive "$SOURCE" "$topic/scripts" "$topic/scripts_shim" "$topic/configs/params_headroom.json" \
  "$stages"/retrieval-anchor-rerank/{prepare.py,evaluate.py,panel.json} \
  "$stages"/retrieval-rerank/{prepare.py,panel.json} \
  "$stages"/retrieval-conjunction/{prepare.py,panel.json} | tar -x -C "$bundle"
git -C "$repo" archive "$FREEZE" "$stages/retrieval-recall-budget" | tar -x -C "$bundle"
mkdir "$bundle/dev"
# Research runs use committed files even while other agents edit the checkout.
# RETRIEVAL_CODE_REF=working is reserved for pre-commit technical tests.
code_ref=${RETRIEVAL_CODE_REF:-HEAD}
if [[ "$code_ref" == working ]]; then
  cp "$repo"/topics/compositional-retrieval/code/dev-3090/*.py "$bundle/dev/"
  cp "$here"/*.py "$bundle/dev/"
  export DEV_COMMIT=$(git -C "$repo" rev-parse HEAD) DEV_DIRTY=1
else
  git -C "$repo" archive "$code_ref" topics/compositional-retrieval/code/dev-3090 codex/retrieval3090 | tar -x -C "$bundle"
  cp "$bundle"/topics/compositional-retrieval/code/dev-3090/*.py "$bundle/dev/"
  cp "$bundle"/codex/retrieval3090/*.py "$bundle/dev/"
  export DEV_COMMIT=$(git -C "$repo" rev-parse "$code_ref^{commit}") DEV_DIRTY=0
fi
# Relocate path literals in this disposable copy, preserving all model/scoring parameters.
"$python" - "$bundle" "$runtime" <<'PY'
from pathlib import Path
import sys
root, runtime = Path(sys.argv[1]), sys.argv[2]
for p in root.rglob('*.py'):
    if p.is_symlink():
        continue
    body = p.read_text()
    updated = body.replace('/work/ChihChungHsu/cactus8603/icml', runtime)
    if updated != body:
        p.write_text(updated)
PY
export EXP="$bundle/$topic" RETRIEVAL_ROOT="$runtime"
export DEV_BUNDLE="$bundle" DEV_SOURCE="$SOURCE" DEV_FREEZE="$FREEZE"
"$python" -IB "$bundle/dev/local_entry.py" "$entry" "$@"
