#!/usr/bin/env bash
# Run a dev-3090 entry point inside a bundle of the frozen retrieval code.
#
#   bash bundle.sh cache_scores.py --preflight
#   bash bundle.sh cache_scores.py --run --out "$ICML/runs/retrieval-dev/<tag>"
#   bash bundle.sh analyze.py --cache "$ICML/runs/retrieval-dev/<tag>"
#   bash bundle.sh test_analyze.py
#
# The frozen dependencies come from Git, exactly as the 26806 job used them:
# shared scripts and the prior stages from SOURCE, the recall-budget stage from FREEZE.
# The dev files are copied from this working tree, so uncommitted edits run too;
# meta.json records whether they were dirty.
set -euo pipefail

SOURCE=2daedf522c6d64b860277657d3067b61da3a0f2b
FREEZE=569b6c053f0f06006b7eab01ab64b815cd136702
entry=${1:?Pass an entry point, e.g. cache_scores.py}
shift

here=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)
repo=$(git -C "$here" rev-parse --show-toplevel)
# Frozen modules hard-code this root (common.py ICML). On a 3090 host, mount or symlink the same path.
ICML=${ICML:-/work/ChihChungHsu/cactus8603/icml}
[[ "$ICML" == /work/ChihChungHsu/cactus8603/icml ]] || {
    echo "Frozen modules read /work/ChihChungHsu/cactus8603/icml; symlink that path instead of overriding ICML" >&2
    exit 2
}
python=${PYTHON:-$ICML/envs/vllm033/bin/python}

export TZ=Asia/Taipei TMPDIR="$ICML/tmp" HF_HOME="$ICML/hf_home" PYTHONUNBUFFERED=1
export HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 OMP_NUM_THREADS=${OMP_NUM_THREADS:-4}
umask 077
bundle=$(mktemp -d "$TMPDIR/cr-dev3090.XXXXXXXX")
trap 'rm -r --one-file-system -- "$bundle"' EXIT

topic=killtests/compositional-retrieval
stages=codex/experiments
git -C "$repo" archive "$SOURCE" "$topic/scripts" "$topic/scripts_shim" "$topic/configs/params_headroom.json" \
    "$stages"/retrieval-anchor-rerank/{prepare.py,evaluate.py,panel.json} \
    "$stages"/retrieval-rerank/{prepare.py,panel.json} \
    "$stages"/retrieval-conjunction/{prepare.py,panel.json} | tar -x -C "$bundle"
git -C "$repo" archive "$FREEZE" "$stages/retrieval-recall-budget" | tar -x -C "$bundle"
mkdir "$bundle/dev"
cp "$here"/*.py "$bundle/dev/"

export EXP="$bundle/$topic"
export DEV_BUNDLE="$bundle" DEV_SOURCE="$SOURCE" DEV_FREEZE="$FREEZE"
export DEV_COMMIT=$(git -C "$repo" rev-parse HEAD)
export DEV_DIRTY=$([[ -z "$(git -C "$repo" status --porcelain -- "$here")" ]] && echo 0 || echo 1)
"$python" -IB "$bundle/dev/$entry" "$@"
