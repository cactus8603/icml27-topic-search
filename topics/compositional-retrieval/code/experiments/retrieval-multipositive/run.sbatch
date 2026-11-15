#!/usr/bin/env bash
#SBATCH --job-name=cr-multipositive
#SBATCH --partition=h200q
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=48G
#SBATCH --time=00:25:00
#SBATCH --output=/work/ChihChungHsu/cactus8603/icml/tmp/cr-multipositive-%j.log
set -euo pipefail

freeze=${1:?Pass the frozen commit}
[[ "$freeze" =~ ^[0-9a-f]{40}$ ]]
[[ "$#" == 1 || "$#" == 2 && "$2" == --check-only ]]
root=/work/ChihChungHsu/cactus8603
repo="$root/icml27-topic-search"
shared="$root/icml"
stage=codex/experiments/retrieval-multipositive
topic=killtests/compositional-retrieval
source=2daedf522c6d64b860277657d3067b61da3a0f2b
data="$shared/data/coco2017-retrieval-multipositive"
[[ -z "$(git -C "$repo" status --porcelain --untracked-files=all -- "$stage")" ]]
git -C "$repo" diff --quiet "$freeze" HEAD -- "$stage"

export TZ=Asia/Taipei TMPDIR="$shared/tmp" HF_HOME="$shared/hf_home" PYTHONUNBUFFERED=1
export HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1
umask 077
bundle=$(mktemp -d "$TMPDIR/cr-multipositive-${SLURM_JOB_ID:-check}.XXXXXXXX")
trap 'rm -r --one-file-system -- "$bundle"' EXIT
git -C "$repo" archive "$source" "$topic/scripts" "$topic/configs" \
    codex/experiments/retrieval-anchor-rerank/{prepare,evaluate}.py \
    codex/experiments/retrieval-rerank/{prepare.py,panel.json} \
    codex/experiments/retrieval-conjunction/{prepare.py,panel.json} | tar -x -C "$bundle"
git -C "$repo" archive "$freeze" "$stage" | tar -x -C "$bundle"

export EXP="$bundle/$topic"
python="$shared/envs/vllm033/bin/python"
entry="$bundle/$stage"
receipt_sha=$("$python" -IB - "$entry/assets.json" <<'PY'
import json
from pathlib import Path
import re
import sys

data = json.loads(Path(sys.argv[1]).read_bytes())
if set(data) != {"receipt_sha256"} or not re.fullmatch(r"[a-f0-9]{64}", data["receipt_sha256"]):
    raise ValueError("A fixed JPEG receipt fingerprint is required")
print(data["receipt_sha256"])
PY
)
test "$(sha256sum "$data/image-receipt.json" | cut -d ' ' -f1)" = "$receipt_sha"
args=(--plan "$data/panel.json" --assets "$data" --receipt-sha256 "$receipt_sha")
if [[ "${2:-}" == --check-only ]]; then
    export CUDA_VISIBLE_DEVICES=''
    "$python" -IB "$entry/test_panel.py"
    "$python" -IB "$entry/test_images.py"
    "$python" -IB "$entry/test_ranking.py"
    "$python" -IB "$entry/panel.py" verify --manifest "$data/panel.json"
    args+=(--check-only)
else
    job=${SLURM_JOB_ID:?Submit through Slurm}
    [[ "$job" =~ ^[1-9][0-9]*$ ]]
    args+=(--output "$shared/runs/retrieval-multipositive/$job" --freeze "$freeze")
fi
CUDA_VISIBLE_DEVICES='' "$python" -IB "$entry/images.py" verify --plan "$data/panel.json" --directory "$data"
"$python" -IB "$entry/ranking.py" "${args[@]}"
