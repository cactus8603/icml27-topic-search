#!/usr/bin/env bash
#SBATCH --job-name=rsd-fork-budget
#SBATCH --partition=h200q
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=48G
#SBATCH --time=01:30:00
#SBATCH --output=/work/ChihChungHsu/cactus8603/icml/tmp/rsd-fork-budget-%j.log
set -euo pipefail

freeze=${1:?Pass the full reviewed commit}
mode=${2:?Use --check-only, smoke or diagnostic}
[[ "$freeze" =~ ^[0-9a-f]{40}$ ]]
case "$mode" in
    --check-only|smoke) [[ "$#" == 2 ]] ;;
    diagnostic) [[ "$#" == 3 && "$3" =~ ^[1-9][0-9]*$ ]] ;;
    *) exit 2 ;;
esac

shared=/work/ChihChungHsu/cactus8603/icml
repo=/work/ChihChungHsu/cactus8603/icml27-topic-search
stage=codex/experiments/rl-fork-budget
fork_stage=codex/experiments/rl-single-token-fork
topic=killtests/rl-sparse-distill
python="$shared/envs/oat-vllm072/bin/python"
[[ -x "$python" ]]
[[ -z "$(git -C "$repo" status --porcelain --untracked-files=all -- "$stage")" ]]
git -C "$repo" diff --quiet "$freeze" HEAD -- "$stage" "$fork_stage"
git -C "$repo" cat-file -e "$freeze:$stage/diagnostic.py"

export TZ=Asia/Taipei TMPDIR="$shared/tmp" HF_HOME="$shared/hf_home"
export HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 HF_HUB_DISABLE_IMPLICIT_TOKEN=1
export OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 TOKENIZERS_PARALLELISM=false
export PYTHONDONTWRITEBYTECODE=1 PYTHONUNBUFFERED=1 DO_NOT_TRACK=1
export CUBLAS_WORKSPACE_CONFIG=:4096:8
unset OPENAI_API_KEY OPENAI_BASE_URL HF_TOKEN HUGGING_FACE_HUB_TOKEN HUGGINGFACE_HUB_TOKEN
umask 077
bundle=$(mktemp -d "$TMPDIR/rsd-fork-budget-${SLURM_JOB_ID:-check}.XXXXXXXX")
trap 'rm -r --one-file-system -- "$bundle"' EXIT
export XDG_CACHE_HOME="$bundle/xdg"
git -C "$repo" archive "$freeze" "$stage" "$fork_stage/experiment.py" "$topic/scripts" \
    "$topic/configs/pair_b.json" "$topic/configs/pairs.json" "$topic/data/eval.jsonl" \
    codex/experiments/rl-a-short-stop/pilot.py codex/experiments/rl-template/probe.py | tar -x -C "$bundle"
entry="$bundle/$stage/diagnostic.py"
grader="$repo/$topic/pylib"

if [[ "$mode" == --check-only ]]; then
    CUDA_VISIBLE_DEVICES='' "$python" -IBu "$bundle/$stage/test_diagnostic.py"
    CUDA_VISIBLE_DEVICES='' "$python" -IBu "$entry" preflight --grader "$grader" --freeze "$freeze"
    exit 0
fi

job=${SLURM_JOB_ID:?Use Slurm for GPU generation}
[[ "$job" =~ ^[1-9][0-9]*$ ]]
outdir="$shared/runs/rsd-fork-budget"
if [[ "$mode" == diagnostic ]]; then
    smoke="$outdir/smoke-$3.json"
    "$python" -IB - "$smoke" "$freeze" "$3" <<'PY'
import json
import sys

receipt_path, freeze, job = sys.argv[1:]
with open(receipt_path, encoding="utf-8") as source:
    receipt = json.load(source)
if (receipt["mode"], receipt["freeze"], receipt["job"], receipt["synthetic_controls_passed"]) != (
        "smoke", freeze, job, True):
    raise ValueError("Expected the passing same-commit synthetic GPU receipt")
PY
    destination="$outdir/$job.json"
else
    destination="$outdir/smoke-$job.json"
fi
[[ ! -e "$destination" && ! -L "$destination" ]]
echo "[codex] $(date '+%F %R') one-token fork budget $mode freeze=$freeze"
"$python" -IBu "$entry" "$mode" --grader "$grader" --freeze "$freeze" --job "$job" --output "$bundle/receipt.json"
mkdir -p "$outdir"
ln "$bundle/receipt.json" "$destination"
sha256sum -- "$destination"
