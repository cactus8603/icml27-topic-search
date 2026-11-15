#!/usr/bin/env bash
#SBATCH --job-name=cr-anchor-rerank
#SBATCH --partition=h200q
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=48G
#SBATCH --time=00:30:00
#SBATCH --output=/work/ChihChungHsu/cactus8603/icml/tmp/cr-anchor-rerank-%j.log
set -euo pipefail

freeze=${1:?Pass the frozen commit}
[[ "$freeze" =~ ^[0-9a-f]{40}$ ]]
[[ "$#" == 1 || "$#" == 2 && "$2" == --check-only ]]
root=/work/ChihChungHsu/cactus8603
repo="$root/icml27-topic-search"
shared="$root/icml"
stage=codex/experiments/retrieval-anchor-rerank
previous=codex/experiments/retrieval-rerank
conjunction=codex/experiments/retrieval-conjunction
topic=killtests/compositional-retrieval
inputs=("$stage" "$previous/prepare.py" "$previous/panel.json" "$conjunction/prepare.py"
        "$conjunction/panel.json" "$topic/scripts" "$topic/configs/params_headroom.json")
changes=$(git -C "$repo" status --porcelain --untracked-files=all -- "${inputs[@]}")
[[ -z "$changes" ]]
git -C "$repo" diff --quiet "$freeze" HEAD -- "${inputs[@]}"
export TZ=Asia/Taipei TMPDIR="$shared/tmp" HF_HOME="$shared/hf_home" PYTHONUNBUFFERED=1
export HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1
bundle=$(mktemp -d "$TMPDIR/cr-anchor-rerank-${SLURM_JOB_ID:-check}.XXXXXXXX")
trap 'rm -r --one-file-system -- "$bundle"' EXIT
git -C "$repo" archive "$freeze" "${inputs[@]}" | tar -x -C "$bundle"
export EXP="$bundle/$topic"
python="$shared/envs/vllm033/bin/python"
if [[ "${2:-}" == --check-only ]]; then
    "$python" -IB "$bundle/$stage/test_evaluate.py"
    "$python" -IB "$bundle/$stage/evaluate.py" --preflight
else
    job=${SLURM_JOB_ID:?Submit through Slurm}
    [[ "$job" =~ ^[1-9][0-9]*$ ]]
    "$python" -IB "$bundle/$stage/evaluate.py" --score --freeze "$freeze"
fi
