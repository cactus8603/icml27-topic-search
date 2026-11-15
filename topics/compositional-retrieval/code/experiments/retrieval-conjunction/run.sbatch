#!/usr/bin/env bash
#SBATCH --job-name=cr-core-conjunction
#SBATCH --partition=h200q
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=48G
#SBATCH --time=00:50:00
#SBATCH --output=/work/ChihChungHsu/cactus8603/icml/tmp/cr-core-conjunction-%j.log
set -euo pipefail

freeze=${1:?Pass the frozen commit}
[[ "$freeze" =~ ^[0-9a-f]{40}$ ]]
[[ "$#" == 1 || "$#" == 2 && "$2" == --check-only ]]
root=/work/ChihChungHsu/cactus8603
repo="$root/icml27-topic-search"
shared="$root/icml"
stage=codex/experiments/retrieval-conjunction
topic=killtests/compositional-retrieval
changes=$(git -C "$repo" status --porcelain --untracked-files=all -- "$stage" "$topic/scripts" "$topic/configs/params_headroom.json")
[[ -z "$changes" ]]
git -C "$repo" diff --quiet "$freeze" HEAD -- "$stage" "$topic/scripts" "$topic/configs/params_headroom.json"
export TZ=Asia/Taipei TMPDIR="$shared/tmp" HF_HOME="$shared/hf_home" PYTHONUNBUFFERED=1
export HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1
bundle=$(mktemp -d "$TMPDIR/cr-core-conjunction-${SLURM_JOB_ID:-check}.XXXXXXXX")
trap 'rm -r --one-file-system -- "$bundle"' EXIT
git -C "$repo" archive "$freeze" "$stage" "$topic/scripts" "$topic/configs/params_headroom.json" | tar -x -C "$bundle"
export EXP="$bundle/$topic"
args=(--preflight)
if [[ "${2:-}" != --check-only ]]; then
    job=${SLURM_JOB_ID:?Submit through Slurm}
    args=(--freeze "$freeze" --output "$shared/runs/retrieval-conjunction/$job")
fi
"$shared/envs/vllm033/bin/python" -IB "$bundle/$stage/evaluate.py" "${args[@]}"
