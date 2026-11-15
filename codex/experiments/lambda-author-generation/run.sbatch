#!/usr/bin/env bash
#SBATCH --job-name=lambda-author-generate
#SBATCH --partition=h200q
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --time=01:30:00
#SBATCH --dependency=afterok:26526:26527
#SBATCH --output=/work/ChihChungHsu/cactus8603/icml/tmp/lambda-author-generate-%j.log
set -euo pipefail

root=/work/ChihChungHsu/cactus8603
repo="$root/icml27-topic-search"
shared="$root/icml"
stage=codex/experiments/lambda-author-generation
author=6964328411bf0ae49de47fa1bdfcfe8f94a75cce
source="$shared/ext/shifting_the_gradient"
python="$shared/envs/inoc-phaseb-author-6964328/env/bin/python"
freeze=${1:?Pass the full freeze commit and optionally --check-only}
[[ "$freeze" =~ ^[0-9a-f]{40}$ ]]
[[ "$#" == 1 || "$#" == 2 && "$2" == --check-only ]]
[[ -z "$(git -C "$repo" status --porcelain --untracked-files=all -- "$stage")" ]]
git -C "$repo" diff --quiet "$freeze" HEAD -- "$stage"
git -C "$repo" cat-file -e "$freeze:$stage/generate.py"

export TZ=Asia/Taipei TMPDIR="$shared/tmp" PYTHONDONTWRITEBYTECODE=1
export HF_HOME="$shared/hf_home" HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 HF_HUB_DISABLE_IMPLICIT_TOKEN=1
export TOKENIZERS_PARALLELISM=false OMP_NUM_THREADS="${SLURM_CPUS_PER_TASK:-1}"
unset OPENAI_API_KEY OPENAI_BASE_URL HF_TOKEN HUGGING_FACE_HUB_TOKEN HUGGINGFACE_HUB_TOKEN
snapshot=$(mktemp -d "$TMPDIR/lambda-author-gen-${SLURM_JOB_ID:-check}.XXXXXXXX")
trap 'rm -r --one-file-system -- "$snapshot"' EXIT
git -C "$repo" archive "$freeze" "$stage" | tar -x -C "$snapshot"
mkdir "$snapshot/author"
git -C "$source" archive "$author" grad_shift scripts/eval_trait.py data/trait_questions/extract/spanish.json \
    | tar -x -C "$snapshot/author"
entry="$snapshot/$stage/generate.py"
args=(--source "$snapshot/author")
if [[ "${2:-}" == --check-only ]]; then
    export CUDA_VISIBLE_DEVICES=''
    "$python" -IBu "$entry" "${args[@]}"
    exit 0
fi
job=${SLURM_JOB_ID:?Submit through Slurm after both training jobs}
[[ "$job" =~ ^[0-9]+$ ]]
output="$shared/runs/inoculation-lambda/author-generation-$job"
[[ -f "$shared/runs/inoculation-lambda/author-training-26526/none/receipt.json" ]]
[[ -f "$shared/runs/inoculation-lambda/author-training-26527/ip/receipt.json" ]]
echo "[codex] $(date '+%Y-%m-%d %H:%M') Lambda Spanish generation source=$author freeze=$freeze"
for arm in base none ip; do
    "$python" -IBu "$entry" "${args[@]}" --arm "$arm" --output "$output" --commit "$freeze"
done
