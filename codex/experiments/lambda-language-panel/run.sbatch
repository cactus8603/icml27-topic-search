#!/usr/bin/env bash
#SBATCH --job-name=lambda-language-panel
#SBATCH --partition=h200q
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gres=none
#SBATCH --cpus-per-task=2
#SBATCH --mem=4G
#SBATCH --time=00:10:00
#SBATCH --output=/work/ChihChungHsu/cactus8603/icml/tmp/lambda-language-panel-%j.log
set -euo pipefail

root=/work/ChihChungHsu/cactus8603
repo="$root/icml27-topic-search"
shared="$root/icml"
stage=codex/experiments/lambda-language-panel
importer=codex/experiments/lambda-author-import
importer_freeze=bcbcc012d6f7f55b58e2bacafc9e1906c3741521
author_freeze=6964328411bf0ae49de47fa1bdfcfe8f94a75cce
questions=data/trait_questions/extract/spanish.json
python="$shared/envs/lambda-language-fasttext093/bin/python"
freeze=${1:?Use the full frozen language panel commit}
[[ "$freeze" =~ ^[0-9a-f]{40}$ ]]
[[ "$#" == 1 || "$#" == 2 && "$2" == --check-only ]]
changes=$(git -C "$repo" status --porcelain --untracked-files=all -- "$stage")
[[ -z "$changes" ]]
git -C "$repo" diff --quiet "$freeze" HEAD -- "$stage"
git -C "$repo" cat-file -e "$freeze:$stage/panel.py"

export TZ=Asia/Taipei TMPDIR="$shared/tmp" CUDA_VISIBLE_DEVICES=''
export HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 HF_HUB_DISABLE_IMPLICIT_TOKEN=1
export OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1
unset OPENAI_API_KEY OPENAI_BASE_URL HF_TOKEN HUGGING_FACE_HUB_TOKEN HUGGINGFACE_HUB_TOKEN
umask 077
snapshot=$(mktemp -d "$TMPDIR/lambda-language-panel-${SLURM_JOB_ID:-check}.XXXXXXXX")
trap 'rm -r --one-file-system -- "$snapshot"' EXIT
git -C "$repo" archive "$freeze" "$stage" | tar -x -C "$snapshot"
git -C "$repo" archive "$importer_freeze" "$importer/import_scores.py" | tar -x -C "$snapshot"
git -C "$shared/ext/shifting_the_gradient" archive "$author_freeze" "$questions" | tar -x -C "$snapshot"
args=(--check-only)
if [[ "${2:-}" != --check-only ]]; then
    job=${SLURM_JOB_ID:?Submit the reviewed experiment via Slurm}
    [[ "$job" =~ ^[1-9][0-9]*$ ]]
    args=(--output "$shared/runs/inoculation-lambda/language-panel-$job")
else
    "$python" -IBu "$snapshot/$stage/test_diagnostic.py"
fi
"$python" -IBu "$snapshot/$stage/panel.py" --questions "$snapshot/$questions" --freeze "$freeze" "${args[@]}"
