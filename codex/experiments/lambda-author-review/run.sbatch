#!/usr/bin/env bash
#SBATCH --job-name=lambda-review-packet
#SBATCH --partition=h200q
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gres=none
#SBATCH --cpus-per-task=2
#SBATCH --mem=8G
#SBATCH --time=00:15:00
#SBATCH --dependency=afterok:26566
#SBATCH --output=/work/ChihChungHsu/cactus8603/icml/tmp/lambda-review-packet-%j.log
set -euo pipefail

root=/work/ChihChungHsu/cactus8603
repo="$root/icml27-topic-search"
shared="$root/icml"
stage=codex/experiments/lambda-author-review
generator=codex/experiments/lambda-author-generation
generation_commit=639bf144b1df0b1cb29ded89b5f5cad68213a606
author=6964328411bf0ae49de47fa1bdfcfe8f94a75cce
freeze=${1:?Use the full frozen review commit}
[[ "$freeze" =~ ^[0-9a-f]{40}$ ]]
[[ "$#" == 1 || "$#" == 2 && "$2" == --check-only ]]
changes=$(git -C "$repo" status --porcelain --untracked-files=all -- "$stage")
[[ -z "$changes" ]]
git -C "$repo" diff --quiet "$freeze" HEAD -- "$stage"
git -C "$repo" cat-file -e "$freeze:$stage/bundle.py"

export TZ=Asia/Taipei TMPDIR="$shared/tmp" HF_HOME="$shared/hf_home" CUDA_VISIBLE_DEVICES=''
export HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 HF_HUB_DISABLE_IMPLICIT_TOKEN=1 TOKENIZERS_PARALLELISM=false
export OMP_NUM_THREADS=1
unset OPENAI_API_KEY OPENAI_BASE_URL HF_TOKEN HUGGING_FACE_HUB_TOKEN HUGGINGFACE_HUB_TOKEN
umask 077
snapshot=$(mktemp -d "$TMPDIR/lambda-review-${SLURM_JOB_ID:-check}.XXXXXXXX")
trap 'rm -r --one-file-system -- "$snapshot"' EXIT
git -C "$repo" archive "$freeze" "$stage" | tar -x -C "$snapshot"
git -C "$repo" archive "$generation_commit" "$generator/generate.py" | tar -x -C "$snapshot"
mkdir "$snapshot/author"
git -C "$shared/ext/shifting_the_gradient" archive "$author" grad_shift scripts/eval_trait.py \
    data/trait_questions/extract/spanish.json | tar -x -C "$snapshot/author"
args=(--check-only)
if [[ "${2:-}" != --check-only ]]; then
    job=${SLURM_JOB_ID:?Use the Slurm dependency on the three-arm generation}
    [[ "$job" =~ ^[1-9][0-9]*$ ]]
    args=(--output "$shared/runs/inoculation-lambda/author-review-$job")
fi
"$shared/envs/inoc-phaseb-author-6964328/env/bin/python" -IBu "$snapshot/$stage/bundle.py" \
    --source "$snapshot/author" --generation-freeze "$generation_commit" --freeze "$freeze" "${args[@]}"
