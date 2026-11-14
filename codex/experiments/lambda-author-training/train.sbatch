#!/usr/bin/env bash
#SBATCH --job-name=lambda-author-train
#SBATCH --partition=h200q
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --time=02:00:00
#SBATCH --array=0
#SBATCH --output=/work/ChihChungHsu/cactus8603/icml/tmp/lambda-author-train-%A_%a.log
set -euo pipefail

root=/work/ChihChungHsu/cactus8603
repo="$root/icml27-topic-search"
shared="$root/icml"
stage=codex/experiments/lambda-author-training
topic=killtests/inoculation-lambda
official=6964328411bf0ae49de47fa1bdfcfe8f94a75cce
helper=1280b1ff17b18684358fa9a56227d5df5c44e43b
checkout="$shared/ext/shifting_the_gradient"
base="$shared/hf_home/hub/models--Qwen--Qwen2.5-7B-Instruct/snapshots/a09a35458c702b33eeacc393d103063234e8bc28"
author_python="$shared/envs/inoc-phaseb-author-6964328/env/bin/python"
freeze=${1:?Pass the full freeze commit, optionally followed by --check-only or the none job ID}
[[ "$freeze" =~ ^[0-9a-f]{40}$ ]]
[[ "$#" -le 2 ]]
[[ "$(git -C "$repo" rev-parse --verify "$freeze^{commit}")" == "$freeze" ]]
[[ -z "$(git -C "$repo" status --porcelain --untracked-files=all -- "$stage")" ]]
git -C "$repo" diff --quiet "$freeze" HEAD -- "$stage"
git -C "$repo" cat-file -e "$freeze:$stage/worker.py"

export TZ=Asia/Taipei TMPDIR="$shared/tmp" PYTHONDONTWRITEBYTECODE=1
export HF_HOME="$shared/hf_home" HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 HF_HUB_DISABLE_IMPLICIT_TOKEN=1
export TOKENIZERS_PARALLELISM=false OMP_NUM_THREADS="${SLURM_CPUS_PER_TASK:-1}"
unset OPENAI_API_KEY OPENAI_BASE_URL HF_TOKEN HUGGING_FACE_HUB_TOKEN HUGGINGFACE_HUB_TOKEN
args=()
if [[ "${2:-}" == --check-only ]]; then
    export CUDA_VISIBLE_DEVICES=''
else
    coord=${SLURM_ARRAY_TASK_ID:?Submit one arm as Slurm array task 0 or 1}
    array=${SLURM_ARRAY_JOB_ID:?Submit one arm as Slurm array task 0 or 1}
    job=${SLURM_JOB_ID:?Submit one arm as Slurm array task 0 or 1}
    [[ "$coord" =~ ^[01]$ && "$array" =~ ^[0-9]+$ && "$job" =~ ^[0-9]+$ ]]
    args+=(--shard "$coord" --array-job "$array" --job "$job"
          --output "$shared/runs/inoculation-lambda/author-training-$array")
    if [[ "$coord" == 1 ]]; then
        none=${2:?Pass the verified predecessor array ID and use an afterok dependency}
        [[ "$none" =~ ^[0-9]+$ && "$none" != "$array" ]]
        prior="$shared/runs/inoculation-lambda/author-training-$none/none/receipt.json"
        [[ -f "$prior" && ! -L "$prior" ]]
        args+=(--prior-none "$prior")
    else
        [[ "$#" == 1 ]]
    fi
fi
snapshot=$(mktemp -d "$TMPDIR/lambda-author-train-${SLURM_JOB_ID:-check}.XXXXXXXX")
trap 'rm -r --one-file-system -- "$snapshot"' EXIT
git -C "$repo" archive "$freeze" "$stage" | tar -x -C "$snapshot"
git -C "$repo" archive "$helper" "$topic/phaseb/feasibility.py" "$topic/phasea_v2/inputs.py" \
    "$topic/phasea_v2/estimator.py" | tar -x -C "$snapshot"
mkdir "$snapshot/author"
git -C "$checkout" archive "$official" grad_shift scripts/finetune_lora.py \
    data/persona_vectors_dataset/spanish/misaligned_2.jsonl data/prompt_files/spanish_3samp.json \
    | tar -x -C "$snapshot/author"
echo "[codex] $(date '+%Y-%m-%d %H:%M') Lambda Spanish training-only source=$official freeze=$freeze"
"$shared/envs/inoc/bin/python" -IBu "$snapshot/$stage/stage.py" --helpers "$snapshot/$topic" \
    --checkout "$checkout" --source "$snapshot/author" --snapshot "$base" --scratch "$snapshot" \
    --author-python "$author_python" --commit "$freeze" "${args[@]}"
