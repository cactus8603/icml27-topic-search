#!/usr/bin/env bash
#SBATCH --job-name=lambda-gcd-eligibility
#SBATCH --partition=h200q
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --time=02:00:00
#SBATCH --array=0-14%1
#SBATCH --output=/work/ChihChungHsu/cactus8603/icml/tmp/lambda-gcd-%A_%a.log
set -euo pipefail

shared=/work/ChihChungHsu/cactus8603/icml
repo=/work/ChihChungHsu/cactus8603/icml27-topic-search
stage=codex/experiments/lambda-gcd-run
gcd=codex/experiments/lambda-gcd
training=codex/experiments/lambda-author-training
topic=killtests/inoculation-lambda
source="$shared/ext/shifting_the_gradient"
material=64eb9427f4bb14adfeff9bc25fb38b65ba728032
worker=f7bf7002f111d775b7a1bb5b01773fe6a12a5e7b
helper=1280b1ff17b18684358fa9a56227d5df5c44e43b
author=6964328411bf0ae49de47fa1bdfcfe8f94a75cce
freeze=${1:?Pass a full frozen Git commit}
[[ "$freeze" =~ ^[0-9a-f]{40}$ && "$#" == 2 ]]
changes=$(git -C "$repo" status --porcelain --untracked-files=all -- "$stage")
[[ -z "$changes" ]]
git -C "$repo" diff --quiet "$freeze" HEAD -- "$stage"
if [[ "$2" != --check-only ]]; then
    git -C "$repo" cat-file -e "$freeze:$stage/AUTHORIZATION.json"
    [[ "$2" == /* && -f "$2" ]]
fi

export TZ=Asia/Taipei TMPDIR="$shared/tmp" HF_HOME="$shared/hf_home" PYTHONDONTWRITEBYTECODE=1
export HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 HF_HUB_DISABLE_IMPLICIT_TOKEN=1
export OMP_NUM_THREADS="${SLURM_CPUS_PER_TASK:-2}" TOKENIZERS_PARALLELISM=false
unset OPENAI_API_KEY OPENAI_BASE_URL HF_TOKEN HUGGING_FACE_HUB_TOKEN HUGGINGFACE_HUB_TOKEN
snapshot=$(mktemp -d "$TMPDIR/lambda-gcd-cell-${SLURM_JOB_ID:-check}.XXXXXXXX")
trap 'rm -r --one-file-system -- "$snapshot"' EXIT
git -C "$repo" archive "$freeze" "$stage" | tar -x -C "$snapshot"
git -C "$repo" archive "$material" "$gcd" | tar -x -C "$snapshot"
git -C "$repo" archive "$worker" "$training/worker.py" | tar -x -C "$snapshot"
git -C "$repo" archive "$helper" "$topic/phaseb/feasibility.py" "$topic/phasea_v2/inputs.py" \
    "$topic/phasea_v2/estimator.py" "$topic/configs/criteria.yaml" "$topic/configs/reference.json" | tar -x -C "$snapshot"
mkdir "$snapshot/author"
git -C "$source" archive "$author" grad_shift scripts/finetune_lora.py | tar -x -C "$snapshot/author"
python="$shared/envs/inoc-phaseb-author-6964328/env/bin/python"
entry="$snapshot/$stage/run_cell.py"
if [[ "$2" == --check-only ]]; then
    "$python" -IBu "$entry" --source "$snapshot/author"
else
    task=${SLURM_ARRAY_TASK_ID:?Submit through the reviewed Slurm array}
    array=${SLURM_ARRAY_JOB_ID:?Submit through the reviewed Slurm array}
    [[ "$task" =~ ^[0-9]+$ && "$array" =~ ^[0-9]+$ ]]
    "$python" -IBu "$entry" --source "$snapshot/author" --cell "$task" --freeze "$freeze" --scratch "$snapshot" \
        --output "$shared/runs/inoculation-lambda/gcd-reliability-$array" --gate2 "$2" --helpers "$snapshot/$topic"
fi
