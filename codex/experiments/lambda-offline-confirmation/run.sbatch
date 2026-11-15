#!/usr/bin/env bash
#SBATCH --job-name=lambda-gcd-offline
#SBATCH --partition=h200q
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --time=00:25:00
#SBATCH --array=0-14%1
#SBATCH --output=/work/ChihChungHsu/cactus8603/icml/tmp/lambda-gcd-offline-%A_%a.log
set -euo pipefail
umask 077

shared=/work/ChihChungHsu/cactus8603/icml
repo=/work/ChihChungHsu/cactus8603/icml27-topic-search
stage=codex/experiments/lambda-offline-confirmation
gcd=codex/experiments/lambda-gcd
staged=codex/experiments/lambda-gcd-run
training=codex/experiments/lambda-author-training
material=64eb9427f4bb14adfeff9bc25fb38b65ba728032
runner=b54a86fb05e845e2c11139822beac8846c63c77a
worker=f7bf7002f111d775b7a1bb5b01773fe6a12a5e7b
author=6964328411bf0ae49de47fa1bdfcfe8f94a75cce
freeze=${1:?Pass the full frozen Git commit}
mode=${2:-train}
[[ "$freeze" =~ ^[0-9a-f]{40}$ ]]
[[ -z "$(git -C "$repo" status --porcelain --untracked-files=all -- "$stage")" ]]
git -C "$repo" diff --quiet "$freeze" HEAD -- "$stage"

case "$mode" in
    --check-only) [[ "$#" == 2 ]]; mode=check ;;
    --judge) [[ "$#" == 3 && "$3" =~ ^[1-9][0-9]*$ ]]; mode=judge; array=$3 ;;
    train)
        [[ "$#" == 1 ]]
        array=${SLURM_ARRAY_JOB_ID:?Submit this experiment as a Slurm array}
        task=${SLURM_ARRAY_TASK_ID:?Submit this experiment as a Slurm array}
        [[ "$array" =~ ^[1-9][0-9]*$ && "$task" =~ ^[0-9]+$ ]]
        ;;
    *) exit 2 ;;
esac

export TZ=Asia/Taipei TMPDIR="$shared/tmp" HF_HOME="$shared/hf_home" PYTHONDONTWRITEBYTECODE=1
export HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 HF_HUB_DISABLE_IMPLICIT_TOKEN=1
export OMP_NUM_THREADS="${SLURM_CPUS_PER_TASK:-2}" TOKENIZERS_PARALLELISM=false
unset OPENAI_API_KEY OPENAI_BASE_URL HF_TOKEN HUGGING_FACE_HUB_TOKEN HUGGINGFACE_HUB_TOKEN
snapshot=$(mktemp -d "$TMPDIR/lambda-gcd-offline-${SLURM_JOB_ID:-local}.XXXXXXXX")
trap 'rm -r --one-file-system -- "$snapshot"' EXIT
git -C "$repo" archive "$freeze" "$stage" | tar -x -C "$snapshot"
git -C "$repo" archive "$material" "$gcd" | tar -x -C "$snapshot"
git -C "$repo" archive "$runner" "$staged/run_cell.py" "$staged/train_entry.py" | tar -x -C "$snapshot"
git -C "$repo" archive "$worker" "$training/worker.py" | tar -x -C "$snapshot"
mkdir "$snapshot/author"
git -C "$shared/ext/shifting_the_gradient" archive "$author" grad_shift scripts/finetune_lora.py | tar -x -C "$snapshot/author"
python="$shared/envs/inoc-phaseb-author-6964328/env/bin/python"
entry="$snapshot/$stage/pilot.py"

case "$mode" in
    check) "$python" -IBu "$entry" check --source "$snapshot/author" ;;
    judge) "$python" -IBu "$entry" judge --source "$snapshot/author" --freeze "$freeze" --array "$array" ;;
    train)
        "$python" -IBu "$entry" train --source "$snapshot/author" --freeze "$freeze" --array "$array" \
            --task "$task" --scratch "$snapshot"
        ;;
esac
