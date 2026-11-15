#!/usr/bin/env bash
#SBATCH --job-name=lambda-gcd-family-outcomes
#SBATCH --partition=h200q
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --time=00:25:00
#SBATCH --output=/work/ChihChungHsu/cactus8603/icml/tmp/lambda-gcd-family-outcomes-%A_%a.log
set -euo pipefail
umask 077

shared=/work/ChihChungHsu/cactus8603/icml
repo="${LAMBDA_GCD_REPO:-/work/ChihChungHsu/cactus8603/icml27-topic-search}"
parent=codex/experiments/lambda-gcd-family-design
stage="$parent/outcomes"
materials=64eb9427f4bb14adfeff9bc25fb38b65ba728032
runner=b54a86fb05e845e2c11139822beac8846c63c77a
author=6964328411bf0ae49de47fa1bdfcfe8f94a75cce
freeze=${1:?Pass the full frozen Git commit}
mode=${2:?Use --check-only, train, judge, train-auto, or judge-auto}
[[ "$freeze" =~ ^[0-9a-f]{40}$ ]]
files=("$parent/prepare_design.py" "$parent/training/readiness.py" "$stage")
[[ -z "$(git -C "$repo" status --porcelain --untracked-files=all -- "${files[@]}")" ]]
git -C "$repo" diff --quiet "$freeze" HEAD -- "${files[@]}"

case "$mode" in
    --check-only)
        [[ "$#" == 2 || "$#" == 4 ]]
        entry_mode=check
        options=()
        if [[ "$#" == 4 ]]; then options+=(--manifest "$3" --manifest-sha256 "$4"); fi
        ;;
    train|judge|train-auto|judge-auto)
        entry_mode=${mode%-auto}
        if [[ "$mode" == *-auto ]]; then
            expected_args=6
            options=(--manifest "${3:-}" --manifest-sha256 "${4:-}"
                     --readiness-job "${5:-}" --prediction-join-job "${6:-}")
        else
            expected_args=8
            options=(--manifest "${3:-}" --manifest-sha256 "${4:-}" --readiness "${5:-}" --readiness-sha256 "${6:-}"
                     --predictions "${7:-}" --predictions-sha256 "${8:-}")
        fi
        if [[ "$entry_mode" == judge ]]; then expected_args=$((expected_args + 1)); fi
        [[ "$#" == "$expected_args" && "${SLURM_JOB_ID:-}" =~ ^[1-9][0-9]*$ ]]
        if [[ "$entry_mode" == train ]]; then
            [[ "${SLURM_ARRAY_JOB_ID:-}" =~ ^[1-9][0-9]*$ && "${SLURM_ARRAY_TASK_ID:-}" =~ ^[0-9]+$ ]]
            options+=(--array "$SLURM_ARRAY_JOB_ID" --task "$SLURM_ARRAY_TASK_ID")
        else
            [[ "${!expected_args}" =~ ^[1-9][0-9]*$ && -z "${SLURM_ARRAY_JOB_ID:-}" ]]
            options+=(--array "${!expected_args}")
        fi
        ;;
    *) exit 2 ;;
esac

export TZ=Asia/Taipei TMPDIR="$shared/tmp" HF_HOME="$shared/hf_home" PYTHONDONTWRITEBYTECODE=1
export HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 HF_HUB_DISABLE_IMPLICIT_TOKEN=1
export OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 TOKENIZERS_PARALLELISM=false
unset OPENAI_API_KEY OPENAI_BASE_URL HF_TOKEN HUGGING_FACE_HUB_TOKEN HUGGINGFACE_HUB_TOKEN
bundle=$(mktemp -d "$TMPDIR/lambda-gcd-family-outcomes-${SLURM_JOB_ID:-check}.XXXXXXXX")
trap 'rm -r --one-file-system -- "$bundle"' EXIT
export XDG_CACHE_HOME="$bundle/cache"
git -C "$repo" archive "$freeze" "${files[@]}" | tar -xf - -C "$bundle"
git -C "$repo" archive "$materials" codex/experiments/lambda-gcd | tar -xf - -C "$bundle"
git -C "$repo" archive "$runner" codex/experiments/lambda-gcd-run/run_cell.py \
    codex/experiments/lambda-author-training/worker.py | tar -xf - -C "$bundle"
mkdir "$bundle/author" "$bundle/cell"
git -C "$shared/ext/shifting_the_gradient" archive "$author" grad_shift scripts/finetune_lora.py \
    steering_vectors/Qwen_Qwen2.5-7B-Instruct_extract/sycophantic_response_avg_diff_thresh50.pt | tar -xf - -C "$bundle/author"

python="$shared/envs/inoc-phaseb-author-6964328/env/bin/python"
entry=("$bundle/$stage/cell.py" "$entry_mode" --source "$bundle/author" --scratch "$bundle/cell" --freeze "$freeze")
if [[ "$mode" == --check-only ]]; then
    export CUDA_VISIBLE_DEVICES=''
    core=$("$python" -IB -c 'import os; print(min(os.sched_getaffinity(0)))')
    taskset -c "$core" "$python" -IB "$bundle/$stage/test_panel.py"
    taskset -c "$core" "$python" -IBu "$bundle/$stage/train_entry.py" --help >/dev/null
    taskset -c "$core" "$python" -IBu "${entry[@]}" "${options[@]}"
else
    "$python" -IBu "${entry[@]}" "${options[@]}"
fi
