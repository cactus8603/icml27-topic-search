#!/usr/bin/env bash
#SBATCH --job-name=lambda-gcd-family-train-ready
#SBATCH --partition=h200q
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=64G
#SBATCH --time=00:15:00
#SBATCH --output=/work/ChihChungHsu/cactus8603/icml/tmp/lambda-gcd-family-train-ready-%j.log
set -euo pipefail
umask 077

shared=/work/ChihChungHsu/cactus8603/icml
repo=/work/ChihChungHsu/cactus8603/icml27-topic-search
parent=codex/experiments/lambda-gcd-family-design
stage="$parent/training"
materials=64eb9427f4bb14adfeff9bc25fb38b65ba728032
runner=b54a86fb05e845e2c11139822beac8846c63c77a
author=6964328411bf0ae49de47fa1bdfcfe8f94a75cce
freeze=${1:?Pass the full frozen Git commit}
mode=${2:?Use --check-only or readiness}
[[ "$freeze" =~ ^[0-9a-f]{40}$ ]]
files=("$parent/prepare_design.py" "$stage/readiness.py" "$stage/test_readiness.py" "$stage/run.sbatch")
[[ -z "$(git -C "$repo" status --porcelain -- "${files[@]}")" ]]
git -C "$repo" diff --quiet "$freeze" HEAD -- "${files[@]}"

export TZ=Asia/Taipei TMPDIR="$shared/tmp" HF_HOME="$shared/hf_home"
export HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 HF_HUB_DISABLE_IMPLICIT_TOKEN=1
export PYTHONDONTWRITEBYTECODE=1 OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1 TOKENIZERS_PARALLELISM=false
unset OPENAI_API_KEY OPENAI_BASE_URL HF_TOKEN HUGGING_FACE_HUB_TOKEN HUGGINGFACE_HUB_TOKEN
bundle=$(mktemp -d "$TMPDIR/lambda-gcd-family-train-ready-${SLURM_JOB_ID:-check}.XXXXXXXX")
trap 'rm -r --one-file-system -- "$bundle"' EXIT
export XDG_CACHE_HOME="$bundle/cache"
git -C "$repo" archive "$freeze" "${files[@]}" | tar -x -C "$bundle"
git -C "$repo" archive "$materials" codex/experiments/lambda-gcd | tar -x -C "$bundle"
git -C "$repo" archive "$runner" codex/experiments/lambda-gcd-run/run_cell.py | tar -x -C "$bundle"
mkdir "$bundle/author" "$bundle/synthetic"
git -C "$shared/ext/shifting_the_gradient" archive "$author" grad_shift scripts/finetune_lora.py \
    steering_vectors/Qwen_Qwen2.5-7B-Instruct_extract/sycophantic_response_avg_diff_thresh50.pt | tar -x -C "$bundle/author"

python="$shared/envs/inoc-phaseb-author-6964328/env/bin/python"
entry=("$bundle/$stage/readiness.py" --source "$bundle/author" --scratch "$bundle/synthetic" --freeze "$freeze")
case "$mode" in
    --check-only)
        [[ "$#" == 2 ]]
        export CUDA_VISIBLE_DEVICES=''
        export READINESS_AUTHOR_ROOT="$bundle/author"
        core=$("$python" -IB -c 'import os; print(min(os.sched_getaffinity(0)))')
        taskset -c "$core" "$python" -IB "$bundle/$stage/test_readiness.py"
        taskset -c "$core" "$python" -IBu "${entry[@]}"
        ;;
    readiness)
        [[ "$#" == 4 && "${SLURM_JOB_ID:-}" =~ ^[1-9][0-9]*$ && -z "${SLURM_ARRAY_JOB_ID:-}" ]]
        "$python" -IBu "${entry[@]}" \
            --manifest "$3" --manifest-sha256 "$4" \
            --output "$shared/runs/lambda-gcd-family-train-ready/$SLURM_JOB_ID/summary.json"
        ;;
    *) exit 2 ;;
esac
