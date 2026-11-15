#!/usr/bin/env bash
#SBATCH --job-name=lambda-gcd-family-preflight
#SBATCH --partition=h200q
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gres=none
#SBATCH --cpus-per-task=1
#SBATCH --mem=8G
#SBATCH --time=00:15:00
#SBATCH --output=/work/ChihChungHsu/cactus8603/icml/tmp/lambda-gcd-family-preflight-%j.log
set -euo pipefail
umask 077

shared=/work/ChihChungHsu/cactus8603/icml
repo=/work/ChihChungHsu/cactus8603/icml27-topic-search
stage=codex/experiments/lambda-gcd-family-design
materials=64eb9427f4bb14adfeff9bc25fb38b65ba728032
runner=b54a86fb05e845e2c11139822beac8846c63c77a
author=6964328411bf0ae49de47fa1bdfcfe8f94a75cce
freeze=${1:?Pass the full frozen Git commit}
[[ "$freeze" =~ ^[0-9a-f]{40}$ ]]
git -C "$repo" diff --quiet "$freeze" HEAD -- "$stage/prepare_design.py" "$stage/run.sbatch" "$stage/test_prepare_design.py"

export TZ=Asia/Taipei TMPDIR="$shared/tmp" HF_HOME="$shared/hf_home" PYTHONDONTWRITEBYTECODE=1
export HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 HF_HUB_DISABLE_IMPLICIT_TOKEN=1
export OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1 TOKENIZERS_PARALLELISM=false CUDA_VISIBLE_DEVICES=''
unset OPENAI_API_KEY OPENAI_BASE_URL HF_TOKEN HUGGING_FACE_HUB_TOKEN HUGGINGFACE_HUB_TOKEN
bundle=$(mktemp -d "$TMPDIR/lambda-gcd-family-preflight-${SLURM_JOB_ID:-check}.XXXXXXXX")
trap 'rm -r --one-file-system -- "$bundle"' EXIT
export XDG_CACHE_HOME="$bundle/cache"
git -C "$repo" archive "$freeze" "$stage/prepare_design.py" "$stage/test_prepare_design.py" | tar -x -C "$bundle"
git -C "$repo" archive "$materials" codex/experiments/lambda-gcd | tar -x -C "$bundle"
git -C "$repo" archive "$runner" codex/experiments/lambda-gcd-run/run_cell.py | tar -x -C "$bundle"
mkdir "$bundle/author"
git -C "$shared/ext/shifting_the_gradient" archive "$author" grad_shift scripts/finetune_lora.py \
    steering_vectors/Qwen_Qwen2.5-7B-Instruct_extract/sycophantic_response_avg_diff_thresh50.pt | tar -x -C "$bundle/author"
python="$shared/envs/inoc-phaseb-author-6964328/env/bin/python"
entry="$bundle/$stage/prepare_design.py"
core=$("$python" -IB -c 'import os; print(min(os.sched_getaffinity(0)))')

case "${2:-}" in
    --check-only)
        [[ $# -eq 2 ]]
        taskset -c "$core" "$python" -B -m unittest discover -s "$bundle/$stage" -p 'test_prepare_design.py'
        taskset -c "$core" "$python" -IBu "$entry" --source "$bundle/author"
        ;;
    "")
        [[ $# -eq 1 && "${SLURM_JOB_ID:-}" =~ ^[1-9][0-9]*$ ]]
        taskset -c "$core" "$python" -IBu "$entry" --source "$bundle/author" \
            --output "$shared/data/lambda-gcd/family-design-$SLURM_JOB_ID"
        ;;
    *) exit 2 ;;
esac
