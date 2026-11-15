#!/usr/bin/env bash
#SBATCH --job-name=lambda-gcd-family-step0
#SBATCH --partition=h200q
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=64G
#SBATCH --time=00:15:00
#SBATCH --output=/work/ChihChungHsu/cactus8603/icml/tmp/lambda-gcd-family-step0-%A_%a.log
set -euo pipefail
umask 077

root=/work/ChihChungHsu/cactus8603
repo="$root/icml27-topic-search"
shared="$root/icml"
parent=codex/experiments/lambda-gcd-family-design
stage="$parent/signals"
freeze=${1:?Pass the full frozen Git commit}
mode=${2:?Use --check-only, cell or join}
[[ "$freeze" =~ ^[0-9a-f]{40}$ ]]
files=("$parent/prepare_design.py" "$parent/training/readiness.py" "$stage/step0.py" "$stage/run.sbatch" "$stage/test_step0.py")
[[ -z "$(git -C "$repo" status --porcelain -- "${files[@]}")" ]]
git -C "$repo" diff --quiet "$freeze" HEAD -- "${files[@]}"

case "$mode" in
    --check-only) [[ "$#" == 2 && -z "${SLURM_JOB_ID:-}" ]] ;;
    cell) [[ "$#" == 6 && "${SLURM_ARRAY_JOB_ID:-}" =~ ^[1-9][0-9]*$ && "${SLURM_ARRAY_TASK_ID:-}" =~ ^[0-9]+$ ]] ;;
    join) [[ "$#" == 7 && -z "${SLURM_ARRAY_JOB_ID:-}" && "${SLURM_JOB_ID:-}" =~ ^[1-9][0-9]*$ && "$7" =~ ^[1-9][0-9]*$ ]] ;;
    *) exit 2 ;;
esac

export TZ=Asia/Taipei TMPDIR="$shared/tmp" HF_HOME="$shared/hf_home" PYTHONDONTWRITEBYTECODE=1
export HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 HF_HUB_DISABLE_IMPLICIT_TOKEN=1
export OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1 TOKENIZERS_PARALLELISM=false
unset OPENAI_API_KEY OPENAI_BASE_URL HF_TOKEN HUGGING_FACE_HUB_TOKEN HUGGINGFACE_HUB_TOKEN
bundle=$(mktemp -d "$TMPDIR/lambda-gcd-family-step0-${SLURM_JOB_ID:-check}.XXXXXXXX")
trap 'rm -r --one-file-system -- "$bundle"' EXIT
export XDG_CACHE_HOME="$bundle/cache"
git -C "$repo" archive "$freeze" "${files[@]}" | tar -x -C "$bundle"
git -C "$repo" archive 64eb9427f4bb14adfeff9bc25fb38b65ba728032 codex/experiments/lambda-gcd | tar -x -C "$bundle"
git -C "$repo" archive b54a86fb05e845e2c11139822beac8846c63c77a codex/experiments/lambda-gcd-run/run_cell.py | tar -x -C "$bundle"
git -C "$repo" archive b401effd9f560401e3ec9bb5ac0de70bb3275b4e killtests/inoculation-lambda/scripts/compute_lambda.py | tar -x -C "$bundle"
mkdir "$bundle/author" "$bundle/scratch"
git -C "$shared/ext/shifting_the_gradient" archive 6964328411bf0ae49de47fa1bdfcfe8f94a75cce grad_shift scripts/finetune_lora.py \
    steering_vectors/Qwen_Qwen2.5-7B-Instruct_extract/sycophantic_response_avg_diff_thresh50.pt | tar -x -C "$bundle/author"

python="$shared/envs/inoc-phaseb-author-6964328/env/bin/python"
entry=("$bundle/$stage/step0.py" --source "$bundle/author" --scratch "$bundle/scratch" --freeze "$freeze")
if [[ "$mode" == --check-only ]]; then
    export CUDA_VISIBLE_DEVICES=''
    core=$("$python" -IB -c 'import os; print(min(os.sched_getaffinity(0)))')
    GCD_STEP0_TEST_AUTHOR="$bundle/author" taskset -c "$core" "$python" -IBu "$bundle/$stage/test_step0.py"
    taskset -c "$core" "$python" -IBu "${entry[@]}" check
    exit 0
fi

data_job=$3
ready_job=$5
[[ "$data_job" =~ ^[1-9][0-9]*$ && "$ready_job" =~ ^[1-9][0-9]*$ && "$4" =~ ^[0-9a-f]{64}$
    && ( "$6" =~ ^[0-9a-f]{64}$ || "$6" == receipt ) ]]
[[ "$(sacct -X -n -P -j "$data_job" -o JobID,State,ExitCode,JobName%50)" == "$data_job|COMPLETED|0:0|lambda-gcd-family-preflight" ]]
[[ "$(sacct -X -n -P -j "$ready_job" -o JobID,State,ExitCode,JobName%50)" == "$ready_job|COMPLETED|0:0|lambda-gcd-family-train-ready" ]]
entry+=(--manifest "$shared/data/lambda-gcd/family-design-$data_job/manifest.json" --manifest-sha256 "$4"
    --readiness "$shared/runs/lambda-gcd-family-train-ready/$ready_job/summary.json"
    --readiness-log "$shared/tmp/lambda-gcd-family-train-ready-$ready_job.log" --readiness-sha256 "$6")
if [[ "$mode" == cell ]]; then
    "$python" -IBu "${entry[@]}" cell --array "$SLURM_ARRAY_JOB_ID" --task "$SLURM_ARRAY_TASK_ID"
else
    export CUDA_VISIBLE_DEVICES=''
    array=$7
    sacct -X -n -P -j "$array" -o JobID,State,ExitCode,JobName%50 | awk -F'|' -v array="$array" '
        $2 != "COMPLETED" || $3 != "0:0" || $4 != "lambda-gcd-family-step0" { bad=1; next }
        { split($1, row, "_"); if (row[1] != array || row[2] !~ /^(0|[1-9][0-9]*)$/ || row[2] > 50 || seen[row[2]]++) bad=1; else count++ }
        END { exit bad || count != 51 }'
    "$python" -IBu "${entry[@]}" join --array "$array"
fi
