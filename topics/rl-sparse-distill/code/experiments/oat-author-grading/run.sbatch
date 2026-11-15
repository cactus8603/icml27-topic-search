#!/usr/bin/env bash
#SBATCH --job-name=oat-author-grade
#SBATCH --partition=h200q
#SBATCH --gres=none
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=12G
#SBATCH --time=00:30:00
#SBATCH --output=/work/ChihChungHsu/cactus8603/icml/tmp/oat-author-grade-%j.log
set -euo pipefail

root=/work/ChihChungHsu/cactus8603
shared="$root/icml"
topic="$root/icml27-topic-search/codex/experiments/oat-author-grading"
stage="$shared/tmp/oat-author-grading-dfca49d"
freeze=${1:?Pass the reviewed freeze commit}
job=${SLURM_JOB_ID:?Use sbatch for historical grading}
export TZ=Asia/Taipei PYTHONDONTWRITEBYTECODE=1 PYTHONUNBUFFERED=1
export TMPDIR="$shared/tmp" HF_HOME="$shared/hf_home"
export HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 HF_DATASETS_OFFLINE=1
export OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1

"$shared/envs/vllm033/bin/python" -B "$topic/regrade.py" --assets "$stage" \
  --freeze "$freeze" --job-id "$job" --workers 8 \
  --output "$topic/results/$job/author-grade-aggregate.json"
