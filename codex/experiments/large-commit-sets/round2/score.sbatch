#!/bin/bash -l
#SBATCH --job-name=commit-sets-r2-score
#SBATCH --partition=h200q
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=12G
#SBATCH --time=00:30:00

set -euo pipefail
source /work/ChihChungHsu/cactus8603/icml27-topic-search/codex/experiments/large-commit-sets/round2/paths.sh
mode=score
freeze "${1:?Pass the frozen commit hash}"
train_job="${2:?Pass the train collection job ID}"
test_job="${3:?Pass the test collection job ID}"
if [[ ! "$train_job" =~ ^[0-9]+$ || ! "$test_job" =~ ^[0-9]+$ ]]; then
  printf '%s\n' 'Train and test job IDs must be numeric.' >&2
  exit 2
fi
results="$repo/$topic/results"
"$python" -B "$frozen_work/$topic/score.py" --train-data "$train_data" \
  --test-data "$test_data" --parquet-python "$parquet_python" \
  --train-artifact "$results/train-$train_job.jsonl" --test-artifact "$results/test-$test_job.jsonl" \
  --output "$results/verdict-$SLURM_JOB_ID.json"
