#!/bin/bash -l
#SBATCH --job-name=commit-sets-r2
#SBATCH --partition=h200q
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=48G
#SBATCH --time=00:30:00

set -euo pipefail
source /work/ChihChungHsu/cactus8603/icml27-topic-search/codex/experiments/large-commit-sets/round2/paths.sh
mode="${1:?Pass check, smoke, train or test}"
freeze "${2:?Pass the frozen commit hash}"
revision=08b83a6feb34df1a6011b80c3c00c7563e963b07
model="${LLADA_MODEL_PATH:-$HF_HOME/hub/models--GSAI-ML--LLaDA-8B-Instruct/snapshots/$revision}"
extra=()
case "$mode" in
  check|smoke) ;;
  train|test) extra=(--output "$repo/$topic/results/$mode-$SLURM_JOB_ID.jsonl") ;;
  *) printf '%s\n' 'Pass check, smoke, train or test.' >&2; exit 2 ;;
esac
"$python" -B "$frozen_work/$topic/collect.py" --mode "$mode" --model "$model" \
  --train "$train_data" --test "$test_data" --parquet-python "$parquet_python" "${extra[@]}"
