#!/usr/bin/env bash
#SBATCH --job-name=lcs-top-two
#SBATCH --partition=h200q
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=32G
#SBATCH --time=00:45:00
#SBATCH --output=/work/ChihChungHsu/cactus8603/icml/tmp/lcs-top-two-%j.log
set -euo pipefail

root=/work/ChihChungHsu/cactus8603
shared="$root/icml"
topic="$root/icml27-topic-search/codex/experiments/large-commit-sets/top-two-selector"
revision=08b83a6feb34df1a6011b80c3c00c7563e963b07
mode=${1:?Pass smoke or main}
freeze=${2:?Pass the reviewed freeze}
case "$mode" in smoke|main) ;; *) exit 2 ;; esac
export TMPDIR="$shared/tmp" HF_HOME="$shared/hf_home"
export HF_MODULES_CACHE="$shared/tmp/lcs-top-two-modules-${SLURM_JOB_ID:?Use sbatch}"
export HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 TOKENIZERS_PARALLELISM=false
export PYTHONUNBUFFERED=1 PYTHONDONTWRITEBYTECODE=1 OPENBLAS_NUM_THREADS=1 TZ=Asia/Taipei

"$shared/envs/t2irare/bin/python" -B "$topic/trial.py" --mode "$mode" --freeze "$freeze" \
  --model "$HF_HOME/hub/models--GSAI-ML--LLaDA-8B-Instruct/snapshots/$revision" \
  --train "$shared/data/gsm8k-hf/main/train-00000-of-00001.parquet" \
  --test "$shared/data/gsm8k/main/test-00000-of-00001.parquet" \
  --math "$shared/data/MATH-500/test.jsonl" --parquet-python "$shared/envs/biasgen/bin/python"
