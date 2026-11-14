#!/usr/bin/env bash
#SBATCH --job-name=fv3-token
#SBATCH --partition=h200q
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=48G
#SBATCH --time=00:20:00
#SBATCH --output=/work/ChihChungHsu/cactus8603/icml/tmp/fv3-token-%j.log
set -euo pipefail

root=/work/ChihChungHsu/cactus8603
shared="$root/icml"
topic="$root/icml27-topic-search/codex/experiments/fallible-verifier"
freeze=${1:?Pass the reviewed freeze commit}
run="$topic/results/phase3-synthetic/${SLURM_JOB_ID:?Use sbatch to run the GPU diagnostic}"
python="$shared/envs/vllm033/bin/python"
weak="$shared/models/Qwen3-0.6B"
strong="$shared/hf_home/hub/models--Qwen--Qwen3-32B-AWQ/snapshots/0499c3ac83fdef8810b907a23894ba91e95eddd8"
export TZ=Asia/Taipei PYTHONDONTWRITEBYTECODE=1 PYTHONUNBUFFERED=1
export TMPDIR="$shared/tmp" HF_HOME="$shared/hf_home"
export XDG_CACHE_HOME="$shared/tmp/fv3-token-$SLURM_JOB_ID"
export HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 TOKENIZERS_PARALLELISM=false
export OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 VLLM_NO_USAGE_STATS=1
# Compute nodes use the same PyTorch sampler as the Round 2 deployment.
export VLLM_USE_FLASHINFER_SAMPLER=0

{
  "$python" -B "$topic/phase3_synthetic.py" --freeze "$freeze" --run-dir "$run" --role weak --model-dir "$weak"
  "$python" -B "$topic/phase3_synthetic.py" --freeze "$freeze" --run-dir "$run" --role strong --model-dir "$strong"
} 2>&1 | awk '{print "[codex] " strftime("%Y-%m-%d %H:%M") " " $0; fflush()}'
