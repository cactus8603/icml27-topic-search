#!/usr/bin/env bash
#SBATCH --job-name=lam_gate1_fd
#SBATCH --time=06:00:00
#SBATCH --partition=h200q
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=96G
#SBATCH --output=/work/ChihChungHsu/cactus8603/icml27-topic-search/killtests/inoculation-lambda/logs/%x_%j.out
# 閘門 1：有限差分（fp32，7 個 trait）。預估 1.5–3 GPU 小時。
set -euo pipefail
cd /work/ChihChungHsu/cactus8603/icml27-topic-search/killtests/inoculation-lambda
export TMPDIR=/work/ChihChungHsu/cactus8603/icml/tmp HF_HOME=/work/ChihChungHsu/cactus8603/icml/hf_home HF_HUB_OFFLINE=1
PY=/work/ChihChungHsu/cactus8603/icml/envs/inoc/bin/python
echo "git: $(git -C ../.. rev-parse HEAD 2>/dev/null || echo none) job: ${SLURM_JOB_ID:-local}" | tee -a logs/RUNLOG.md
for t in evil sycophantic hallucinating pirate spanish allcaps optimistic; do
  $PY scripts/compute_lambda.py --config configs/grant_${t}_qwen7b.yaml --finite_diff
done
$PY scripts/aggregate.py --phase a | tee results/aggregate_phaseA_gate1.txt
