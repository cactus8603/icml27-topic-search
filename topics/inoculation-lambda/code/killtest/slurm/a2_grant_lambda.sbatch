#!/usr/bin/env bash
#SBATCH --job-name=lam_grant
#SBATCH --time=04:00:00
#SBATCH --partition=h200q
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=96G
#SBATCH --output=/work/ChihChungHsu/cactus8603/icml27-topic-search/killtests/inoculation-lambda/logs/%x_%j.out
# Grant（2604.16423）七個 trait 的 Λ（bf16，LoRA-B，3 個 A 種子 × 4 折）。預估 1–2 GPU 小時。
set -euo pipefail
cd /work/ChihChungHsu/cactus8603/icml27-topic-search/killtests/inoculation-lambda
export TMPDIR=/work/ChihChungHsu/cactus8603/icml/tmp HF_HOME=/work/ChihChungHsu/cactus8603/icml/hf_home HF_HUB_OFFLINE=1
PY=/work/ChihChungHsu/cactus8603/icml/envs/inoc/bin/python
echo "git: $(git -C ../.. rev-parse HEAD 2>/dev/null || echo none) job: ${SLURM_JOB_ID:-local}" | tee -a logs/RUNLOG.md
for t in evil sycophantic hallucinating pirate spanish allcaps optimistic; do
  $PY scripts/compute_lambda.py --config configs/grant_${t}_qwen7b.yaml
done
