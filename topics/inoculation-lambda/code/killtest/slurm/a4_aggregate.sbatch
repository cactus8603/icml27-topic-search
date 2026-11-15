#!/usr/bin/env bash
#SBATCH --job-name=lam_agg
#SBATCH --time=00:20:00
#SBATCH --partition=h200q
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=96G
#SBATCH --output=/work/ChihChungHsu/cactus8603/icml27-topic-search/killtests/inoculation-lambda/logs/%x_%j.out
# 彙整：閘門 1 PASS/FAIL ＋ 對照論文數字的 Spearman（參考）。實際只用 CPU，但依要求掛 1 GPU（可改 --gres 為空，由審查者決定）。
set -euo pipefail
cd /work/ChihChungHsu/cactus8603/icml27-topic-search/killtests/inoculation-lambda
export TMPDIR=/work/ChihChungHsu/cactus8603/icml/tmp HF_HOME=/work/ChihChungHsu/cactus8603/icml/hf_home HF_HUB_OFFLINE=1
PY=/work/ChihChungHsu/cactus8603/icml/envs/inoc/bin/python
echo "git: $(git -C ../.. rev-parse HEAD 2>/dev/null || echo none) job: ${SLURM_JOB_ID:-local}" | tee -a logs/RUNLOG.md
$PY scripts/aggregate.py --phase a | tee results/aggregate_phaseA.txt
