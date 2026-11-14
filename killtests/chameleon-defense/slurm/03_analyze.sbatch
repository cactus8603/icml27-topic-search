#!/bin/bash
#SBATCH --job-name=cham-analyze
#SBATCH --partition=h200q
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=96G
#SBATCH --time=02:00:00
#SBATCH --output=logs/%x-%j.out
# 閘門 1（合成）→ 閘門 2–4 原始數字 → 機械判讀。邏輯迴歸用 GPU（torch LBFGS）。
set -euo pipefail
source slurm/env.sh
cache=${CHAM_CACHE:-$RUN/cache}
result_dir="$EXP/results/analysis_${SLURM_JOB_ID:?run through Slurm}"
mkdir "$result_dir"
$PY -I scripts/selftest.py --out "$result_dir"
$PY -I scripts/analyze.py all --cache "$cache" --results "$result_dir"
$PY -I scripts/aggregate.py --results "$result_dir"
