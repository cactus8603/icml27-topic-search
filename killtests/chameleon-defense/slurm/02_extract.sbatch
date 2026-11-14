#!/bin/bash
#SBATCH --job-name=cham-extract
#SBATCH --partition=h200q
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=96G
#SBATCH --time=03:00:00
#SBATCH --output=logs/%x-%j.out
# 全量激活抽取（約 3.6 萬列；快取約 2.1 GB）。可續跑（已存在的 shard 會跳過）。
set -euo pipefail
source slurm/env.sh
BS=${BS:-16}
$PY -I scripts/extract_acts.py --sets benign harm_train harm_test dec_train dec_test --batch-size $BS --out $RUN/cache
