#!/bin/bash
#SBATCH --job-name=t2ir-regen
#SBATCH --partition=h200q
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --time=01:00:00
#SBATCH --output=logs/%x-%j.out
# 判準 B 盲審影像：重生成未保存的項目（通常 <= 200 張，數分鐘）。
set -euo pipefail
source slurm/env.sh
quota -g -s || true  # 磁碟用量（本題上限 35 GB，含已用 25.5 GB；剩餘群組額度不足 20 GB 時暫停）
$PY scripts/regen_images.py
