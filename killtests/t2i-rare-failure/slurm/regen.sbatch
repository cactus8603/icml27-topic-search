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
$PY scripts/regen_images.py
