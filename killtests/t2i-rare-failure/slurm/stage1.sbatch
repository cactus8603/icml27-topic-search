#!/bin/bash
#SBATCH --job-name=t2ir-s1
#SBATCH --partition=h200q
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=16
#SBATCH --mem=96G
#SBATCH --time=04:00:00
#SBATCH --array=0-5
#SBATCH --output=logs/%x-%A_%a.out
# 階段 1：每個 array 任務 = 1 個模型 x 14 個提示 x seeds [0,2000)，共 28,000 張（推估 ~52 分鐘，未實測）。
# 可續跑：chunk 檔已存在就跳過；超時重送同一指令即可。
set -euo pipefail
source slurm/env.sh
quota -g -s || true  # 磁碟用量（本題上限 35 GB，含已用 25.5 GB；剩餘群組額度不足 20 GB 時暫停）
MODELS=(sd14 uce esd salun rece advunlearn)
M=${MODELS[$SLURM_ARRAY_TASK_ID]}
$PY scripts/generate.py --model "$M" --prompts all --seed-start 0 --seed-end 2000 --batch-size ${BS:-32}
