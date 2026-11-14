#!/bin/bash
#SBATCH --job-name=t2ir-s2
#SBATCH --partition=h200q
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=16
#SBATCH --mem=96G
#SBATCH --time=04:00:00
#SBATCH --output=logs/%x-%A_%a.out
# 階段 2：--array 範圍由 select_stage2_cells.py 印出（0 到 任務數-1）。每任務 1 格 x 25,000 張（推估 ~46 分鐘，未實測）。
# 送出：sbatch --export=ALL,TASKS=configs/stage2a_tasks.tsv --array=0-<N-1> slurm/stage2.sbatch   （2b 改用 stage2b_tasks.tsv）
set -euo pipefail
source slurm/env.sh
quota -g -s || true  # 磁碟用量（本題上限 35 GB，含已用 25.5 GB；剩餘群組額度不足 20 GB 時暫停）
LINE=$(sed -n "$((SLURM_ARRAY_TASK_ID+1))p" "${TASKS:?請指定 TASKS=configs/stage2a_tasks.tsv 或 stage2b_tasks.tsv}")
read -r M P S E <<< "$LINE"
echo "task $SLURM_ARRAY_TASK_ID: model=$M prompt=$P seeds=[$S,$E)"
$PY scripts/generate.py --model "$M" --prompts "$P" --seed-start "$S" --seed-end "$E" --batch-size ${BS:-32}
