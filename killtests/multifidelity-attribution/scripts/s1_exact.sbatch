#!/bin/bash
#SBATCH --job-name=mf-exact1
#SBATCH --partition=h200q
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=96G
#SBATCH --time=03:00:00
#SBATCH --array=5-19
#SBATCH --output=/work/ChihChungHsu/cactus8603/icml27-topic-search/killtests/multifidelity-attribution/logs/%x-%A_%a.out
source /work/ChihChungHsu/cactus8603/icml27-topic-search/killtests/multifidelity-attribution/scripts/common.sh
# 階段 1（僅階段 0 結論為 CONTINUE 才送）：精確 Y 擴到 k=200（查詢 50..199，15 個任務各 10 個）。
QS=$((SLURM_ARRAY_TASK_ID*10)); $PYJ $S/exact_y.py --cfg $CFG --qstart $QS --qend $((QS+10))
