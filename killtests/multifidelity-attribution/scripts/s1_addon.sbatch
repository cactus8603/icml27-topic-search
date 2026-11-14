#!/bin/bash
#SBATCH --job-name=mf-addon
#SBATCH --partition=h200q
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=96G
#SBATCH --time=03:00:00
#SBATCH --array=20-39
#SBATCH --output=/work/ChihChungHsu/cactus8603/icml27-topic-search/killtests/multifidelity-attribution/logs/%x-%A_%a.out
source /work/ChihChungHsu/cactus8603/icml27-topic-search/killtests/multifidelity-attribution/scripts/common.sh
# 灰區唯一加碼（審查後定案 2026-11-14，S6）：增加查詢，不增加重訓。精確 Y 再算查詢 200..399（20 個任務各 10 個；
# 查詢 = query_seed=20261114 排列的第 200..399 個，與 k_total=400 的查詢池一致，事先固定）。
# 重訓與便宜方法在階段 0 就已記錄／計算全部 400 個查詢，不必重跑。之後 aggregate --stage 1 --addon 以 k=400（B=80）重判，不得再加碼。
QS=$((SLURM_ARRAY_TASK_ID*10)); $PYJ $S/exact_y.py --cfg $CFG --qstart $QS --qend $((QS+10))
