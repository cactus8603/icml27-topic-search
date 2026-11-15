#!/bin/bash
#SBATCH --job-name=mf-exact
#SBATCH --partition=h200q
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=96G
#SBATCH --time=03:00:00
#SBATCH --array=0-4
#SBATCH --output=/work/ChihChungHsu/cactus8603/icml27-topic-search/killtests/multifidelity-attribution/logs/%x-%A_%a.out
source /work/ChihChungHsu/cactus8603/icml27-topic-search/killtests/multifidelity-attribution/scripts/common.sh
# 精確 Y，階段 0：k=50，每個任務 10 個查詢（一個 qchunk）。
QS=$((SLURM_ARRAY_TASK_ID*10)); $PYJ $S/exact_y.py --cfg $CFG --qstart $QS --qend $((QS+10))
