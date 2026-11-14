#!/bin/bash
#SBATCH --job-name=mf-retrain
#SBATCH --partition=h200q
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --time=04:00:00
#SBATCH --array=0-7
#SBATCH --output=/work/ChihChungHsu/cactus8603/icml27-topic-search/killtests/multifidelity-attribution/logs/%x-%A_%a.out
source /work/ChihChungHsu/cactus8603/icml27-topic-search/killtests/multifidelity-attribution/scripts/common.sh
# M=200 個移除 1% 的重訓（+4 次無移除重複，由 task 0 做）。8 個任務輪流分配 j。可續跑。
$PYJ $S/retrain.py --cfg $CFG --task $SLURM_ARRAY_TASK_ID --ntasks 8
