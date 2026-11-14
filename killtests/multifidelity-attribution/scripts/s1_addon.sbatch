#!/bin/bash
#SBATCH --job-name=mf-addon
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
# 灰區唯一加碼：再 200 個重訓（j=200..399）。之後 aggregate 以 --M 400 重判，不得再加碼。
$PYJ $S/retrain.py --cfg $CFG --task $SLURM_ARRAY_TASK_ID --ntasks 8 --j-start 200 --j-end 400 --noise 0
