#!/bin/bash
#SBATCH --job-name=mf-crossseed
#SBATCH --partition=h200q
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --time=04:00:00
#SBATCH --array=0-3
#SBATCH --output=/work/ChihChungHsu/cactus8603/icml27-topic-search/killtests/multifidelity-attribution/logs/%x-%A_%a.out
source /work/ChihChungHsu/cactus8603/icml27-topic-search/killtests/multifidelity-attribution/scripts/common.sh
# 階段 1：跨種子 LDS 用——訓練 seed=5（審查後定案 S2：避開 TRAK 集成的 seed 1–4）的 100 次重訓（同樣的移除子集 j=0..99），輸出到 retrain_cs。
$PYJ $S/retrain.py --cfg $CFG --task $SLURM_ARRAY_TASK_ID --ntasks 4 --j-start 0 --j-end 100 --noise 0 --train-seed 5 --outname retrain_cs
