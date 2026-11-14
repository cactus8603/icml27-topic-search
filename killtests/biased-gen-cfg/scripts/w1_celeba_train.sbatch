#!/bin/bash
#SBATCH --job-name=bg-celeba-train
#SBATCH --partition=h200q
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=40G
#SBATCH --time=06:00:00
#SBATCH --array=0-5
#SBATCH --output=/work/ChihChungHsu/cactus8603/icml27-topic-search/killtests/biased-gen-cfg/logs/%x-%A_%a.out
# (a) CelebA 重現：6 個模型各一個 task（index = SLURM_ARRAY_TASK_ID，seed = index）。可續跑（重送同一指令即可）。
source /work/ChihChungHsu/cactus8603/icml27-topic-search/killtests/biased-gen-cfg/scripts/common.sh
$PY $S/train_celeba.py -i $SLURM_ARRAY_TASK_ID -n $CELEBA_N -W $CELEBA_WIDTH -B $CELEBA_BATCH -LR $CELEBA_LR \
    --epochs $CELEBA_EPOCHS --data-root $CELEBA_DATA_ROOT --save-root $BG_RUNS/celeba
