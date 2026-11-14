#!/bin/bash
#SBATCH --job-name=bg-w2-u833
#SBATCH --partition=h200q
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=40G
#SBATCH --time=04:00:00
#SBATCH --array=0-5
#SBATCH --output=/work/ChihChungHsu/cactus8603/icml27-topic-search/killtests/biased-gen-cfg/logs/%x-%A_%a.out
# 第 2 週：類別 0 專用無條件模型（條文 4）：只用區塊中根==0 的全部樣本（約 817–869 筆），pdrop=1.0，batch 85，seed=pick。
# N 與 K=1 相同（h>2.0 時送出覆寫 --array=0-3）。
source /work/ChihChungHsu/cactus8603/icml27-topic-search/killtests/biased-gen-cfg/scripts/common.sh
$PY $S/train_hier_cond_w2.py --data $HIER_DATA --pick $SLURM_ARRAY_TASK_ID --seed $SLURM_ARRAY_TASK_ID \
    --n $HIER_N --epochs $HIER_EPOCHS --pdrop 1.0 --batch 85 --class-only 0 --out $BG_RUNS/hier_u833/pick$SLURM_ARRAY_TASK_ID
