#!/bin/bash
#SBATCH --job-name=bg-w2-k1
#SBATCH --partition=h200q
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=40G
#SBATCH --time=10:00:00
#SBATCH --array=0-5
#SBATCH --output=/work/ChihChungHsu/cactus8603/icml27-topic-search/killtests/biased-gen-cfg/logs/%x-%A_%a.out
# 第 2 週：K=1 常數條件負控制（條文 3）。同資料（區塊 pick）、同 seed、pdrop=0.1，條件標籤固定 0。
# h（第 1 週平均 GPU 小時）> 2.0 時只跑 4 個：送出時覆寫 --array=0-3。
source /work/ChihChungHsu/cactus8603/icml27-topic-search/killtests/biased-gen-cfg/scripts/common.sh
W2_RUNS_ROOT=${W2_RUNS_ROOT:-$BG_RUNS}
$PY $S/train_hier_cond_w2.py --data $HIER_DATA --pick $SLURM_ARRAY_TASK_ID --seed $SLURM_ARRAY_TASK_ID \
    --n $HIER_N --epochs $HIER_EPOCHS --pdrop $HIER_PDROP --const-label 0 --out "$W2_RUNS_ROOT/hier_k1/pick$SLURM_ARRAY_TASK_ID"
