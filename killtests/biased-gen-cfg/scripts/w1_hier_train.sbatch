#!/bin/bash
#SBATCH --job-name=bg-hier-train
#SBATCH --partition=h200q
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=40G
#SBATCH --time=10:00:00
#SBATCH --array=0-2
#SBATCH --output=/work/ChihChungHsu/cactus8603/icml27-topic-search/killtests/biased-gen-cfg/logs/%x-%A_%a.out
# (b) 階層資料 + 根符號條件 + condition dropout：3 份不相交資料（pick 0,1,2），seed = pick。
source /work/ChihChungHsu/cactus8603/icml27-topic-search/killtests/biased-gen-cfg/scripts/common.sh
$PY $S/train_hier_cond.py --data $HIER_DATA --pick $SLURM_ARRAY_TASK_ID --seed $SLURM_ARRAY_TASK_ID \
    --n $HIER_N --epochs $HIER_EPOCHS --pdrop $HIER_PDROP --out $BG_RUNS/hier/pick$SLURM_ARRAY_TASK_ID
