#!/bin/bash
#SBATCH --job-name=bg-w2-k6
#SBATCH --partition=h200q
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=40G
#SBATCH --time=10:00:00
#SBATCH --array=3-5
#SBATCH --output=/work/ChihChungHsu/cactus8603/icml27-topic-search/killtests/biased-gen-cfg/logs/%x-%A_%a.out
# 第 2 週：K=6 補 pick 3–5（與第 1 週完全相同的 train_hier_cond.py 與設定；seed = pick）。不要送出，除非使用者核可。
source /work/ChihChungHsu/cactus8603/icml27-topic-search/killtests/biased-gen-cfg/scripts/common.sh
W2_RUNS_ROOT=${W2_RUNS_ROOT:-$BG_RUNS}
$PY $S/train_hier_cond.py --data $HIER_DATA --pick $SLURM_ARRAY_TASK_ID --seed $SLURM_ARRAY_TASK_ID \
    --n $HIER_N --epochs $HIER_EPOCHS --pdrop $HIER_PDROP --out "$W2_RUNS_ROOT/hier/pick$SLURM_ARRAY_TASK_ID"
