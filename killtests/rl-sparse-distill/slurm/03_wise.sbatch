#!/bin/bash -l
#SBATCH --partition=h200q
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=16
#SBATCH --mem=170G
#SBATCH --output=/work/ChihChungHsu/cactus8603/icml27-topic-search/killtests/rl-sparse-distill/slurm/logs/%x-%j.out
#SBATCH --job-name=rsd-wise
#SBATCH --time=06:00:00
# 用法：sbatch slurm/03_wise.sbatch PAIR LAMBDA   （內插模型存節點暫存，跑完刪除；n=128、test）
export PAIR=$1; LAM=$2
set -e
source /work/ChihChungHsu/cactus8603/icml27-topic-search/killtests/rl-sparse-distill/slurm/env.sh
W=$TMPDIR/wise-$PAIR-$LAM-$SLURM_JOB_ID
$PY -I scripts/make_wise.py --base $BASE --rl $RL --lam $LAM --out $W
OUT=results/$PAIR/wise_$LAM
$PY -I scripts/gen_vllm.py --model $W --out $OUT --split test --n 128
rm -rf $W
$PY -I scripts/grade.py $OUT --workers 16
