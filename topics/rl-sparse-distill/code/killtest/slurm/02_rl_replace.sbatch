#!/bin/bash -l
#SBATCH --partition=h200q
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=16
#SBATCH --mem=170G
#SBATCH --output=/work/ChihChungHsu/cactus8603/icml27-topic-search/killtests/rl-sparse-distill/slurm/logs/%x-%j.out
#SBATCH --job-name=rsd-replace
#SBATCH --time=24:00:00
# 用法：sbatch slurm/02_rl_replace.sbatch PAIR SHARD NSHARD   （HF 雙模型 lock-step；n=128、test）
export PAIR=$1; SH=$2; NS=$3
set -e
source /work/ChihChungHsu/cactus8603/icml27-topic-search/killtests/rl-sparse-distill/slurm/env.sh
$PY -BI scripts/artifacts.py "$PAIR" rl_replace
export RSD_RUN_ID=${RSD_RUN_ID:-${SLURM_ARRAY_JOB_ID:-$SLURM_JOB_ID}}
OUT=results/$PAIR/rl_replace
$PY -I scripts/gen_replace.py --cond rl_replace --base-model $BASE --rl-model $RL --out $OUT --split test --n 128 --rows-per-batch 256 --shard $SH/$NS
$PY -I scripts/grade.py "$OUT" --pair "$PAIR" --condition rl_replace --n-shards "$NS" --workers 16
