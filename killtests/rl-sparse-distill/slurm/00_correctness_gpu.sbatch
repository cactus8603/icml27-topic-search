#!/bin/bash -l
#SBATCH --partition=h200q
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=16
#SBATCH --mem=170G
#SBATCH --output=/work/ChihChungHsu/cactus8603/icml27-topic-search/killtests/rl-sparse-distill/slurm/logs/%x-%j.out
#SBATCH --job-name=rsd-correct
#SBATCH --time=01:00:00
# 用法：sbatch slurm/00_correctness_gpu.sbatch A   （真實模型、fp32、GPU 上重跑 smoke 閘門 T1-T5）
export PAIR=${1:-A}
source /work/ChihChungHsu/cactus8603/icml27-topic-search/killtests/rl-sparse-distill/slurm/env.sh
$PY -I tests/smoke_cpu.py $BASE $RL cuda 64 | tee results/correctness_$PAIR.log
