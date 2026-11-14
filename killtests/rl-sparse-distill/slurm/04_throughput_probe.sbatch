#!/bin/bash -l
#SBATCH --partition=h200q
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=16
#SBATCH --mem=170G
#SBATCH --output=/work/ChihChungHsu/cactus8603/icml27-topic-search/killtests/rl-sparse-distill/slurm/logs/%x-%j.out
#SBATCH --job-name=rsd-probe
#SBATCH --time=00:40:00
# 用法：sbatch slurm/04_throughput_probe.sbatch A   — 10 分鐘級吞吐實測（vLLM 與 HF rl_replace 各 8 題），超過推估 2 倍就重排
export PAIR=${1:-A}
source /work/ChihChungHsu/cactus8603/icml27-topic-search/killtests/rl-sparse-distill/slurm/env.sh
rm -rf results/probe_$PAIR
/usr/bin/time -v $PY -I scripts/gen_vllm.py --model $RL --out results/probe_$PAIR/vllm --split test --n 128 --limit 64 2>&1 | grep -E "Elapsed|done"
/usr/bin/time -v $PY -I scripts/gen_replace.py --cond rl_replace --base-model $BASE --rl-model $RL --out results/probe_$PAIR/replace --split test --n 128 --limit 8 --rows-per-batch 256 2>&1 | grep -E "Elapsed|batch"
