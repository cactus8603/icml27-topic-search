#!/bin/bash -l
#SBATCH --partition=h200q
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=16
#SBATCH --mem=170G
#SBATCH --output=/work/ChihChungHsu/cactus8603/icml27-topic-search/killtests/rl-sparse-distill/slurm/logs/%x-%j.out
#SBATCH --job-name=rsd-rlhf
#SBATCH --time=24:00:00
# 用法：sbatch slurm/05_rl_hf.sbatch PAIR SHARD NSHARD
# 判準 (i) 的比較對象：HF 引擎的 RL 取樣（gen_replace.py --cond rl），與 02 的 rl_replace 同引擎、同 split、n=128、同 seed 規則（--seed 預設 1234，分片規則相同）。
# 輸出 results/<P>/rl_hf；aggregate 的 (i) 讀這份。每配對預估 +約 10 GPU 小時（只載 RL 一個模型）。
# env.sh 已設 pipefail；此腳本不用 tee，Python 失敗即回傳非 0。
export PAIR=$1; SH=$2; NS=$3
set -e
source /work/ChihChungHsu/cactus8603/icml27-topic-search/killtests/rl-sparse-distill/slurm/env.sh
OUT=results/$PAIR/rl_hf
$PY -I scripts/gen_replace.py --cond rl --base-model $BASE --rl-model $RL --out $OUT --split test --n 128 --rows-per-batch 256 --shard $SH/$NS || exit 1
if [ $(ls $OUT/raw_*of$NS.jsonl.gz 2>/dev/null | wc -l) -eq $NS ]; then $PY -I scripts/grade.py $OUT --workers 16 || exit 1; fi
