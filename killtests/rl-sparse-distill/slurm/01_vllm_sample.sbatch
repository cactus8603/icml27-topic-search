#!/bin/bash -l
#SBATCH --partition=h200q
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=16
#SBATCH --mem=170G
#SBATCH --output=/work/ChihChungHsu/cactus8603/icml27-topic-search/killtests/rl-sparse-distill/slurm/logs/%x-%j.out
#SBATCH --job-name=rsd-vllm
#SBATCH --time=06:00:00
# 用法：sbatch slurm/01_vllm_sample.sbatch PAIR COND SHARD NSHARD
#   COND = base | rl   （n=128、test；閘門 0 與判準 1 共用同一份）
#   COND = base_greedy | rl_greedy （n=1、temperature 0、split all；閘門 0 重現用）
export PAIR=$1; COND=$2; SH=${3:-0}; NS=${4:-1}
set -e
source /work/ChihChungHsu/cactus8603/icml27-topic-search/killtests/rl-sparse-distill/slurm/env.sh
case $COND in base*) M=$BASE;; *) M=$RL;; esac
OUT=results/$PAIR/$COND
case $COND in
  *_greedy) $PY -I scripts/gen_vllm.py --model $M --out $OUT --split all --n 1 --temperature 0 --top-p 1 --shard $SH/$NS ;;
  *)        $PY -I scripts/gen_vllm.py --model $M --out $OUT --split test --n 128 --shard $SH/$NS ;;
esac
# 全部分片都到齊才判分（最後一個完成的 job 負責）；raw 檔數 = NS 才執行
if [ $(ls $OUT/raw_*of$NS.jsonl.gz 2>/dev/null | wc -l) -eq $NS ]; then $PY -I scripts/grade.py $OUT --workers 16; fi
