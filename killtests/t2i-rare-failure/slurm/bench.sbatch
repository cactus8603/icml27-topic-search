#!/bin/bash
#SBATCH --job-name=t2ir-bench
#SBATCH --partition=h200q
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=16
#SBATCH --mem=64G
#SBATCH --time=00:30:00
#SBATCH --output=logs/%x-%j.out
# 吞吐實測 + 重現/雜訊底線（約 10 分鐘 GPU）：sd14 + E1。
#  (1) seeds 0-512 batch 32；(2) 同 seeds batch 64（重現檢查，比對事件翻轉）；(3) seeds 512-1024 batch 64（較大 batch 吞吐）。
# 用實測張/秒更新 GPU 小時；若 < 估計的 50%（4.5 張/秒）需重排整個計畫（PREREG 成本規則）。
set -euo pipefail
source slurm/env.sh
$PY scripts/generate.py --model sd14 --prompts E1 --seed-start 0 --seed-end 512 --chunk 512 --batch-size 32 --out results_bench
$PY scripts/generate.py --model sd14 --prompts E1 --seed-start 0 --seed-end 512 --chunk 512 --batch-size 64 --out results_bench2
$PY scripts/generate.py --model sd14 --prompts E1 --seed-start 512 --seed-end 1024 --chunk 512 --batch-size 64 --out results_bench2
$PY scripts/compare_runs.py results_bench results_bench2 || true
