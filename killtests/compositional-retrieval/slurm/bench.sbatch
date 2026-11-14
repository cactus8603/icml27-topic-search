#!/bin/bash
#SBATCH --job-name=cr-bench
#SBATCH --partition=h200q
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=32G
#SBATCH --time=00:20:00
#SBATCH --output=/work/ChihChungHsu/cactus8603/icml27-topic-search/killtests/compositional-retrieval/logs/%x-%j.out
# 10 分鐘吞吐實測：兩個模型 × (HT 影像、COCO 影像)，每項最多量 60 秒，之後推估全部 E0–E2 的 GPU 小時。
# 前置：scripts/download_assets.py 已在 login 節點跑完（COCO val2017 需已解壓）。
set -euo pipefail
source slurm/env.sh
$PY scripts/throughput.py --device cuda --seconds 120 --out results/throughput.json
