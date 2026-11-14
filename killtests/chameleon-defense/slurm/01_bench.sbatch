#!/bin/bash
#SBATCH --job-name=cham-bench
#SBATCH --partition=h200q
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --time=00:25:00
#SBATCH --output=logs/%x-%j.out
# 10 分鐘吞吐實測：harm_test 前 100 例（約 400 列後停）。印 rows/s 與估計總 GPU 小時。
# 實測推估超過 PREREG 估計（0.5 h）的 2 倍 → 重排計畫。
set -euo pipefail
source slurm/env.sh
$PY -I scripts/build_datasets.py --out $RUN/data
$PY -I scripts/extract_acts.py --sets harm_test --limit 100 --bench --batch-size 16 --out $RUN/bench_cache
$PY -I scripts/extract_acts.py --sets harm_test --limit 100 --bench --batch-size 32 --out $RUN/bench_cache
