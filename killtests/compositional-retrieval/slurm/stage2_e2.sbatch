#!/bin/bash
#SBATCH --job-name=cr-stage2
#SBATCH --partition=h200q
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=48G
#SBATCH --time=01:00:00
#SBATCH --output=/work/ChihChungHsu/cactus8603/icml27-topic-search/killtests/compositional-retrieval/logs/%x-%j.out
# 階段 2：E2 條件數稀釋（COCO val2017，5000 張影像 × 2 模型）。建議在階段 1 的閘門通過後才送出。
# 查詢集建立（CPU）與 embedding（GPU）在同一工作中；分析與最終 aggregate 也在此完成。
set -euo pipefail
source slurm/env.sh
$PY scripts/e2_build_queries.py --seed-key seed       --out results/e2/queries_main.json
$PY scripts/e2_build_queries.py --seed-key addon_seed --out results/e2/queries_addon.json
for m in clip_b32 cs_clip; do
  $PY scripts/e2_extract.py --model $m --device cuda
  $PY scripts/e2_analyze.py --model $m --out results/e2/analysis_${m}_main.json
  $PY scripts/e2_analyze.py --model $m --with-addon --out results/e2/analysis_${m}_main+addon.json
done
$PY scripts/aggregate.py
