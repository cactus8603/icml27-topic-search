#!/bin/bash
#SBATCH --job-name=cr-stage1
#SBATCH --partition=h200q
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=48G
#SBATCH --time=01:00:00
#SBATCH --output=/work/ChihChungHsu/cactus8603/icml27-topic-search/killtests/compositional-retrieval/logs/%x-%j.out
# 階段 1：閘門 G1（單元測試、與官方腳本交叉核對）、G2（重現 58.2/50.4/42.2 與 CLIP 基線）、G3（雜訊底線、重跑一致）
# 同時產出 E1 所需的 embedding 與分析。全部在同一次 GPU 配置內做完（約 10–20 分鐘）。
set -euo pipefail
source slurm/env.sh
$PY scripts/unit_tests.py                                   # G1：已知答案
for m in clip_b32 cs_clip; do
  $PY scripts/extract_ht.py --model $m --device cuda --batch-size 64             # 主跑
  $PY scripts/extract_ht.py --model $m --device cuda --batch-size 17 --tag _rerun # G3：不同 batch 的重跑
  $PY scripts/analyze_ht.py --model $m
  $PY scripts/crosscheck_repo.py --model $m --subset coco_qwen --npz results/ht/${m}_coco_qwen.npz --device cuda   # G1：官方腳本交叉核對
done
$PY scripts/crosscheck_repo.py --merge
$PY scripts/rerun_compare.py --tag-a "" --tag-b _rerun
$PY scripts/aggregate.py        # 此時 E2 尚未跑：P2 顯示「未能測試（資料不足）」，閘門結果已可判讀
