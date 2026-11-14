#!/bin/bash
#SBATCH --job-name=cr-probe2
#SBATCH --partition=h200q
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=48G
#SBATCH --time=03:00:00
#SBATCH --output=/work/ChihChungHsu/cactus8603/icml27-topic-search/killtests/compositional-retrieval/logs/%x-%j.out
# 第二探針（停損後；同一批資料已看過；凍結後只跑一次）。GPU 只用於：正控制句文字 embedding、COCO val2017 5000 張影像 embedding。
# 其後 CPU 分析與機械判定在同一工作內完成。預估 < 10 分鐘 GPU。
# 必須在協調者 commit 凍結 PREREG_probe2.md 等檔之後才送出。
set -euo pipefail
source slurm/env.sh
mkdir -p results/probe2
# 凍結檢查：相關檔案須已 commit 且無未提交修改
for f in PREREG_probe2.md configs/params_probe2.json scripts/probe2_condpmi.py scripts/probe2_embed.py scripts/aggregate_probe2.py scripts/unit_tests_probe2.py slurm/probe2.sbatch; do
  git ls-files --error-unmatch "$f" >/dev/null
  git diff --quiet HEAD -- "$f" || { echo "未凍結：$f 有未 commit 的修改"; exit 3; }
done
$PY scripts/unit_tests_probe2.py
for m in clip_b32 cs_clip; do
  $PY scripts/probe2_embed.py --model $m --device cuda
  $PY scripts/probe2_condpmi.py --model $m
done
$PY scripts/aggregate_probe2.py
