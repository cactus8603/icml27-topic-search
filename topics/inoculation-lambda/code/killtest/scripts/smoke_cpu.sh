#!/usr/bin/env bash
# CPU 極小 smoke test：Qwen2.5-0.5B（本機），每設定 2 筆樣本、3 個防禦；不碰 GPU。
set -euo pipefail
cd "$(dirname "$0")/.."
export TMPDIR=/work/ChihChungHsu/cactus8603/icml/tmp HF_HUB_OFFLINE=1 CUDA_VISIBLE_DEVICES=""
PY=/work/ChihChungHsu/cactus8603/icml/envs/inoc/bin/python
M=/work/ChihChungHsu/cactus8603/models/Qwen2.5-0.5B
$PY scripts/compute_lambda.py --config configs/grant_sycophantic_qwen7b.yaml --smoke --model $M --device cpu
$PY scripts/compute_lambda.py --config configs/grant_sycophantic_qwen7b.yaml --smoke --finite_diff --model $M --device cpu
$PY scripts/compute_lambda.py --config configs/pg_qwen7b_idmedical.yaml --smoke --model $M --device cpu
$PY scripts/aggregate.py --phase a --include-smoke || true
