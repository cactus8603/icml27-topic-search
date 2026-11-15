#!/usr/bin/env bash
# CPU 極小 smoke test：Qwen2.5-0.5B（本機），每設定 2 筆樣本、3 個防禦；不碰 GPU。
set -euo pipefail
cd "$(dirname "$0")/.."
export TMPDIR=/work/ChihChungHsu/cactus8603/icml/tmp HF_HUB_OFFLINE=1 CUDA_VISIBLE_DEVICES=""
PY=/work/ChihChungHsu/cactus8603/icml/envs/inoc/bin/python
M=/work/ChihChungHsu/cactus8603/models/Qwen2.5-0.5B
run_parent=/work/ChihChungHsu/cactus8603/icml/runs/inoculation-lambda/phasea_legacy
mkdir -p "$run_parent"
run=$(mktemp -d "$run_parent/smoke.XXXXXX")
printf 'Smoke output: %s\n' "$run"
"$PY" -B scripts/compute_lambda.py --config configs/grant_sycophantic_qwen7b.yaml --smoke --model "$M" --device cpu --out "$run/grant_sycophantic_qwen7b_smoke"
"$PY" -B scripts/compute_lambda.py --config configs/grant_sycophantic_qwen7b.yaml --smoke --finite_diff --model "$M" --device cpu --out "$run/grant_sycophantic_qwen7b_fd_smoke"
"$PY" -B scripts/compute_lambda.py --config configs/pg_qwen7b_idmedical.yaml --smoke --model "$M" --device cpu --out "$run/pg_qwen7b_idmedical_smoke"
"$PY" -B scripts/aggregate.py --phase a --include-smoke --results-dir "$run" --output-dir "$run" ||
  test -s "$run/aggregate_phaseA_smoke.json"
