#!/bin/bash
#SBATCH --job-name=cr-headroom
#SBATCH --partition=h200q
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=56G
#SBATCH --time=03:00:00
#SBATCH --output=/work/ChihChungHsu/cactus8603/icml27-topic-search/killtests/compositional-retrieval/logs/%x-%j.out
set -euo pipefail

source /work/ChihChungHsu/cactus8603/icml27-topic-search/killtests/compositional-retrieval/slurm/env.sh
export PYTHONDONTWRITEBYTECODE=1
export OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1
PY_HEADROOM=/work/ChihChungHsu/cactus8603/icml/envs/vllm033/bin/python
reranker=${HEADROOM_RERANKER:-core_reranker}
case "$reranker" in
    core_reranker|vlm_fallback) ;;
    *) printf 'Invalid HEADROOM_RERANKER %s\n' "$reranker" >&2; exit 2 ;;
esac

attempt_id=${SLURM_JOB_ID:?正式補量需由 Slurm 執行}
out_dir=results/headroom/$attempt_id
mkdir -p results/headroom
mkdir "$out_dir" || { printf '補量輸出目錄已存在：%s\n' "$out_dir" >&2; exit 2; }
output=(--out-dir "$out_dir" --attempt-id "$attempt_id")
queries=$out_dir/e2_queries_main.json
"$PY_HEADROOM" -I scripts/headroom.py --preflight --report-availability --reranker "$reranker"
"$PY_HEADROOM" -I scripts/test_headroom.py

models=(cs_clip core_embed siglip2)
if "$PY_HEADROOM" -c 'import json; m = json.load(open("configs/params_headroom.json")); raise SystemExit(not m["models"]["clic"]["revision"])'; then
    models+=(clic)
fi
models+=("$reranker")
for model in "${models[@]}"; do
    if ! "$PY_HEADROOM" -I scripts/headroom.py --model "$model" "${output[@]}"; then
        printf '%s 未完整評估；保留已寫入子集，見本作業日誌\n' "$model" >&2
    fi
done
"$PY_HEADROOM" -I scripts/headroom.py --aggregate --reranker "$reranker" "${output[@]}"

if ! "$PY_HEADROOM" -I scripts/headroom.py --e2-preflight; then
    printf 'E2 未能測試，缺描述性輸入\n' >&2
    exit 0
fi
if ! "$PY_HEADROOM" -I scripts/e2_build_queries.py --seed-key seed --out "$queries"; then
    printf 'E2 未能測試，查詢生成失敗\n' >&2
    exit 0
fi
for model in "${models[@]}"; do
    if ! "$PY_HEADROOM" -I scripts/headroom.py --e2-model "$model" --e2-queries "$queries" "${output[@]}"; then
        printf '%s E2 未能測試，見本作業日誌\n' "$model" >&2
    fi
done
