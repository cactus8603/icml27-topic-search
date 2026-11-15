#!/usr/bin/env bash
#SBATCH --job-name=lam_pg
#SBATCH --time=04:00:00
#SBATCH --partition=h200q
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=96G
#SBATCH --output=/work/ChihChungHsu/cactus8603/icml/tmp/%x_%j.out
# 2611.12864 層段：Qwen2.5-7B / 14B 的 Λ（全參數、5 個層段、bf16）。預估 1–2 GPU 小時。
# OLMo-3-7B 為選配（需先 INCLUDE_OLMO=1 下載）；有權重才跑。
set -euo pipefail
cd -- "${SLURM_SUBMIT_DIR:-$PWD}"
export TMPDIR=/work/ChihChungHsu/cactus8603/icml/tmp HF_HOME=/work/ChihChungHsu/cactus8603/icml/hf_home HF_HUB_OFFLINE=1
PY=/work/ChihChungHsu/cactus8603/icml/envs/inoc/bin/python
run_id=${LAMBDA_PHASE_A_RUN_ID:?Set LAMBDA_PHASE_A_RUN_ID to the shared Phase A run name}
[[ $run_id =~ ^[A-Za-z0-9][A-Za-z0-9_-]*$ ]] || { printf 'Invalid Phase A run name.\n' >&2; exit 2; }
run="/work/ChihChungHsu/cactus8603/icml/runs/inoculation-lambda/phasea_legacy/$run_id"
mkdir -p "$run/logs" "$run/results"
echo "[codex] $(TZ=Asia/Taipei date '+%Y-%m-%d %H:%M') git: $(git rev-parse HEAD 2>/dev/null || echo none) job: ${SLURM_JOB_ID:-local}" | tee -a "$run/logs/RUNLOG.md"
for c in pg_qwen7b_idmedical pg_qwen7b_idsports; do
  "$PY" -B scripts/compute_lambda.py --config configs/${c}.yaml --out "$run/results/${c}"
done
# 按 Hugging Face 實際解析的離線快照核對索引指向的所有權重。
if "$PY" scripts/check_cached_model.py Qwen/Qwen2.5-14B-Instruct; then
  for c in pg_qwen14b_idmedical pg_qwen14b_idsports; do
    "$PY" -B scripts/compute_lambda.py --config configs/${c}.yaml --out "$run/results/${c}"
  done
else
  echo "[codex] $(TZ=Asia/Taipei date '+%Y-%m-%d %H:%M') SKIP 14B：權重不完整" | tee -a "$run/logs/RUNLOG.md"
fi
if "$PY" scripts/check_cached_model.py allenai/Olmo-3-7B-Instruct; then
  "$PY" -B scripts/compute_lambda.py --config configs/pg_olmo7b_idmedical.yaml --out "$run/results/pg_olmo7b_idmedical"
fi
