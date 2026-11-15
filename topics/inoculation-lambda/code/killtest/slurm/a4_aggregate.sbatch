#!/usr/bin/env bash
#SBATCH --job-name=lam_agg
#SBATCH --time=00:20:00
#SBATCH --partition=h200q
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=96G
#SBATCH --output=/work/ChihChungHsu/cactus8603/icml/tmp/%x_%j.out
# 彙整：閘門 1 PASS/FAIL ＋ 對照論文數字的 Spearman（參考）。實際只用 CPU，但依要求掛 1 GPU（可改 --gres 為空，由審查者決定）。
set -euo pipefail
cd -- "${SLURM_SUBMIT_DIR:-$PWD}"
export TMPDIR=/work/ChihChungHsu/cactus8603/icml/tmp HF_HOME=/work/ChihChungHsu/cactus8603/icml/hf_home HF_HUB_OFFLINE=1
PY=/work/ChihChungHsu/cactus8603/icml/envs/inoc/bin/python
run_id=${LAMBDA_PHASE_A_RUN_ID:?Set LAMBDA_PHASE_A_RUN_ID to the shared Phase A run name}
[[ $run_id =~ ^[A-Za-z0-9][A-Za-z0-9_-]*$ ]] || { printf 'Invalid Phase A run name.\n' >&2; exit 2; }
run="/work/ChihChungHsu/cactus8603/icml/runs/inoculation-lambda/phasea_legacy/$run_id"
mkdir -p "$run/logs" "$run/results"
echo "[codex] $(TZ=Asia/Taipei date '+%Y-%m-%d %H:%M') git: $(git rev-parse HEAD 2>/dev/null || echo none) job: ${SLURM_JOB_ID:-local}" | tee -a "$run/logs/RUNLOG.md"
"$PY" -B scripts/aggregate.py --phase a --results-dir "$run/results" --output-dir "$run/results" |
  (set -o noclobber; tee /dev/stderr > "$run/results/aggregate_phaseA.txt")
