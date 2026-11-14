#!/usr/bin/env bash
#SBATCH --job-name=lam_spanish_diag
#SBATCH --time=01:30:00
#SBATCH --partition=h200q
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=96G
#SBATCH --output=/work/ChihChungHsu/cactus8603/icml27-topic-search/killtests/inoculation-lambda/logs/%x_%j.out
set -euo pipefail
cd /work/ChihChungHsu/cactus8603/icml27-topic-search/killtests/inoculation-lambda
export TMPDIR=/work/ChihChungHsu/cactus8603/icml/tmp HF_HOME=/work/ChihChungHsu/cactus8603/icml/hf_home HF_HUB_OFFLINE=1
export PYTHONUNBUFFERED=1
PY=/work/ChihChungHsu/cactus8603/icml/envs/inoc/bin/python
FROZEN_COMMIT=df7cdb280888512e9006f61f642ad5caf479603c
FROZEN_DIR=$(mktemp -d "$TMPDIR/lambda-spanish-${SLURM_JOB_ID}.XXXXXX")
trap 'rm -r -- "$FROZEN_DIR"' EXIT
for file in scripts/compute_lambda.py configs/criteria.yaml configs/grant_spanish_qwen7b.yaml; do
  mkdir -p "$FROZEN_DIR/${file%/*}"
  git show "$FROZEN_COMMIT:killtests/inoculation-lambda/$file" > "$FROZEN_DIR/$file"
done
echo "job=${SLURM_JOB_ID} frozen_input=$FROZEN_COMMIT host=$(hostname) date=$(date -Is)"
sha256sum "$FROZEN_DIR"/scripts/compute_lambda.py "$FROZEN_DIR"/configs/*.yaml
sha256sum -c <<< "${DIAGNOSTIC_PY_SHA256:?Diagnostic source hash required}  scripts/diagnose_spanish_fd.py"
OFFICIAL_RAW=results/_failed_26261/grant_spanish_qwen7b_fd/lambda_raw.json
sha256sum -c <<< "${OFFICIAL_RAW_SHA256:?Official output hash required}  $OFFICIAL_RAW"
"$PY" -B scripts/diagnose_spanish_fd.py --frozen-root "$FROZEN_DIR" \
  --official-raw "$OFFICIAL_RAW" \
  --out "results/spanish_step_diagnostic_${SLURM_JOB_ID}"
