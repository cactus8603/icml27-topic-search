#!/usr/bin/env bash
#SBATCH --job-name=lam_pg
#SBATCH --time=04:00:00
#SBATCH --partition=h200q
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=96G
#SBATCH --output=/work/ChihChungHsu/cactus8603/icml27-topic-search/killtests/inoculation-lambda/logs/%x_%j.out
# 2611.12864 層段：Qwen2.5-7B / 14B 的 Λ（全參數、5 個層段、bf16）。預估 1–2 GPU 小時。
# OLMo-3-7B 為選配（需先 INCLUDE_OLMO=1 下載）；有權重才跑。
set -euo pipefail
cd /work/ChihChungHsu/cactus8603/icml27-topic-search/killtests/inoculation-lambda
export TMPDIR=/work/ChihChungHsu/cactus8603/icml/tmp HF_HOME=/work/ChihChungHsu/cactus8603/icml/hf_home HF_HUB_OFFLINE=1
PY=/work/ChihChungHsu/cactus8603/icml/envs/inoc/bin/python
echo "git: $(git -C ../.. rev-parse HEAD 2>/dev/null || echo none) job: ${SLURM_JOB_ID:-local}" | tee -a logs/RUNLOG.md
for c in pg_qwen7b_idmedical pg_qwen7b_idsports; do
  $PY scripts/compute_lambda.py --config configs/${c}.yaml
done
# 14B 需 8 個 safetensors 分片都在（目前下載未完成，額度滿時暫停）；不完整就跳過並記錄
N14=$(ls /work/ChihChungHsu/cactus8603/icml/hf_home/hub/models--Qwen--Qwen2.5-14B-Instruct/snapshots/*/model-*.safetensors 2>/dev/null | wc -l)
if [ "$N14" -ge 8 ] && ! find /work/ChihChungHsu/cactus8603/icml/hf_home/hub/models--Qwen--Qwen2.5-14B-Instruct -name "*.incomplete" | grep -q .; then
  for c in pg_qwen14b_idmedical pg_qwen14b_idsports; do
    $PY scripts/compute_lambda.py --config configs/${c}.yaml
  done
else
  echo "SKIP 14B：權重不完整" | tee -a logs/RUNLOG.md
fi
if ls /work/ChihChungHsu/cactus8603/icml/hf_home/hub/models--allenai--Olmo-3-7B-Instruct/snapshots/*/config.json >/dev/null 2>&1; then
  $PY scripts/compute_lambda.py --config configs/pg_olmo7b_idmedical.yaml
fi
