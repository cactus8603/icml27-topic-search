#!/usr/bin/env bash
# 下載本題需要的模型到共用 hf_home（登入節點，不佔 GPU）。預設只抓 Qwen2.5-7B/14B-Instruct（約 15GB + 30GB）。
# OLMo-3-7B-Instruct（約 15GB，選配）：INCLUDE_OLMO=1。
set -euo pipefail
export TMPDIR=/work/ChihChungHsu/cactus8603/icml/tmp HF_HOME=/work/ChihChungHsu/cactus8603/icml/hf_home
HF=/work/ChihChungHsu/cactus8603/icml/envs/inoc/bin/hf
M="Qwen/Qwen2.5-7B-Instruct Qwen/Qwen2.5-14B-Instruct"
[ "${INCLUDE_OLMO:-0}" = 1 ] && M="$M allenai/Olmo-3-7B-Instruct"
for m in $M; do "$HF" download "$m" --exclude "*.bin" "original/*" ; done
