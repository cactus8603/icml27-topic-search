#!/bin/bash
# 下載 A、B 配對的 4 個模型到 HF_HOME（不佔 GPU）。**共約 16.4 GB**；/work 額度緊時不要執行，等舊題目清完。
# 用法：bash scripts/download_models.sh   （只下載 safetensors 與 json/tokenizer 檔）
source /work/ChihChungHsu/cactus8603/icml27-topic-search/killtests/rl-sparse-distill/slurm/env.sh 2>/dev/null || true
export TMPDIR=/work/ChihChungHsu/cactus8603/icml/tmp HF_HOME=/work/ChihChungHsu/cactus8603/icml/hf_home HF_HUB_OFFLINE=0
PY=/work/ChihChungHsu/cactus8603/icml/envs/vllm033/bin/python
for m in Qwen/Qwen2.5-Math-1.5B sail/Qwen2.5-Math-1.5B-Oat-Zero Qwen/Qwen2.5-1.5B hkust-nlp/Qwen-2.5-1.5B-SimpleRL-Zoo; do
  $PY -I -c "from huggingface_hub import snapshot_download as s;print(s('$m',allow_patterns=['*.json','*.safetensors','*.txt']))" || { echo "下載失敗：$m（磁碟滿？）"; exit 1; }
done
