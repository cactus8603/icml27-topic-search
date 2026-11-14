# 共用環境（被各 sbatch source）。不要寫入 /tmp（共用分割區已滿）。
K=/work/ChihChungHsu/cactus8603/icml27-topic-search/killtests/rl-sparse-distill
ICML=/work/ChihChungHsu/cactus8603/icml
export TMPDIR=$ICML/tmp PIP_CACHE_DIR=$ICML/tmp/pip-cache UV_CACHE_DIR=$ICML/tmp/uv-cache XDG_CACHE_HOME=$ICML/tmp/xdg
export HF_HOME=$ICML/hf_home HF_HUB_OFFLINE=1 TOKENIZERS_PARALLELISM=false
export VLLM_NO_USAGE_STATS=1 DO_NOT_TRACK=1
PY=$ICML/envs/vllm033/bin/python   # vLLM 0.33.0 + torch 2.14 + transformers 5.20.2（既有 env，不再安裝）
cd $K
set -o pipefail
# 由 pairs.json 取模型路徑（本機快取，離線）
resolve() { $PY -I -c "import os;from huggingface_hub import hf_hub_download as h;print(os.path.dirname(h('$1','config.json',local_files_only=True)))"; }
pairval() { $PY -I -c "import json;print(json.load(open('$K/configs/pairs.json'))['pairs']['$1']['$2'])"; }
BASE=$(resolve $(pairval ${PAIR:-A} base)); RL=$(resolve $(pairval ${PAIR:-A} rl))
test -n "$BASE" -a -n "$RL" -a -d "$BASE" -a -d "$RL" || { echo "model resolve failed" >&2; exit 2; }
