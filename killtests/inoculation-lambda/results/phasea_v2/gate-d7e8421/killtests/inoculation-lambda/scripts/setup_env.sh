#!/usr/bin/env bash
# 建立 /work/ChihChungHsu/cactus8603/icml/envs/inoc（uv venv，不動既有環境）。登入節點執行，不佔 GPU。
# 系統 python3 沒有 ensurepip，故用 uv（既有 envs 亦為 uv venv）。
set -euo pipefail
export TMPDIR=/work/ChihChungHsu/cactus8603/icml/tmp PIP_CACHE_DIR=/work/ChihChungHsu/cactus8603/icml/tmp/pip-cache UV_CACHE_DIR=/work/ChihChungHsu/cactus8603/icml/tmp/uv-cache
UV=${UV:-/home/khyeh/.local/bin/uv}
ENV=/work/ChihChungHsu/cactus8603/icml/envs/inoc
"$UV" venv --python /usr/bin/python3 "$ENV"
"$UV" pip install --python "$ENV/bin/python" "torch==2.8.0" "transformers==4.57.6" "accelerate" "numpy" "scipy" "pyyaml" "safetensors" "huggingface_hub" "easy-dataset-share==0.1.0"
"$ENV/bin/python" -c "import torch,transformers,scipy;print(torch.__version__,transformers.__version__,scipy.__version__)"
