#!/usr/bin/env bash
# 取得 2611.12864 的 EM 資料（bad_medical_advice、extreme_sports；上游加密壓縮檔，密碼為上游公開文件所載）
# 並複製 repo 內附的 4 個 OOD 資料集。登入節點執行，不佔 GPU。步驟取自該 repo 的 scripts/download_data.py（已閱讀）。
set -euo pipefail
export TMPDIR=/work/ChihChungHsu/cactus8603/icml/tmp PIP_CACHE_DIR=/work/ChihChungHsu/cactus8603/icml/tmp/pip-cache UV_CACHE_DIR=/work/ChihChungHsu/cactus8603/icml/tmp/uv-cache
ROOT=$(cd "$(dirname "$0")/.." && pwd)
PY=/work/ChihChungHsu/cactus8603/icml/envs/inoc/bin
EXT=/work/ChihChungHsu/cactus8603/icml/ext/persona-generalization
mkdir -p "$ROOT/data/em" "$ROOT/data/_tmp"
cd "$ROOT/data/_tmp"
curl -sL -m 120 -o training_datasets.zip.enc \
  https://raw.githubusercontent.com/jacksonwebster/emergent-misalignment/main/training_data/training_datasets.zip.enc
"$PY/easy-dataset-share" magic-unprotect-dir training_datasets.zip.enc -p model-organisms-em-datasets --remove-canaries
for f in bad_medical_advice.jsonl extreme_sports.jsonl; do
  cp "$(find . -name "$f" | head -1)" "$ROOT/data/em/$f"
done
cp "$EXT"/data/em/bad_{cooking,cybersec,diy,social}_advice.jsonl "$ROOT/data/em/"
cd "$ROOT"; rm -rf data/_tmp
wc -l data/em/*.jsonl
