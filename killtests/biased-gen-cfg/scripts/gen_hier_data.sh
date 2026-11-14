#!/bin/bash
# 在登入節點／CPU 上產生階層資料（q=6, l=4, sigma=1, q_eff=4, seed 0；文法 M 與原作 1M 樣本檔相同）。
# 40000 筆 = 8 個 5000 筆區塊：0-2 訓練（週 1）、3-5 週 2 追加、7 = 共同測試。
source /work/ChihChungHsu/cactus8603/icml27-topic-search/killtests/biased-gen-cfg/scripts/common.sh
mkdir -p "$(dirname "$HIER_DATA")"
cd $BG_EXT
$PY modules/gen_filtered_hierarchical_data_wforbidden.py --n-trials 40000 --workers ${WORKERS:-16} --output "$HIER_DATA"
