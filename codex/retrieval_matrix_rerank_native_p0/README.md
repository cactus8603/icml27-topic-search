# [codex] 原生重排器 MCQ P0 候選模組

此目錄準備 Qwen-Swift、Qwen-native、Jina-native、通用 Qwen True/False 四類互不冒用資格的協定。來源、權重、exact request、原分組、targets 與缺口見 `SOURCE_AUDIT.md`／`protocols.json`。只有可驗 targets 的 variant 才能進正式 P0；CPU source/template PASS 不是模型資格。

沿用 frozen CORE 的原始資料配對與 `backends.py`，不修改其來源。`adapters.py` 僅驗 GPU／BF16權重及 Qwen 原 vision helper 沒把圖片換成 NULL。`runner.py` 分 smoke、score、analyze；storage32題不改原 forward batches，每個原呼叫先保存 immutable raw transaction，恢復時先讀原分數。float64 容器保留原 Python scalar，原 list／scalar／TrueFalse分支 logprobs 另存，未重新縮放。

`qualification.py review` 由不同 producer 的 reviewer 從原 raw／arrays 重新核全部分母及 strict comparisons；`qualify` 再綁相同原檔與完整 model／variant digest。三個 benchmark 皆保留，缺 matching targets 的協定不能靠接近別種表數變成 PASS。Jina image-choice 與 CORE/Qwen text-choice 各有明確名稱，另一軸僅作 diagnostic。

```bash
HF_HOME="$PWD/hf_home" HF_MODULES_CACHE="$PWD/hf_home/modules" CUDA_VISIBLE_DEVICES='' \
  dl/retrieval-matrix-rerank/env/bin/python -B -m unittest discover \
  -s codex/retrieval_matrix_rerank_native_p0 -p 'test_*.py'
CUDA_VISIBLE_DEVICES='' dl/retrieval-matrix-rerank/env/bin/python -B \
  codex/retrieval_matrix_rerank_native_p0/build_plan.py --out NEW_REVIEW_PLAN.json
```

builder 只產 REVIEW_REQUIRED，所有模型協定預設未核。獨立 reviewer 確認後，才可批准該 model 的 verified_protocol 並重算 model_spec_sha256，將 plan 設 FROZEN／reviewed=true，commit 所有執行來源與 plan。這個 source review 不代替實際官方 MCQ 重現。runtime 還要求 root 中央 dispatcher 的 owner/task、恰一 assigned GPU、原套件與模組路徑／SHA；不自行找卡、不 fallback。

GPU 由 root 日後使用 `bash run.sh COMMIT PLAN VARIANT OUT`；本次未送件或執行。score 輸出和 synthetic smoke 分開；效果分析另用 `runner.py --stage analyze`。實際主結果的讀取應另凍結 analysis PREREG 記錄，保留 producer PREREG 原 bytes。原生缺 targets 的候選若要做探索，需另立已授權的計畫；本 P0 gate 不放行。
