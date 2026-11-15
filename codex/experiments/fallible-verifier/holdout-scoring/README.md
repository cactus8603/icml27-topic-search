# 固定驗證器策略的逐筆推論收據

沿用真實開發診斷的原候選與雙映射 prompt、單 token 允許清單和 vLLM 原生 logprobs。申請 GPU 前須將開發診斷確認的 prompt profile 及每個模型的棄權寬度寫成 `measurement.json` 和 `decision.json`，與選擇理由一起凍結。現階段預檢只解碼 250 題的開發 prompt 並驗證每個角色的資產，尚未打開正式留出候選或讀取任何標籤。 [開發診斷](../real_holdout/PREREG_dev_likelihood.md) · [CPU 評估介面](../holdout-evaluation/EVALUATION_INTERFACE.md)

`measurement.json` 須含 `schema`（1）、`profile`（`raw` 或 `neutral`）、固定 `panel_commit` 與 `native_likelihood_commit`。`decision.json` 須含 `schema`（1）以及 `abstain_abs_log_odds` 的 `weak`、`strong` 兩個有限非負值。語意 log odds 為兩字母映射的半差。落在閉區間 `[-bound, +bound]` 標為 `abstain`。任一映射超過共同 token 上限或缺少有效 logprob 時保留候選，標為 `missing`。其餘依符號輸出 0 或 1。兩個角色使用同一標記 profile，並各自核對模型版本與 token 設定。

每個分割獨立送件。推論讀取來源檔的整份 SHA-256，逐列只解碼指定分割，檢查候選和題目 ID。每題的四種解答都保留，角色各產出一份 ID、判決及狀態的私有 JSON。啟動程式從固定 Git 版本執行，逐份紀錄收據雜湊，並隔離模型引擎的原始日誌。政策擬合與風險、訊號及成本比較讀取推論收據，並用另行固定的規格驗收。

```bash
stage=codex/experiments/fallible-verifier/holdout-scoring
shared=/work/ChihChungHsu/cactus8603/icml
"$shared/envs/t2irare/bin/python" -IB "$stage/test_scoring.py"
bash "$stage/run.sbatch" FULL_FREEZE_COMMIT --check-only
```
