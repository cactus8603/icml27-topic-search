# 固定驗證器策略的逐筆推論收據

沿用真實開發診斷的原候選與雙映射 prompt、單 token 允許清單和 vLLM 原生 logprobs。正式推論前須將開發校準後核定的 prompt profile 及每個模型的棄權寬度寫成 `measurement.json` 和 `decision.json`，與選擇理由一起凍結。 [開發診斷](../real_holdout/PREREG_dev_likelihood.md) · [CPU 評估介面](../holdout-evaluation/EVALUATION_INTERFACE.md)

## 先送開發校準

`26565` 只有聚合診斷，沒有可沿用的逐筆連續分數。本次先以 `neutral` 格式重測 250 道 `train` 題的 1,000 個候選。這個格式區分答案末行標記與判決代碼 `A`。同題排序與原文格式的差尚未確立，格式選擇用於降低兩者共用字母的影響。沿用原雙映射、兩個已存模型、vLLM 0.33.0、候選 ID 和最大輸入長度。每個角色送 2,000 個邏輯請求，預檢各有 1,998 個唯一請求。GPU 收據只保存 ID 與語意 log odds 或缺測狀態，不載入金標。

CPU 複核每份分數的作業、模型、版本、提示指紋、ID 與雜湊後，才解碼同一批 `train` 金標。固定查看絕對 log odds 寬度 `0、0.25、0.5、1、2、4、8`，保留各已知真類的全數、觀測數及錯誤，另報棄權、缺測和未知標籤。這些是校準表，留出風險與省費門檻沿用 Round 2。寬度須另由審查者根據開發校準核定。開始正式評估前保持 747 題留出封存。

一張 H200 依序載入兩個模型，沿用現有權重。`26565` 用較多請求實測 1 分 52 秒、Slurm 顯存峰值約 110 GiB，供資源估算。校準申請 GPU 上限 20 分鐘，沿用 4 CPU、48 GiB RAM。輸出是 `icml/runs/fallible-verifier/gsm8k-policy-calibration/<job>/` 下兩份私有分數收據與 `calibration.json`，估低於 1 MiB。

```bash
stage=codex/experiments/fallible-verifier/holdout-scoring
dev_freeze=REVIEWED_FULL_CALIBRATION_COMMIT
bash "$stage/run.sbatch" "$dev_freeze" --check-only
sbatch --parsable --time=00:20:00 "$stage/run.sbatch" "$dev_freeze" calibrate-dev
```

## 凍結後的正式留出

`measurement.json` 須含 `schema`（1）、`profile`（`raw` 或 `neutral`）、固定 `panel_commit` 與 `native_likelihood_commit`。`decision.json` 須含 `schema`（1）以及 `abstain_abs_log_odds` 的 `weak`、`strong` 兩個有限非負值。語意 log odds 為兩字母映射的半差。落在閉區間 `[-bound, +bound]` 標為 `abstain`。任一映射超過共同 token 上限或缺少有效 logprob 時保留候選，標為 `missing`。其餘依符號輸出 0 或 1。兩個角色使用同一標記 profile，並各自核對模型版本與 token 設定。

每個分割獨立送件。推論讀取來源檔的整份 SHA-256，逐列只解碼指定分割，檢查候選和題目 ID。每題的四種解答都保留，角色各產出一份 ID、判決及狀態的私有 JSON。啟動程式從固定 Git 版本執行，逐份紀錄收據雜湊，並隔離模型引擎的原始日誌。政策擬合與風險、訊號及成本比較讀取推論收據，並用另行固定的規格驗收。

```bash
stage=codex/experiments/fallible-verifier/holdout-scoring
shared=/work/ChihChungHsu/cactus8603/icml
"$shared/envs/t2irare/bin/python" -IB "$stage/test_scoring.py"
bash "$stage/run.sbatch" FULL_FREEZE_COMMIT --check-only
```

同一 commit 凍結校準後選定的兩份設定、評估規格及程式。`train` 和 `test` 各寫一份雙角色收據，先後啟動。CPU 評估驗證兩個成功的 Slurm 作業、同一凍結版本及兩邊收據後才解碼留出標籤。政策和條件校準由 `train` 提供，`test` 只用來計分與計算配對區間。

```bash
freeze=REVIEWED_FULL_SCORING_AND_EVALUATION_COMMIT
score=codex/experiments/fallible-verifier/holdout-scoring/run.sbatch
evaluate=codex/experiments/fallible-verifier/holdout-evaluation/run.sbatch
train=$(sbatch --parsable "$score" "$freeze" train)
train=${train%%;*}
test=$(sbatch --parsable --dependency="afterok:$train" "$score" "$freeze" test)
test=${test%%;*}
sbatch --parsable --dependency="afterok:$train:$test" "$evaluate" "$freeze" "$train" "$test"
```
