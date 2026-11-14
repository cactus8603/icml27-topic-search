# Attention 寫入訊號獨立種子驗證

[codex] 2026-11-15 台北時間。校準 26446 的選擇規則與健康檢查由未參與實作的代理複核。依 [已凍結的驗證計畫](PREREG.md#獨立驗證計畫) 選定 4×，校準收據 SHA-256 為 `28676645e5e7722c4f07ebb084bcb2cfab2492590310bde1b8b9c060f0d511df`。

固定 12 個保留 seeds、三個強基線、各自的正常參照和 4× 壓力臂。資料、切分、訓練、品質守門、互斥探針窗口、局部干預及整 seed 留出 ridge logistic，沿用凍結的校準程式與設定。資料集仍是同一份 OpenWebText shard。這是新 seed 對照，資料集外泛化需另外驗證。

必須完成全部 36 對，且每對通過品質守門。損傷定義沿用更新後減更新前 NLL ≥0.05 nats，限 ramp 結束後。至少 30 次損傷及 6 個可算 AUROC 的留出 seed 才判讀。等權平均 AUROC 增量至少 0.03 支持人工壓力訊號。記錄每 seed 的對照與增量、描述性區間及局部改善診斷。沒有涵蓋、缺檔或品質崩退時記「不可判」。局部回退用來界定觀察步的損傷，不參與門檻。實際 optimizer 與等品質訓練速度需另外驗證。

一張 H200，8 CPU、32 GiB 系統 RAM，最長 1 小時。共 36 個 prefix ×（100+2×400）步，約 32,400 步、1.33 億訓練 tokens，另有量測。以校準實測粗估 20–35 分鐘。Corpus 與逐步小型記錄合計估計 <0.1 GiB，無模型 checkpoint 落盤。使用凍結檔案執行，機械彙總為 `results/validation-<job-id>/validation.json`，再由獨立代理複核。

```sh
bash codex/experiments/gate-optimizer/round2/validate.sbatch FULL_COMMIT_SHA --check-only
sbatch --parsable codex/experiments/gate-optimizer/round2/validate.sbatch FULL_COMMIT_SHA
```

- [codex] 已讀 `results/validation-26502/validation.json`，2026-11-15 07:13。凍結版 `4d87430`，獨立複核與整合核對。判準沿用原登記。
