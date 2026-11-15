# 正常學習率的單步損傷與寫入訊號

第三輪把 1× 分支當作訓練品質參照，未測該分支單步損傷。本次重跑同一 48 個種子的三個 optimizer，用互斥的訊號與結果 token 探針測正常排程。人工壓力原判讀沿用。 [第三輪](../gate-optimizer/round3/PREREG.md)

- 固定原 OpenWebText shard、模型、資料順序、base LR、100 步 warmup prefix 及 400 步名義 cosine continuation。沿用原 1× 分支規則，損傷資料只取第 30 至 400 步、每 10 步一筆，與 4× 壓力計分位置及 token 配對。共 144 個分支、5,472 筆探針。四個原分段及 CPU 完結後的雜湊已固定。新跑與原分支同裝置與軟體；逐步 LR 完全相同、原定所有 dev 評估差絕對值最多 0.01 nats，才通過探針未實質變更軌跡的守門。
- 單步損傷定義為固定結果批次的更新後減更新前 NLL ≥0.05 nats。事件至少 4 次、跨 2 個 seed 及 2 個 optimizer 才支持此面板的現象。至少 30 次損傷與 6 個可算 AUROC 的留出 seed 才量第三輪同一組既有診斷加 attention 寫入的等權增量，沿用 ridge 及 +0.03 幅度。覆蓋不足時保留事件計數。閘門前後及硬體失配的事例留作技術診斷。
- 事件步上沿用整步回退參考及四種局部介入，另報通過同量匹配的步數。局部回收只回答這些量測步，正常排程的等品質速度另計。本研究範圍是同一小模型、資料 shard 和有限探針。

一個 H200 工作完整走完後寫聚合 JSON。先前兩分支每 12 個種子約 18 分鐘。此程式只重跑 1×，新加探針，估 **40–65 分鐘**，硬上限 90 分鐘。沿用 8 CPU／32 GiB，暫存含 token 快取與源碼約 0.04 GiB，作業完成清理。無權重下載或 checkpoint。CPU 預檢以微型合成語料對三個 optimizer 驗證加入探針及強制局部反事實後，所有訓練 trace 與未加探針的原 1× 逐值相同。

```bash
stage=codex/experiments/gate-optimizer-nominal
freeze=$(git rev-parse HEAD)
bash "$stage/run.sbatch" "$freeze" --check-only
sbatch --parsable "$stage/run.sbatch" "$freeze"
```

## 讀取紀錄

- [codex] 2026-11-15 12:04（台北），Astra 子代理與 Codex 分別讀取 `nominal-26594/summary.json`，凍結版本 `d89b15761fbf6be47bb7098d3a088061d76751ab`。
