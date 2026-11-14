# 固定 4× 壓力的新種子獨立驗證

[codex] 2026-11-15 台北時間。第二輪固定 4× 驗證的彙總 `results/validation-26502/validation.json` 已於 07:13 獨立複核，SHA-256 `086cdab5a836e33cf4f33a25298617ece0fa14b152897c4f104a39f6d408a57b`。依舊判定保留該輪各項結論，新一輪以擴大的獨立種子組評估相同問題。

## 資料與固定設計

固定 48 個全新 seeds。索引 0–47，各取 UTF-8 `gate-optimizer-round3-v1:<索引>` 的 SHA‑256 首四個位元組，以 big-endian 解為整數後保留低 31 位。先驗證無重複及與首輪 dev／test、校準、第二輪獨立驗證 seeds 互斥。依索引順序每 12 個切一個分段。沒有替補、追加組別或分段判讀，分析資料只來自本輪 48 組。

使用 `4d87430fd7cda6e74ea9e48deb57411911733e04` 的 `cell`、`judge`、原驗證總結及底層設定，保留同一份 OpenWebText 分割和輸入順序規則。每個分段重新建 token 快取，要求 manifest SHA-256 與前次相同。每個 seed 配對 StableAdamW、Muon、AdamW + sandwich RMSNorm 各自的正常參照與 4× 壓力，訓練步數、資料、20 步 ramp、10 步探針、局部回退及互斥 signal／outcome 資料沿用原規格。共 144 對，每對最多 38 個 ramp 後探針。

## 單次判讀

四個分段全部完成後驗證工作 ID、凍結版本、輸入雜湊、軟體／裝置、預定種子、精確檔名、分支結構及每個分支雜湊。只在全部收據合格後計算一次原判讀。質量守門仍要求所有配對合格，末步 dev NLL 差不得大於 0.5 nats，且超過 0.5 的連續評估少於三次。損傷仍定義為 ramp 後每個探針的更新後減更新前 NLL ≥0.05 nats。至少 30 次損傷，且至少 6 個留出 seed 可算 AUROC，才計算原整 seed 留出 ridge logistic。等權平均配對 AUROC 增量至少 0.03 支持人工壓力訊號，描述性 95% 區間沿用原程式。資料缺漏、守門未過或覆蓋不足維持不可判。

局部整步回退衡量在選定損傷步回到更新前 NLL 的差，按原規則另報方向、均勻縮步、正交對照與 W_O 回退。範圍限於同 OpenWebText shard、固定 4× 壓力及選定步，正常排程、資料集外泛化及等品質訓練速度需另外驗證。

## 執行與磁碟

四個 GPU 子作業依序最多同時一個，各申請 1 張 H200、8 CPU、32 GiB RAM、1 小時上限。每段工作量近似 26502 的 18 分鐘，保留種子與叢集變動餘裕，預估每段約 20–40 分鐘。一次 CPU 分析申請 2 CPU、8 GiB RAM、20 分鐘。GPU 暫存包含該段約 33 MB 的 token 快取及來源快照，CPU 暫存包含驗證過的分支副本，結束後清除。144 對分支、收據與一份機械彙總寫入 `/work/ChihChungHsu/cactus8603/icml/runs/gate-optimizer/fresh-<array-id>/`，估低於 0.1 GiB，無 checkpoint。任一分段作業失敗時不啟動 CPU 分析。

```bash
stage=codex/experiments/gate-optimizer/round3
freeze=$(git rev-parse HEAD)
bash "$stage/produce.sbatch" "$freeze" --check-only
# 完成獨立審查及協調後執行
array=$(sbatch --parsable "$stage/produce.sbatch" "$freeze")
sbatch --parsable --dependency="afterok:${array%%;*}" "$stage/analyze.sbatch" "$freeze" "${array%%;*}"
```
