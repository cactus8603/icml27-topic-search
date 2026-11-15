# 平行解碼的免額外 forward 選擇器與事後上限

[codex] 2026-11-15 16:20（台北）。制定前已讀第二輪公開彙總。原機械彙總的首次讀取登記為同日 06:09。本次只計算原逐題收集檔的 SHA-256，尚未檢視逐題內容。本診斷重用原 480 道 train、1,000 道 test、首個合格介入狀態及三個候選，範圍限這組 GSM8K、LLaDA-8B-Instruct、原切點及解碼預算。新方法在同面板成立時仍需獨立題組確認。 [原正式判讀](../../../reports/2026-11-15-three-frozen-readbacks-codex.md#大型平行解碼提交集合) · [原協議讀取紀錄](../round2/PREREG.md#讀取紀錄)

## 介入時可用的資訊

原 confidence 解碼在每步用自然 forward 的 logits 取得各位置 argmax token 的 log-probability。原候選生成依這些 log-probability 差、位置跨度及位置索引決定候選順序。第二輪收據的 `confidence` 直接來自同一批介入前 proposal，候選被額外探測或續寫後的分數不進選擇。程式從保存的位置與原始 log-probability 重建兩個交換，驗證共同位置的信心、距離、差值及次序。

介入後只執行所選候選的續寫，部署預算同原 confidence 的每題 32 次 forward。重播直接使用已存在的各候選判分，執行分析時不載入模型。只有現象重算使用原第二輪的探測資料。

固定三個資訊項目形成至多八個狀態。

- 首個合格 cut 為第 2 或第 10 步，以原從零起算的索引表示。
- 將索引 1 候選的 log-probability 差 `drop − add` 和合格 train 題的中位數比較。
- 將兩個交換中最短的 token 位置跨度和合格 train 題的中位數比較。

兩個數值恰等於中位數時均進較低一側。只在至少有 20 道 train 題的狀態提供三個候選動作。用該狀態所有 train 題選答對數最多的單一候選索引，同分依索引 0、1、2 順序。沒有合格狀態、train 未見或少於 20 題的一律回候選 0，留在全題分母。test 不改 cutoffs、狀態資格或所擬合的決策。

## 三項量測

- **現象**重算第二輪原候選在 singleton 加總 margin 與 joint margin 間至少一位置跨過 `log(9)` 的合格題數，核對已公布的 Wilson 區間、覆蓋及原 15% 判定。僅重驗原現象的收據一致性。
- **訊號**在全 1,000 道 test 上比較 train 擬合選擇器與原 confidence 答對率，輸出配對題目 bootstrap 95% 區間、救回及退步題數。固定原 5,000 次重抽、seed 及至少 200 道有候選池的覆蓋要求。歷史診斷中，區間下界至少 +4 點為達參考幅度、上界低於 +4 點為低於參考，其餘未定。
- **上限**在相同的 train cutoffs、合格狀態、三候選與退回規則下，事後用 test 金標為每個有 test 題的合格狀態選擇單一最佳候選索引，得到這組固定狀態的精確有限面板上限。計算原 confidence 與 train 擬合方法各距此上限多少。若狀態上限距原 confidence 少於 +4 點，這組狀態在此面板沒有達到參考幅度的空間。同資格候選池上限允許在合格狀態內逐題依金標選任一候選，其餘仍退候選 0。另列原全池上限作參考，允許所有原有候選池的題目逐題選任一候選，原無候選池的題目仍回候選 0。上限使用已知 test 標籤，僅報點估。

## 執行與收據

使用原封存 `c979eba0260a2a6f5bd3a1de5cf6affb4ea7a5af` 的 row loader 驗收資料雜湊、題目 ID、第一合格 cut、候選和等額續寫。正式入口先核對 `26441`、`26442`、`26443` 均成功，核對已公開原判讀 SHA-256 與來源 metadata。新結果保留來源兩份收據的 SHA-256，與原判讀交叉核對現象、原 confidence 及同池上限。現象、訊號與上限均用原分母，並回報實際有資格改選的樣本數。

| 固定輸入 | SHA-256 |
|---|---|
| `train-26441.jsonl` | `f3ad5fba802c733287259ec2827aaf9a387e28cecff29293c1bde52c6162fd39` |
| `test-26442.jsonl` | `8144334c52bb955006e3b8fde7392ab277f7f2c8337ffd40f0d470cd083e7082` |
| `verdict-26443.json` | `c7397ad04f021b7a5a107464b22342f2268e25b10a7b38e213c798fe9831814d` |

離線單核，無 GPU。透過封存入口先跑合成測試與靜態配置檢查。正式執行只寫一份帶 SHA-256 及固定來源的聚合 JSON。預計少於 16 KiB。依協作規則登記讀取，再獨立複核。

## 讀取紀錄

- [codex] 2026-11-15 16:30（台北）。排隊作業 `26737` 因共用 CPU 配額取消，確認未啟動後用同凍結版本 `b3507617caae44d0a5b64bed9599c7b76fd04194` 的離線單核入口完成。`runs/large-commit-sets-observable-selector/local-gSjPYX2G/summary.json` 共 3,219 bytes，SHA-256 `660b74c1d0f81215cb8fc87f112341d764ea6a56dea572d14977e6ead54bb3d2` 與收據吻合。重算原跨線 466/946。全 1,000 題原 confidence 297、訓練擬合 300（配對 +0.3 個百分點，95% 區間 −2.1 至 +2.7）、同狀態最優 305、同資格逐題最優 426、原逐題最優 430。Astra 獨立複核差額及推論範圍。

```sh
stage=codex/experiments/large-commit-sets/observable-selector
bash "$stage/run.sbatch" FULL_FREEZE_COMMIT --check-only
sbatch "$stage/run.sbatch" FULL_FREEZE_COMMIT
bash "$stage/run.sbatch" FULL_FREEZE_COMMIT --local
```
