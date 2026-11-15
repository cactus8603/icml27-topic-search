# 原種子 Gate (b) 修復後的正式文生圖確認

[codex] 2026-11-15（台北）。原題的樣本、事件、完整度、選格、灰區規則、人工標準、GPU 小時及磁碟限制沿用已凍結的設定。選格、判讀及盲標由原程式執行。歷史 Gate (b) 結果留在原位置。本路徑以原三提示、原 30,000 種子和原門檻的修復重播 `26663` 為前置。啟動前需要其完整來源重算、Slurm 成功狀態及獨立複核後的收據雜湊。

## 盲標的百分比格對照

百分比格的最低凍結事件率擬定為 **1%**（含邊界）。原研究在百分比區和罕見區間劃分量級。取最低整百分比，比罕見區上界 0.1% 高十倍，無資料驅動選門檻。50 個百分比格陽性、50 個陰性、至少 50 個罕見陽性及兩組人工控制門檻均依原規格。控制樣本不足則維持未能測試。這個單一增補須在開啟任何正式雙偵測器目標分數前由另一位審查者確認，並在提交前記錄。人類標註完成前保留盲性。

- [codex] 2026-11-15（台北），未參與程式實作的主代理核對 1% 的量級理由與原組別規則，確認這是凍結前新增的唯一數字。原兩組對照的 80%／10% 人工門檻不變。

## 執行與完整性

- 使用獨立 `icml/runs/t2irare/formal-replay/<freeze>/`，採同一凍結的執行與判讀版本。所有分段與原本相同，盲標檔沒有模型、提示或分數資訊。判讀用的 key 保留在工作資料夾。
- Stage 1 以六陣列任務各生成一個模型的 14 格。CPU 選格先驗所有原定 500 種子的資料／側錄對，按原腳本各階段的前綴與排名規則產生兩個清單。下階段陣列綁定兩份清單的雜湊，每個分片的完成收據對應原陣列配置及材料。
- 設定成本超出原規定、偵測值或來源異常則暫停。程式預檢及合成測試可以先跑，預檢不讀修復作業判讀、實際生成分數或標註。

```bash
stage=codex/experiments/t2i-formal-replay
bash "$stage/run.sbatch" "$freeze" preflight
sbatch --array=0-5%2 "$stage/run.sbatch" "$freeze" --receipt-sha "$receipt_sha" sample 1
bash "$stage/run.sbatch" "$freeze" --receipt-sha "$receipt_sha" plan 2a
sbatch --array="0-$last%2" "$stage/run.sbatch" "$freeze" --receipt-sha "$receipt_sha" --plan-sha "$plan_sha" sample 2a
```

每個陣列以 `formal-stage=1`、`formal-stage=2a`、`formal-stage=2b` 或 `formal-stage=2b_extra` 開頭記錄提交原因。依輸出任務數調整 `last`。未選出格時仍保留選格清單，直接進下一個 CPU 步驟。`2b_extra` 只有原判準 A 灰區允許。

正式取樣完整後執行 `aggregate` 和 `make-blind`。若人工盲標素材需重生成，單張 H200 提交同一入口的 `regen`。只將 `review/index.html`、`review/review_imgs/`、`review/review_blind.csv` 與凍結評分規則交予標註者。人工回填後跑 `review-stats` 及 `aggregate` 更新三項判讀。對照列的身分與原分數在標註後才開啟。

## 讀取紀錄
