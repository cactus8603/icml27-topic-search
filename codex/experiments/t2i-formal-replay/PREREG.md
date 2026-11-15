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
bash "$stage/run.sbatch" "$freeze" --receipt-sha "$receipt_sha" verify-gate
sbatch --array=0-5%2 "$stage/run.sbatch" "$freeze" --receipt-sha "$receipt_sha" sample 1
bash "$stage/run.sbatch" "$freeze" --receipt-sha "$receipt_sha" plan 2a
sbatch --array="0-$last%2" "$stage/run.sbatch" "$freeze" --receipt-sha "$receipt_sha" --plan-sha "$plan_sha" sample 2a
```

每個陣列以 `formal-stage=1`、`formal-stage=2a`、`formal-stage=2b` 或 `formal-stage=2b_extra` 開頭記錄提交原因。依輸出任務數調整 `last`。未選出格時仍保留選格清單，直接進下一個 CPU 步驟。`2b_extra` 只有原判準 A 灰區允許。

### 分段成本與續跑

送出每段前，以 `python3 -B "$stage/schedule.py" "$freeze" PHASE` 核對既有 GPU 配置時間、該段未完成任務、預計耗時和各陣列的每任務時間上限。首段估速來自完整三個偵測器的基準。完成首段後改用累計實際配置時間與已完成列數，含失敗重試成本。加總各陣列上限仍須在原 50 GPU 小時及每階段兩倍估時內。`array_groups` 為空清單表示無需 GPU，`null` 表示按當前速度無法在兩個原限制內完整生成，須先重排成本。

把每組 `indices` 轉成 `sbatch --array=ID,ID%2 --time=MINUTES …`，用同一份選格雜湊。25,000 與尾段 5,000 張分兩組，依原索引執行。若共享卡數要維持兩張，前一組全部終止後才提交後一組。排程查詢使用 `sacct -X -n -P -j ARRAY --format=JobID,State,ExitCode`，每分鐘查一次。全部計畫分片成功且收據通過後，才執行下一個 `plan`。部分失敗時先重算剩餘時間，重新提交尚未成功的原分片，保留已完成資料及原 seed 範圍。提交與重試都以原協作規則登記。

正式取樣完整後執行 `aggregate` 和 `make-blind`。若人工盲標素材需重生成，單張 H200 提交同一入口的 `regen`。只將 `review/index.html`、`review/review_imgs/`、`review/review_blind.csv` 與凍結評分規則交予標註者。人工回填後跑 `review-stats` 及 `aggregate` 更新三項判讀。對照列的身分與原分數在標註後才開啟。

## 讀取紀錄

- [codex] 2026-11-15 16:33（台北）已讀 `icml/runs/t2irare/gate-b-original-replay/26663/precision_replay_verified.json` 及該收據所綁的 60 對 `.npz`／側錄。獨立執行原凍結 `--verify-job 26663`，以逐種子資料重算並核對前置證據及收據，SHA-256 `d88b2d37b09d64a57a3102ecd7049c57d616aa19ed3fcc85a69a5dc447f9cffe`。

## 執行修正

- [codex] 2026-11-15（台北）。`26743` 的 Gate (b) 啟動檢查因原重播文件新增讀取紀錄而退出。正式入口從原來的固定 Git 提交抽取驗證器、生成程式、精度修復及設定，以原 `--verify-job` 重驗全部 60 對資料、先決條件與收據。`verify-gate` 以相同路徑檢查已審查的收據後退出。實驗設計與門檻沿用同一版。
## 分片時間取整修正

- [codex] 2026-11-15 18:17（台北）。比例分配的整分鐘若已足夠，保留判讀及原時間上限。若尾分片被向下取整而不足，改核各分片最小足夠整分鐘，再驗各分片原上限與含歷史作業的 50 GPU 小時總上限。用已公開的原種子重播吞吐和最滿的六格 2b 經回歸測試。實際是否可執行仍用機械選格、已完成的配置時間及實測吞吐。未讀正在執行的正式分數，`26765` 與 `26802` 沿用原封存入口。
- [codex] 2026-11-15 18:26（台北）。`26765_0` 已確認成功，僅讀 `operations/1-0-26766.json` 的階段、模型、種子範圍、分塊數與兩份版本綁定，未讀正式分數。`26802` 仍依賴完整陣列，其餘模型完成後才由正式入口驗收並選格。
