# GCD 四家族的訓練後結果

此入口使用父層已凍結的 17 臂、三種子與互斥題庫。正式作業依序讀取成功的 CPU 設計及 GPU 技術收據，再驗收全部 51 格步 0 預測。逐格以檔案雜湊及配置比對確認無誤後，才呼叫作者原訓練流程。在作者 `_setup_lora` 完成時擷取實際 LoRA-A／B 的位元雜湊；在第一步更新前再次核對張量位元、可訓練模組及資料列數。正式訓練的每格初始值須與該格實際用於預測的初始值完全吻合。代表臂的合成 GPU 技術收據確認方法可運作，逐格位元對照來自完整的正式步 0 收據。

每格只在共享暫存保留作者的最後一份 LoRA，重載已驗指紋的 adapter 及同一底座，以已凍結模板生成同一批 300 筆漏出題與 200 筆 ID 題。部署端使用作者既有 greedy 生成，訓練 PPS 不掛回。無防禦臂另於同一模型暫時停用 adapter 回答 200 筆 ID 題，提供逐種子的訓練前基準。每格永久收據保留二元與未知列、父層清冊及各技術來源指紋、初始值指紋、adapter 雜湊與耗時。收據不包含生成文字或權重。請沿用父層的共享磁碟與作業預算。

CPU 匯總等 51 格在同一 Slurm 陣列下全數成功才驗收，偵測缺格、額外格、非法評分及未通過的來源。漏出上下界保留格式錯誤與矛盾列的全部分母，先按共同題庫的六個情境等權，再按種子等權；以防禦臂算 ICC 與整體漏出全距保守下界。若同種子無防禦的 ID 增益確定為正，計 `候選 − 0.9 × 訓練後無防禦 − 0.1 × 訓練前基準` 的保守區間，確認是否完全落在 ±5 個百分點。全體結果保留各臂，即使 ID 匹配未定。`summary.json` 的 `arms` 提供臂識別、家族、盲評格式所需的 `leakage` 雙界與 ID 匹配狀態，可與已凍結的預測量另行接入共支撐評估器。此入口的預檢和合成測試只確認資料及技術；正式輸出待 GPU 實跑及獨立複核。

執行目錄位於 `icml/runs/lambda-gcd-family-outcomes/<array>/`。各 SHA 指對應檔案的完整 SHA-256。提交前沿用協作規則查核磁碟與凍結狀態。每格沿用 25 分鐘上限，51 格合計上限 21.25 GPU 小時。新題庫的實際耗時待首格測量。

```bash
stage=codex/experiments/lambda-gcd-family-design/outcomes
bash "$stage/run.sbatch" "$freeze" --check-only "$manifest" "$manifest_sha"
args=("$manifest" "$manifest_sha" "$readiness" "$readiness_sha" "$predictions" "$predictions_sha")
sbatch --parsable --array=0-50%1 "$stage/run.sbatch" "$freeze" train "${args[@]}"
sbatch --parsable --dependency="afterok:$outcomes_array" --kill-on-invalid-dep=yes \
    --job-name=lambda-gcd-family-outcome-judge --gres=none --time=00:10:00 \
    --output=/work/ChihChungHsu/cactus8603/icml/tmp/lambda-gcd-family-outcome-judge-%j.log \
    "$stage/run.sbatch" "$freeze" judge "${args[@]}" "$outcomes_array"
```

可用 `train-auto` 和 `judge-auto` 預先排相依。兩入口均查 `sacct` 來源名稱、成功狀態及末行 JSON 公告的輸出路徑與 SHA-256，並保留逐格驗收。步 0 匯總須以 `lambda-gcd-family-step0-join` 命名，輸出在 `icml/tmp/lambda-gcd-family-step0-join-%j.log`。訓練技術檢查沿用 `lambda-gcd-family-train-ready` 及同名 log。依賴無效時由 Slurm 結束。資料、技術輸出、預測或真實初始值不一致時程式在訓練前退出，單格中途失敗不發布結果。

```bash
args=("$manifest" "$manifest_sha" "$readiness_job" "$signal_join_job")
sbatch --parsable --dependency="afterok:$data_job:$readiness_job:$signal_join_job" \
    --kill-on-invalid-dep=yes --array=0-50%1 "$stage/run.sbatch" "$freeze" train-auto "${args[@]}"
sbatch --parsable --dependency="afterok:$outcomes_array" --kill-on-invalid-dep=yes \
    --job-name=lambda-gcd-family-outcome-judge --gres=none --time=00:10:00 \
    --output=/work/ChihChungHsu/cactus8603/icml/tmp/lambda-gcd-family-outcome-judge-%j.log \
    "$stage/run.sbatch" "$freeze" judge-auto "${args[@]}" "$outcomes_array"
```
