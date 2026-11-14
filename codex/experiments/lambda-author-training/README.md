# Lambda Phase B 作者 Spanish 訓練

這是兩臂訓練流程的技術核對。沿用作者 `6964328` 的 `traits.spanish.base.none` 與 `traits.spanish.base.ip`、固定 Spanish 訓練資料 1,000 列和三提示 IP 銀行。用 `1280b1f` 的 Phase B 資料預檢和設定產生器，核對輸入指紋 `1a0873aa84988bf845d9e7edd2ab57c8873c4a359c2459eef99fcd66919ae1c0`、Python 3.10.20、直接依賴版本、完整 Qwen snapshot `a09a35458c702b33eeacc393d103063234e8bc28`。作者程式、設定輔助程式及兩個訓練輸入各從固定 Git commit 複製至暫存。此執行只保留最終 LoRA，對照作者的儲存規則，停用途中快照及會顯示訓練文字的除錯印出。

CPU 預檢用作者的 `resolve_args`、tokenizer 與 `FinetuneDataset`，分別核對兩臂資料筆數及三個固定樣本的非空 token 張量和未遮罩標籤，記錄筆數及檢查計數。兩臂分開送件，先送陣列 `0=none`，陣列 `1=ip` 用 `afterok` 等待前一臂成功，並額外核對其完成收據。每臂沿用 seed 42、BF16、LoRA rank 64、batch 4 × 累積 8、三輪訓練及全 token loss。關閉 HF 網路存取，訓練不呼叫評審。每臂申請 1 張 H200、8 CPU、64 GiB RAM，估計 45 分鐘，Slurm 上限兩小時。估計兩份最終 LoRA 合計約 1.20 GiB，另有小型 tokenizer、收據和日誌。

從載入模型起算，記錄滿十分鐘後作者日誌報告的第一個微批次完成數、實測吞吐和推估全程時間。若較早完成，直接記錄實際耗時。首筆估計超過 90 分鐘時停止該臂並保留 `throughput.json`。若 `none` 估時超限，Slurm 不啟動 `ip`，由協調者決定如何重排。進度從作者原本的日誌取得，使用同一份訓練資料與參數。

每臂產出在 `/work/ChihChungHsu/cactus8603/icml/runs/inoculation-lambda/author-training-<array-job>/{none,ip}/`。作者訓練正常結束、完整存檔標記與檔案大小吻合、無 pending 標記，且 LoRA 來源模型、設定及作者留存的實際參數全部核對後，寫 `receipt.json` 記錄來源版本、輸入及檔案雜湊。草稿或失敗輸出保留在其獨立作業目錄。Phase B 行為重現另需作者評審規格的證據。

```bash
stage=codex/experiments/lambda-author-training
freeze=$(git rev-parse HEAD)
bash "$stage/train.sbatch" "$freeze" --check-only
# 完成獨立審查與協調後執行
none=$(sbatch --parsable --array=0 "$stage/train.sbatch" "$freeze")
none_id=${none%%;*}
sbatch --parsable --array=1 --dependency="afterok:$none_id" "$stage/train.sbatch" "$freeze" "$none_id"
```
