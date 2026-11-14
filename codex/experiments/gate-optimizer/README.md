# Attention gate × optimizer 實驗

執行順序是固定資料切分、開發 seed 調 LR、正式 seed 產生共用 checkpoint、各自分叉正常和壓力排程、機械彙整。判讀定義在 [PREREG.md](PREREG.md)，參數在 [config.json](config.json)。

- `data.py` 讀取已在共用環境的 OpenWebText Parquet 與 GPT-2 tokenizer。兩者都核對 SHA-256。
- `model.py` 實作 tied embedding 的 causal GQA decoder。
- `run.py` 產生 LR 搜索記錄、配對分叉的訓練曲線、探針與單步介入資料。
- `aggregate.py` 根據各次執行寫入的設定彙整三個判準，並標記缺失與未定結果。

## CPU 正確性測試

```bash
cd /work/ChihChungHsu/cactus8603/icml27-topic-search/codex/experiments/gate-optimizer
PY=/work/ChihChungHsu/cactus8603/icml/envs/vllm033/bin/python
OMP_NUM_THREADS=1 "$PY" -B run.py --config config.json \
    --output .smoke/results --corpus .smoke/data --smoke --seconds 45
"$PY" -B aggregate.py .smoke/results
```

重跑時使用另一個空的輸出路徑。正式計算的入口為 `study.sbatch FULL_COMMIT_SHA`。送出前依 [協調規則](../../../COORDINATION.md) 凍結設定、驗證工作樹與佇列，並先建置本資料夾的 `logs/`。用 `bash study.sbatch FULL_COMMIT_SHA --check-only` 可先驗證已追蹤檔案、題目目錄是否乾淨，以及它與該 commit 的差異。正式作業重做相同檢查，從該 commit 匯出每次作業的唯讀快照。資料與程式快照都位於共用 `icml/tmp`，指標寫到 `results/<jobid>/`，並記錄 commit、題目 tree、PREREG 和設定雜湊。
