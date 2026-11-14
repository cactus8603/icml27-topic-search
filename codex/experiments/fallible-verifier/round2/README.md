# 等額暖機的強驗證器重播

在此目錄從共享資料檔檢查候選覆蓋。讀取 Parquet 的題目、作答模型和追蹤識別欄，並校驗固定分片。此步不讀候選解或金標，也不跑模型。

```bash
shared=/work/ChihChungHsu/cactus8603/icml
"$shared/envs/vllm033/bin/python" -B protocol.py --check-only \
  --prior-dir "$shared/tmp/codex-fallible-verifier-08f42e1" \
  --solver-dir "$shared/tmp/codex-fallible-verifier-round2-solver-08f42e1"
"$shared/envs/t2irare/bin/python" -B -m unittest -v test_round2.py
```

正式工作按 [預先登記](PREREG.md) 和 [協作規則](../../../../COORDINATION.md) 凍結並檢查佇列。推論使用一張 H200，申請 60 分鐘、48 GiB CPU 記憶體。分析申請 1 CPU、4 GiB、10 分鐘。兩階段都離線。

```bash
freeze=REVIEWED_FULL_COMMIT
prior_job=PRIOR_TOPIC_TERMINAL_JOB_ID
[[ "$prior_job" =~ ^[1-9][0-9]*$ ]] || exit 2
job=$(sbatch --parsable --dependency="afterany:$prior_job" run.sbatch "$freeze" "$prior_job")
job=${job%%;*}
[[ "$job" =~ ^[1-9][0-9]*$ ]] || exit 2
sbatch --dependency="afterok:$job" analyze.sbatch "$freeze" "$job"
```

GPU 啟動時用 Slurm accounting 核對前置工作已終止。只有本輪推論成功才執行分析。輸出位於 `results/<inference job id>/`，包含模型讀取的 `prompts.json`、分析端的 `gold.json`、兩個 verifier 的結構化輸出與 `summary.json`。兩階段共用凍結 commit 和設定 hash。模型工作只讀 prompts，摘要連同各輸入的 SHA-256 和前置工作資訊保存。在相同資料夾重跑不覆寫輸出。
