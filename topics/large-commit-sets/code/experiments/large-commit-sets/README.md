# 大型提交集合先導

在固定的 LLaDA confidence 解碼中，從同一個狀態比較等大小的三個提交集合。即將提交的目標 token 在每個探測輸入都保持 masked。精確成對與精確聯合探測使用同一候選池和計分函式，所有候選用固定 forward 預算續跑。實驗單位、樣本、門檻及範圍見 [PREREG.md](PREREG.md) 與 [thresholds.json](thresholds.json)。

## 需要的本機資源

- 一張 H200、Slurm `h200q`、4 CPU、48 GB RAM。Smoke 上限 30 分鐘，主跑上限 6 小時。模型約 16 GB，主 JSONL 預估低於 10 MB。
- 預先準備 `GSAI-ML/LLaDA-8B-Instruct` 固定版 `08b83a6feb34df1a6011b80c3c00c7563e963b07`、完整 safetensors shards 和 tokenizer。預設位於共享 HF 快取的 snapshot；`LLADA_MODEL_PATH` 可提供同 SHA 命名的絕對路徑。`pilot.py` 檢查已核對的 checkpoint config 和兩個模型程式雜湊，僅由本機讀取；Transformers 4.46.3 使用 checkpoint 的 `auto_map` 載入。
- 共享 `icml/envs/t2irare` 提供 Transformers 4.46.3、PyTorch、accelerate 及判讀用的 NumPy／SciPy。共享 `icml/envs/biasgen` 的 PyArrow 讀取 `icml/data/gsm8k/main/test-00000-of-00001.parquet` 和 `icml/data/gsm8k-hf/main/train-00000-of-00001.parquet`。Worker 全程 offline。模型模組快取落在共享 `icml/tmp`。

## 從 repo 根目錄執行

純 CPU 合成檢查及 Slurm 語法檢查：

```sh
PILOT_PYTHON=/work/ChihChungHsu/cactus8603/icml/envs/t2irare/bin/python
(cd codex/experiments/large-commit-sets && "$PILOT_PYTHON" -B -m unittest -v test_pilot.py)
bash -n codex/experiments/large-commit-sets/run.sbatch
```

預先凍結主判準後，依共享協作規則查佇列、提交與登記。主跑可以用 Slurm `afterok` 接 smoke。`SMOKE_JOB_ID` 設為已送出並登記的 smoke ID，`FREEZE_COMMIT` 設為凍結本目錄的 commit。每個 `sbatch` 在 `killtests/SUBMITTED.log` 登記。

```sh
sbatch --output='codex/experiments/large-commit-sets/slurm-%j.out' \
  codex/experiments/large-commit-sets/run.sbatch smoke
sbatch --dependency="afterok:$SMOKE_JOB_ID" --time=06:00:00 \
  --output='codex/experiments/large-commit-sets/slurm-%j.out' \
  codex/experiments/large-commit-sets/run.sbatch main "$FREEZE_COMMIT"
```

完成後使用 CPU 判讀完整 `results/main-<job-id>.jsonl`，將 `MAIN_JOB_ID` 設為主跑 job ID。

```sh
cd codex/experiments/large-commit-sets
"$PILOT_PYTHON" -B judge.py "results/main-$MAIN_JOB_ID.jsonl" "results/verdict-$MAIN_JOB_ID.json"
```

Raw JSONL 包含實際樣本、候選位置／固定 token、margin、候選續跑答案、步數與探測耗時。主跑成功前保留 `.partial`，提交後的判讀輸出與試驗記錄依 repo 協作規則保留。
