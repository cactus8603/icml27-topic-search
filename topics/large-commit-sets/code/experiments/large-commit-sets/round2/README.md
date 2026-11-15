# 第二輪等探測預算

固定一個實際到達的 LLaDA 狀態，三個候選各續跑一次。精確單點探測共享；每題兩個額外 probe 組的 full forward 數一致。以 480 道官方訓練題擬合，1,000 道剩餘 GSM8K 測試題提供鎖定評估。細節見 [預登記](PREREG.md)。

沿用第一輪共享模型快照、GSM8K parquet、Transformers 4.46.3 與既有環境。預設每個 GPU 作業配置單 H200、4 CPU 與 48 GB RAM。Smoke 由第二輪兩道獨立訓練題檢查配平、輸入和顯存，上限 30 分鐘。依序收集 train 與 test，各自上限 2h、3h。CPU scorer 配置 4 CPU／12 GB／30 分鐘。

從 repo 根目錄執行測試：

```sh
TOPIC=codex/experiments/large-commit-sets/round2
PILOT_PYTHON=/work/ChihChungHsu/cactus8603/icml/envs/t2irare/bin/python
(cd "$TOPIC" && "$PILOT_PYTHON" -B -m unittest -v test_round2.py)
for SCRIPT in paths.sh collect.sbatch score.sbatch; do bash -n "$TOPIC/$SCRIPT"; done
```

凍結後，依共享規則確認佇列並逐筆登記，使用以下提交命令。依次代入凍結 SHA、已登記的 smoke、train 和 test job IDs。研究判讀只看 CPU scorer 完成的 JSON。

```sh
sbatch --output="$TOPIC/slurm-%j.out" "$TOPIC/collect.sbatch" smoke "$FREEZE"
sbatch --dependency="afterok:$SMOKE_ID" --time=02:00:00 --output="$TOPIC/slurm-%j.out" "$TOPIC/collect.sbatch" train "$FREEZE"
sbatch --dependency="afterok:$TRAIN_ID" --time=03:00:00 --output="$TOPIC/slurm-%j.out" "$TOPIC/collect.sbatch" test "$FREEZE"
sbatch --dependency="afterok:$TEST_ID" --output="$TOPIC/slurm-%j.out" "$TOPIC/score.sbatch" "$FREEZE" "$TRAIN_ID" "$TEST_ID"
```

`collect.py --mode check` 可離線驗證兩份 parquet、三個切分與全部快照檔案，不載入模型、不產生結果。Slurm 啟動時驗證凍結版再將該版程式從 Git 解出至共享暫存區，結果寫回本目錄的 `results/`。中止的 `.partial` 保留作除錯，判讀程式只接完整的 train/test JSONL。
