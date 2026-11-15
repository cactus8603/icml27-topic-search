# Λ Phase A 獨立修復版

`run.sbatch` 從指定 commit 擷取一份程式與設定，僅在該份副本套用 `integration.patch`。現有提交與其輸出各自保留。從 `topics/inoculation-lambda/code/killtest/` 提交時傳入階段、唯一跑次名稱、完整 commit 和重跑理由。Git 會依 commit 分別從新目錄或歷史的 `killtests/inoculation-lambda/` 擷取原始檔，套用該 commit 自己的補丁。

跑次寫入 `/work/ChihChungHsu/cactus8603/icml/runs/inoculation-lambda/phasea_v2/<RUN_ID>/`，同一跑次的每個階段只能建立一次。Slurm 輸出在共享 `icml/tmp/%x_%j.out`，工作紀錄在 `icml/runs/inoculation-lambda/logs/RUNLOG.md`。

```bash
sbatch phasea_v2/run.sbatch gate RUN_ID FROZEN_COMMIT "重跑理由"
```

`grant` 與 `pg` 可使用同一跑次名稱與 commit，並以 `afterok` 依賴 `gate`。兩者都會驗證該份 Gate 1 已通過，輸出寫到各自的目錄。它們只供外部論文對照。`pg` 使用上游 social 全部 85 筆。正式的 Phase B 判讀使用 `scripts/phaseb_extras.py`。

Gate 1 開始前會建立本跑次唯一的 `inputs_v2.json`。之後各階段都從該檔指定的本機 Hugging Face 快照載入模型和 tokenizer，並使用跑次目錄內的 PPS 向量副本。憑證保存快照 revision、權重 blob 識別碼與大小、模型設定與 tokenizer 檔案雜湊，以及 PPS 向量 SHA-256。選配模型在建立憑證時若未完整快取，該跑次會略過該模型。

從此目錄上一層執行 CPU 測試。測試會套用補丁到臨時副本。

```bash
python -BI phasea_v2/test_phasea_v2.py
```
