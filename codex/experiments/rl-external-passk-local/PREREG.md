# DeepScaleR 32K 本機 CPU 重播

[codex] 使用已排程的同一份分析、來源、判準與資料。入口從 Git 分別封存實驗 `10d42bdf86534a44b47c03d56ea5e1dd749cdbac`、判準與計算函式 `5808cb676c8fc5ab52b74d7adb0601caa4cfedd6`，並沿用凍結文件要求的作者信賴帶核對。程式只用既有資料與 CPU，單執行緒上限 30 秒。 [原預登記](../rl-external-passk/PREREG.md)

`--check-only` 執行凍結的合成測試，核對實際資產與判準的 SHA-256，僅輸出資產大小與指紋。封存此目錄後可執行 `--run`。每次在共用目錄 `icml/runs/rsd-external-passk-local/` 發布一份獨立 JSON，內容記錄該次本機身分、主機、台北時間、三個 Git 版本及與 Slurm 版相同的科學欄位。讀取結果時登記本機收據檔名與時間。

```bash
stage=codex/experiments/rl-external-passk-local
bash "$stage/run.sh" --check-only
bash "$stage/run.sh" --run FULL_LAUNCHER_COMMIT
```

## 讀取紀錄

- [codex] 2026-11-15 13:58（台北，13:58:25）。已讀 `rsd-external-passk-local.NtoaYWdF.json`，分析凍結 `10d42bd`，本機入口 `24b976b`。 [詳細判讀](../../reports/2026-11-15-topic-status-codex.md#rl-稀疏決策蒸餾)
- [codex] 2026-11-15 13:58（台北，13:58:29）。Astra 獨立讀取同一份彙總。
