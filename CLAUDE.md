# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

這個 repo 由 Claude Code 與 Codex 共用。動手前，請先讀並遵守 `COORDINATION.md`，再看 `LOCKS.md`、`HOLDS.md`，以及 `killtests/SUBMITTED.log` 的最後 20 行。子代理也要遵守，派工時要把這份規則交給它們。

請用繁體中文回覆使用者。

## 兩台機器，規則適用範圍不同
- **H200 叢集**（Slurm、`/work/ChihChungHsu/cactus8603/icml`）：`COORDINATION.md` 第 2 節的 sbatch 流程與第 6 節的 `/work` 暫存路徑、`quota -g -s` 適用。
- **本機 `acvlab`（7 張 RTX 3090）**：沒有 Slurm、`/work` 和 `quota`。
  - 暫存與快取設 `TMPDIR=$repo/hf_home/tmp`、`HF_HOME=$repo/hf_home`，不寫 `/tmp`。
  - GPU 共用；送工作用 `codex/retrieval3090/resources.py` 的 queue 等空卡，並在 `killtests/SUBMITTED.log` 登記 `submit local3090-<pid>`。
- 共享主目錄 2026-11-16 搬遷過：不要沿用寫死舊工作目錄的 launcher。

## 題目 repo
- 每個題目的正本在題目 repo `icml27-<題目>`，本機 checkout 在 `/ssd4/cactus8603/icml27-repos/`。中樞的 `topics/`、`codex/retrieval*/`、`paper/` 是封存，不要修改；題目的工作到題目 repo 做（規則見 `COORDINATION.md` 第 0 節）。

## Claude 的報告
- 放在 `reports/`，檔名 `YYYY-MM-DD-NN-<主題>.md`。編號接 `reports/` 內檔名的最大編號（有重號，先 `ls` 確認）。`reports/INDEX.md` 只收錄從對話紀錄抽出的子代理報告，自己寫的報告不用列進去。
