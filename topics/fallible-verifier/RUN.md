# 資料集、環境與執行方式

以下路徑的 `$ICML` 是 H200 叢集上的 `/work/ChihChungHsu/cactus8603/icml`。程式與設定在 [本題 code 目錄](code/)，原始資料和權重由列出的叢集路徑讀取。

## 資料集與模型

原 GSM8K 對照從官方 1,319 題選出 997 道未用題，再分成 250 題暖機及 747 題留出，使用公開的四候選。ASDiv-A 續驗使用作者 `55790e5` 的 XML 與五份索引，另分為 250 題暖機（180 一步、70 兩步）和 747 題留出（539 一步、208 兩步）。每題四份候選，暖機 1,000 份、留出 2,988 份。

ASDiv-A 資料位於 `$ICML/data/asdiv-official/prepared-110881d5/`，由 `questions.jsonl`、`references.jsonl` 和 `prepared.json` 綁定。正式續驗需要成功的 GSM1k 暖機作業 `26704` 及驗收摘要雜湊。[資料準備](code/experiments/fallible-verifier/asdiv-replication/PREREG.md)

生成模型為 `Qwen/Qwen2.5-7B-Instruct@a09a354` 和 `sail/Qwen2.5-Math-1.5B-Oat-Zero@a98e477`，每個模型每題生成兩份候選。弱評審用 `Qwen/Qwen3-0.6B@c1899de`，強評審用 `Qwen/Qwen3-32B-AWQ@0499c3a`。

## 環境

`envs/oat-vllm072` 生成候選，Python 3.10.20、vLLM 0.7.2。`envs/vllm033` 執行評審，Python 3.12.16、vLLM 0.33.0。`envs/t2irare` 在 CPU 彙總，Python 3.12.3。使用離線 HF 快取，並從固定歷史提交取用 RL 題的模型配對設定與取樣工具。[RL 配對與取樣程式](../rl-sparse-distill/code/)

## 怎麼跑

~~~bash
bash scripts/check-frozen.sh 885ac70 \
  codex/experiments/fallible-verifier/asdiv-replication/run.sbatch dbc60ce5659b056a85992091aebd1ee2e5d5a3bf --check-only
~~~

正式執行參數依序為 ASDiv-A 凍結提交、指令生成器提交、暖機作業 `26704` 與已核對的暖機摘要 SHA-256。由固定的執行 checkout 提交 `run.sbatch`。輸出位於 `$ICML/runs/fallible-verifier/asdiv-replication/<job>/`，包含 `instruct/`、`oat/`、`panel/`、`scoring/` 及 `summary.json`。[正式操作與判準](code/experiments/fallible-verifier/asdiv-replication/PREREG.md)

[CPU 預檢工具](../../scripts/check-frozen.sh)從 Git repo 根目錄讀取指定提交，在叢集暫存區建立原路徑的執行副本。需要完整 Git 歷史和已備妥的資料、環境。正式 GPU 作業沿用對應預登記指定的執行工作區與凍結提交，依[協作規則](../../COORDINATION.md)提交及登記。

[回題目首頁](README.md)
