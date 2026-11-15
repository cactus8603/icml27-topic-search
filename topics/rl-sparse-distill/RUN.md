# 資料集、環境與執行方式

以下路徑的 `$ICML` 是 H200 叢集上的 `/work/ChihChungHsu/cactus8603/icml`。程式與設定在 [本題 code 目錄](code/)，原始資料和權重由列出的叢集路徑讀取。

## 資料集與模型

原正式面板共有 1,453 題，切為開發 400 題及測試 1,053 題。單 token 分叉診斷使用 SimpleRL 作者 `cf1c785` 的 5,000 題來源，去重並排除已用題後剩 3,348 題，從固定排序取第 128 至 255 個，共 128 題。此面板與原正式切分各自判讀。作者資料位於 `$ICML/tmp/rsd-pair-b-cf1c785/data/math/test.jsonl`。

分叉選擇比較基礎模型 `Qwen/Qwen2.5-Math-1.5B@4a83ca6` 與強化學習版 `sail/Qwen2.5-Math-1.5B-Oat-Zero@a98e477`。兩臂從同一前綴出發，固定後續取樣預算。實作中也保留原始配對的資格檢查。[分叉設計](code/experiments/rl-fork-budget/PREREG.md) · [正式配對規則](code/killtest/README.md)

## 環境

使用 `envs/oat-vllm072/bin/python`。Python 3.10.20、vLLM 0.7.2、Torch 2.5.1、CUDA 12.4、Transformers 4.48.2，另需工作區預備好的 `pylib`（Math-Verify 0.9.0）及離線 HF 快取。若評分器放在其他路徑，以 `RL_GRADER_PATH` 指向該目錄。正式診斷使用 H200。

## 怎麼跑

~~~bash
bash scripts/check-frozen.sh af75bf0 \
  codex/experiments/rl-fork-budget/run.sbatch --check-only
~~~

正式分叉先從指定凍結 checkout 用 `run.sbatch "$FREEZE" smoke` 提交技術短測。短測完成，才以 `run.sbatch "$FREEZE" diagnostic "$SMOKE_JOB"` 並設定 Slurm 的 `afterok` 相依。主試驗搭配新的截斷可比配對時依正式配對規則執行。[提交和輸出](code/experiments/rl-fork-budget/PREREG.md)

診斷摘要在 `$ICML/runs/rsd-fork-budget/<job>.json`，包含數字及來源雜湊。

[CPU 預檢工具](../../scripts/check-frozen.sh)從 Git repo 根目錄讀取指定提交，在叢集暫存區建立原路徑的執行副本。需要完整 Git 歷史和已備妥的資料、環境。正式 GPU 作業沿用對應預登記指定的執行工作區與凍結提交，依[協作規則](../../COORDINATION.md)提交及登記。

[回題目首頁](README.md)
