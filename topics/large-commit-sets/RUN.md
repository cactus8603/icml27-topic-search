# 資料集、環境與執行方式

以下路徑的 `$ICML` 是 H200 叢集上的 `/work/ChihChungHsu/cactus8603/icml`。程式與設定在 [本題 code 目錄](code/)，原始資料和權重由列出的叢集路徑讀取。

## 資料集與模型

此題暫停。最近的 CPU 重播用 GSM8K 官方訓練切分內固定的 480 題開發資料，以及官方測試切分中的 1,000 題留出資料。CPU 重播使用原逐題結果與 26441、26442、26443 的收據。最近獨立 GPU 對照另取 512 道未用過的 GSM8K 訓練題和 128 道 MATH-500 十進位題，分開判讀。[CPU 重播設計](code/experiments/large-commit-sets/observable-selector/PREREG.md) · [獨立 GPU 對照](code/experiments/large-commit-sets/top-two-selector/PREREG.md)

模型為 `GSAI-ML/LLaDA-8B-Instruct@08b83a6feb34df1a6011b80c3c00c7563e963b07`，BF16，原流程固定 32 次前向運算。GSM8K 讀取 `$ICML/data/gsm8k-hf/main/train-00000-of-00001.parquet` 和 `data/gsm8k/main/test-00000-of-00001.parquet`，MATH-500 則在 `data/MATH-500/test.jsonl`，固定版 `6e4ed1a2`。[GSM8K](https://huggingface.co/datasets/openai/gsm8k) · [MATH-500 固定版本](https://huggingface.co/datasets/HuggingFaceH4/MATH-500/commit/6e4ed1a2a79af7d8630a6b768ec859cb5af4d3be)

## 環境

`envs/t2irare` 為主要環境，Python 3.12.3、Torch 2.5.1、Transformers 4.46.3、NumPy 1.26.4、SciPy 1.17.1。`envs/biasgen` 用 PyArrow 26.0.1 讀 Parquet。CPU 重播只用原收據及資料。

## 怎麼跑

~~~bash
bash scripts/check-frozen.sh b3507617caae44d0a5b64bed9599c7b76fd04194 \
  codex/experiments/large-commit-sets/observable-selector/run.sbatch --check-only
~~~

完整 CPU 重播從 `b3507617` 的相應 `run.sbatch` 使用 `--local`，並保留 `c979eba0` 的舊作業收據。這些逐題收據仍在原執行工作區；登記讀取後才判讀。正式 GPU 入口與資料準備方式可查固定的 `9ab67c94` 說明。CPU 結果位於 `$ICML/runs/large-commit-sets-observable-selector/<id>/summary.json`。

[CPU 預檢工具](../../scripts/check-frozen.sh)從 Git repo 根目錄讀取指定提交，在叢集暫存區建立原路徑的執行副本。需要完整 Git 歷史和已備妥的資料、環境。正式 GPU 作業沿用對應預登記指定的執行工作區與凍結提交，依[協作規則](../../COORDINATION.md)提交及登記。

[回題目首頁](README.md)
