# 資料集、環境與執行方式

以下路徑的 `$ICML` 是 H200 叢集上的 `/work/ChihChungHsu/cactus8603/icml`。程式與設定在 [本題 code 目錄](code/)，原始資料和權重由列出的叢集路徑讀取。

## 資料集與模型

用 CIFAR-10 官方 50,000 張訓練圖、10,000 張測試圖，位置為 `$ICML/data/cifar10/cifar-10-batches-py`。新遮罩確認固定測試排列中的第 50 至 99 個查詢，對同一組查詢使用 64 組新遮罩，每組移除 1% 訓練圖；另重播兩組歷史遮罩作技術對照。模型為從零訓練的 ResNet-9，寬度 80，20 個 epoch，每批 500 張。

須備妥 `runs/multifid/stage0` 與 `runs/multifid/signal-confirm/26734` 的歷史權重、張量和收據，後續才可用同一訓練軌跡比較 EK-FAC、Spell 和精確歸因。[基礎作業](code/killtest/) · [新遮罩資料與設計](code/experiments/multifidelity-new-mask-confirm/PREREG.md)

## 環境

`envs/attrib-jax` 用於 GPU 續訓，Python 3.12.3、JAX 0.12.0.1、CUDA 13 外掛。`envs/attrib-torch` 用於其餘數值處理，Python 3.12.3、Torch 2.5.1、NumPy 1.26.4、SciPy 1.17.1。正式作業用一張 H200。

## 怎麼跑

~~~bash
bash scripts/check-frozen.sh 1f3e88e \
  codex/experiments/multifidelity-new-mask-confirm/run.sbatch --check-only
~~~

正式重播需要歷史 `26734` 的結果與日誌。預檢通過後，在相應凍結 checkout 用 `sbatch --parsable "$ENTRY" "$FREEZE"`，其中 `ENTRY` 為上列 `run.sbatch`。輸出位於 `$ICML/runs/multifid/new-mask-confirm/<job>/`，主要有 `retrain.npz`、`retrain-receipt.json` 與 `summary.json`。[完整操作](code/experiments/multifidelity-new-mask-confirm/PREREG.md)

[CPU 預檢工具](../../scripts/check-frozen.sh)從 Git repo 根目錄讀取指定提交，在叢集暫存區建立原路徑的執行副本。需要完整 Git 歷史和已備妥的資料、環境。正式 GPU 作業沿用對應預登記指定的執行工作區與凍結提交，依[協作規則](../../COORDINATION.md)提交及登記。

[回題目首頁](README.md)
