# 資料集、環境與執行方式

以下路徑的 `$ICML` 是 H200 叢集上的 `/work/ChihChungHsu/cactus8603/icml`。程式與設定在 [本題 code 目錄](code/)，原始資料和權重由列出的叢集路徑讀取。

## 資料集與模型

- **同圖排序**用 Half-Truths（`kbora/Half-Truths@ddcd448`），三個子集分別有 509、454、474 筆，主判準用 509 筆。
- **多正例面板**取 COCO train2017 的 984 張 JPEG、123 組、246 條查詢。每條查詢對八張固定圖片有四個正例、四個反例。輸入在 `$ICML/data/coco2017-retrieval-multipositive/`，包含 `panel.json`、`image-receipt.json` 和圖片。
- **固定 50 次重排**使用既有 `data/coco2017-anchor-rerank` 的 8,191 張圖、136 條查詢、118 個重抽樣群。這一輪要用成功作業 `26601` 的原圖池收據。

模型為 `kbora/CS-CLIP@afd6e90`、`Alibaba-NLP/core-reranker-2b@64e1a36`。固定 50 次重排另外用 `Alibaba-NLP/core-emb-2b@cc9870e`。沿用原圖庫後，獨立圖庫的查詢另用固定 COCO 標註建清冊。[多正例設定](code/experiments/retrieval-multipositive/PREREG.md) · [固定預算設定](code/experiments/retrieval-recall-budget/PREREG.md) · [獨立圖庫準備](code/experiments/retrieval-independent-panel/PREREG.md)

## 環境

使用 `envs/vllm033/bin/python`。Python 3.12.16、vLLM 0.33.0、Torch 2.14.0、Transformers 5.20.2。正式編碼和重排使用一張 H200，圖片、模型與 tokenizer 在叢集備妥。

## 怎麼跑

~~~bash
bash scripts/check-frozen.sh a84de2f \
  codex/experiments/retrieval-multipositive/run.sbatch --check-only
bash scripts/check-frozen.sh 569b6c0 \
  codex/experiments/retrieval-recall-budget/run.sbatch --check-only
~~~

預檢會驗 COCO 面板與來源。正式作業在相應凍結 checkout 內，以 `sbatch --parsable "$ENTRY" "$FREEZE"` 執行表列的 `run.sbatch`，並依協作規則登記。結果位於 `$ICML/runs/retrieval-multipositive/<job>/summary.json` 與 `$ICML/runs/retrieval-recall-budget/<job>/{summary.json,per_query.json}`。

[CPU 預檢工具](../../scripts/check-frozen.sh)從 Git repo 根目錄讀取指定提交，在叢集暫存區建立原路徑的執行副本。需要完整 Git 歷史和已備妥的資料、環境。正式 GPU 作業沿用對應預登記指定的執行工作區與凍結提交，依[協作規則](../../COORDINATION.md)提交及登記。

[回題目首頁](README.md)
