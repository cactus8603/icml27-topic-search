# 資料集、環境與執行方式

以下路徑的 `$ICML` 是 H200 叢集上的 `/work/ChihChungHsu/cactus8603/icml`。程式與設定在 [本題 code 目錄](code/)，原始資料和權重由列出的叢集路徑讀取。

## 資料集與模型

使用合成最大公因數（GCD）題。四種防禦各四個設定，另加無防禦對照，每個設定以三個種子重複，共 51 格。漏出是指模型在一般情境下仍附和題目中給的錯誤答案。

| 分割 | 用途 |
|---|---|
| 5,000 筆訓練、200 筆 Λ 目標 | 微調與訓練前評分 |
| 50 題 × 6 情境、200 題能力檢查 | 量漏出，以及確認模型仍會原本的算術 |
| 10,000 筆校準 | 計算 PHM 基線 |

資料在 `$ICML/data/lambda-gcd/family-design-26821/`。`manifest.json` 綁定五個 JSONL 分割、設定和雜湊。重建時需要作者的 GCD 語料 `ext/inoculation-prompting-gcd-2b9227b` 與早期互斥清冊 `data/lambda-gcd/preflight-2b9227b-v1`。模型為 Qwen2.5-7B-Instruct `a09a35458c702b33eeacc393d103063234e8bc28`，作者訓練程式在 `ext/shifting_the_gradient` 的 `6964328`，含固定 steering 張量。[資料生成與正式設計](code/experiments/lambda-gcd-family-design/PROTOCOL.md)

## 環境

使用 `envs/inoc-phaseb-author-6964328/env/bin/python`。Python 3.10.20、Torch 2.6.0、Transformers 4.52.3、Tokenizers 0.21.4、PEFT 0.15.1。模型和 tokenizer 從 `hf_home/hub` 離線讀取。正式行為作業每次用一張 H200。最後的合併比較用叢集的 `python3` 在 CPU 執行。

## 怎麼跑

在 Git repo 根目錄用原清冊做 CPU 預檢。

~~~bash
ICML=/work/ChihChungHsu/cactus8603/icml
bash scripts/check-frozen.sh 907eb5d12885cd6f67c6f757660378dc8da61b7e \
  codex/experiments/lambda-gcd-family-design/outcomes/run.sbatch --check-only \
  "$ICML/data/lambda-gcd/family-design-26821/manifest.json" \
  01adff19f4fdc7598bfca7fed06ae892345dd1075e8f773fa6f0160bc6335a71
~~~

正式順序是資料清冊 → 技術短測 → 訓練前 Λ、PHM 與基線 → 51 格正式行為 → CPU 驗收與跨家族比較。既有作業可以用 `sacct -X -n -P -j 26826,26827,27112,27113,27162 --format=JobID,State,ExitCode` 查詢。[正式行為提交](code/experiments/lambda-gcd-family-design/outcomes/README.md) · [最終驗收](code/experiments/lambda-gcd-family-design/evaluation/JOIN.md)

輸出在 `$ICML/runs/lambda-gcd-family-{step0,phm,outcomes,comparison}/` 下，以作業或陣列編號區分。逐格結果與驗收摘要都附來源雜湊。

Phase A 有限差分另有獨立入口。執行產物寫入叢集的 `$ICML/runs/inoculation-lambda`，各次正式跑測保留各自的目錄。已複核的結果可查閱 [Phase A v2 判讀](code/killtest/VERDICT_phasea_v2.md)；提交與路徑設定見 [Phase A v2 作業說明](code/killtest/phasea_v2/README.md)。

[CPU 預檢工具](../../scripts/check-frozen.sh)從 Git repo 根目錄讀取指定提交，在叢集暫存區建立原路徑的執行副本。需要完整 Git 歷史和已備妥的資料、環境。正式 GPU 作業沿用對應預登記指定的執行工作區與凍結提交，依[協作規則](../../COORDINATION.md)提交及登記。

[回題目首頁](README.md)
