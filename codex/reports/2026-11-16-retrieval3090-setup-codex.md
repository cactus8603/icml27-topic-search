# 3090 檢索開發環境與資料驗收

[codex] 2026-11-16 02:58（台北）。依使用者要求，3090 做小規模方法開發與驗證；大顯存計算、訓練與正式評估回叢集執行。

## 完成狀態

已從 `origin/main` 拉到 `a207d8912ebfccda5ddb8107525d0e9e4d74b93c` 的 dev-3090 規範與程式，建立本機相容環境，取得三個固定模型，並從公開資料精確重建原開發圖庫。完整 CPU 預檢成功退出。未修改三個原暫停／淘汰題的內容或狀態。

## 環境

| 項目 | 本機版本 |
|---|---|
| GPU／驅動 | RTX 3090 24 GB／550.163.01；短測使用索引 3 |
| Python | 3.12.14，專用 venv |
| Torch／torchvision | 2.6.0+cu124／0.21.0+cu124 |
| Transformers／open_clip_torch | 5.20.2／2.32.0 |
| NumPy／Pillow | 2.3.5／11.3.0 |

原環境使用 CUDA 13，不符合本機驅動，因此改用 CUDA 12.4 的 Torch；沒有變更系統驅動。完整套件清單在 `codex/retrieval3090/requirements-lock.txt`，`pip check` 無依賴衝突。三個 Qwen3-VL／CLIP 模型可匯入並實際執行。

本機 Python 位於 `dl/retrieval3090/envs/retrieval3090/bin/python`；由既有 Python 3.12 建立獨立 venv，不修改原有其他專案環境。資料根目錄為 `dl/retrieval3090/`，模型快取與暫存為 `hf_home/`。環境、權重、圖片及快取不進 Git。

## 固定模型與精確資料

三個模型分別固定為：

- CS-CLIP：`afd6e90f39fe49137fe15d6a462d994eae4811f5`。
- CORE-EMBED：`cc9870ea92d64ebd5f8f9fc368d1120453564158`。
- CORE-RERANKER：`64e1a36cd6ecace459a4c7cd8f7d0ef5d6fee30b`。

21 份公開輸入檔案的大小及 LFS SHA-256／Git blob SHA-1 已核對；收據為本機 `dl/retrieval3090/public-download-receipt.json`。COCO 標註 ZIP 的 SHA-256 符合原值 `113a836d90195ee1f884e704da6304dfaaecff1f023f49b6ca93c4aaae470268`。

沒有叢集連線，因此從固定 Git 提交 `2daedf522c6d64b860277657d3067b61da3a0f2b` 的函式重建原歷史排除清單、查詢及圖片資料。原 E2 查詢、完整 COCO val 5,000 張的指紋集合、三份 Half-Truths parquet，以及三代 manifest 均通過原始雜湊。

| manifest | 原 SHA-256，且本機重建完全相同 |
|---|---|
| conjunction | `9bd8a7e4bb167a84ae3e6b6836a248d9e8a92a28c51a3160ab3e599ed7e2d6cd` |
| rerank | `df51d103caa1ec7968d46254b5757789e7f834e1f3a11dae1105b2f8c63ccd3f` |
| 開發 anchor-rerank | `8f43eedbcef31fe5057c26705e97bf9f213f162612d9a15b53da11bb9bacf78d` |

開發目錄有 8,192 張原始 JPEG，原去重後為 **8,191 張有效圖、136 條查詢、118 群**，105 條查詢只有一個正例。包含歷史面板，JPEG 共保留 4,019,526,802 bytes。固定重建程序及下載進度在 `dl/retrieval3090/rebuild-panel-status.json`。

## 驗收

- 原分析工具 7 項合成與凍結評分器一致性測試通過。
- 重建器 4 項完整性自測及獨立只讀程式審查通過，實際完整重建成功。
- CUDA／BF16 小矩陣運算通過，見 `dl/retrieval3090/cuda-check.json`。
- 三個模型各以兩張純色圖片及兩張真實 COCO 圖片短測通過。真實圖片測試的向量形狀為 CORE 2×2048、CS-CLIP 2×512，重排器輸出兩個有限分數。三階段已保留顯存峰值依序為 4.105、0.629、4.111 GiB。
- `bash codex/retrieval3090/run.sh preflight` 成功退出：逐圖 SHA、原 manifest、模型設定及 revision、8,191／136／118 數量均通過，全部相關標註可由 COCO 類別規則重建；條件文字共 76 條。紀錄為 `dl/retrieval3090/preflight-20261116.json`。

真實圖片短測的輸出為 `dl/retrieval3090/smoke-real-images-20261116.json`，執行時工具提交為 `32cd459d7e781ea7d4bf13dee00f01dc7c39aa25` 且工作區程式乾淨。這些數字只代表技術短測，不代表原 batch、全圖庫吞吐或研究效果。

## 來源與後續界線

原 `26601/summary.json` 是未公開的叢集實測收據，本機無法還原，也沒有補造。本機入口明確記錄 `source_receipt_available=false`、`source_summary_sha256=null`、原收據預期 SHA 及本機重建來源；若日後取得原件，則必須通過原始收據校驗。

尚未執行全圖庫 GPU 快取、完整基線重現或 M1／M2 分析。完整快取仍保留原 mAP／Recall／無正例數的容差與失敗即停止條件。本次未開啟或下載獨立封存面板，沒有提交叢集作業或進行訓練。

操作方式見 [本機開發說明](../retrieval3090/README.md)。
