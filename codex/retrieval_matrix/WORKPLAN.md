# [codex] 報告 64 執行與驗收

[codex] 2026-11-16 20:04（台北）。使用者要求完成截圖所列事項，依 `reports/2026-11-16-64-retrieval-baseline-matrix-plan.md` 執行；可並行部分使用子代理。

## 範圍與完成證據

| 階段 | 必須完成的內容 | 完成所需證據 | 目前狀態 |
|---|---|---|---|
| P0 | 六個 embedding 在 COCO、Flickr、MCMR、NegBench 的官方重現；重排器配對任務驗收 | 固定官方來源／資料／設定、完整推論收據、逐題輸出、獨立重算、論文值差異表 | 首批 CORE/Qwen3 各2B/8B × COCO/NegBench/Flickr 共12格已固定；26 CPU tests與獨立協定審查通過，待GPU |
| P1 | COLA、train2017 否定開發資料、自建連言面板；dev-small／dev-full 與 test 界線 | 下載雜湊、圖片覆蓋、固定產生規則、分割檢查 | 連言40,000圖、Flickr1000圖/5000句、SugarCrepe++4757題、COLA10623圖已取得；MCMR全量123561有效圖已解壓，所需圖零缺；COLA官方train未公開、否定dev尚待建立 |
| P2 | 報告 64 方法 A 的 embedding × 主資料與一般檢索矩陣 | P0 資格、模型／資料固定版本、獨立驗收的每格結果 | 尚未開始 |
| P3 | 方法 B × 兩個第一階段，主 K=100；50／200 深度實驗 | 固定候選協定、完整結果與實測成本 | 尚未開始 |
| P4 | 按資料集彙整最強基線、來源與限制 | 可追溯結果表及獨立審查 | 尚未開始 |

尚未執行、來源不可得、技術失敗、重現偏差均分開記錄，缺格不填零，也不視為完成。

## 分工與檔案所有權

| 負責者 | 任務 | 可寫範圍 |
|---|---|---|
| root | 統籌、預登記、資料適配、版本凍結、排隊、結果整合與驗收派工 | 本檔、PREREG、plan builder、launcher、整合報告；協作 log 依 lock 規則 |
| official_p0 | 官方論文／程式／權重／協定、缺少模型取得 | `official_protocols.json`、`official_sources.md`、`acquisition.py`；`dl/retrieval-matrix/sources/`、釘版模型快取 |
| public_data | COLA、SugarCrepe++、Flickr30K 取得與覆蓋驗收 | `data_acquisition.py`、`test_data_acquisition.py`、`data_sources.json`、`data_notes.md`；`dl/retrieval-matrix/data/` |
| matrix_runner | 模型包裝、可續算 runner 與 CPU 指標 | `runner.py`、`models.py`、`metrics.py` 與對應 tests |
| 後續獨立審查者 | 官方設定核對、程式與真實輸出獨立重算 | 新增獨立審查報告，不改實作者程式與結果 |

首批計畫為 `p0-qwen3-batch1.json`；來源、資料、模型 payload、2,734 個隔離依賴檔已核對。套件為原 torch2.14 環境加隔離 Transformers4.57.3，不修改舊環境。只在真正 synthetic GPU smoke 及數字重現完成後，才可將相應狀態列為通過。

平行後續模組 `codex/retrieval_matrix_qwen2/` 與 `codex/retrieval_matrix_rerank/` 分別處理 GME/VLM2Vec 的舊API與DoRA、CORE ms-swift MCQ及其他重排器，不能改首批已凍結的執行程式。

## 執行規則

- 開發只讀自建面板 dev／dev-small。test 僅允許事前固定的基線推論或最後凍結自家方法的單次評估。
- P0 依官方 test 與官方模型協定，不按結果調參以追數字。
- 主重排 K=100；50、200 分列。官方原生 K 不同時，官方重現與統一 K 的適配結果分列。
- 不改舊 suite v1 或舊結果；本模組承接新版規劃。NegBench 官方檔停止作自家方法開發用途。
- MCMR 128 題 pilot 曾測自家方法，後續必須保留曝光揭露，不能把整份 MCMR 稱未見資料。
- 連言面板的正例互斥不等於無條件統計獨立；1,000 題量級不保證 2 pp 檢定力。
- gold／ignore 只能計分。完整影像圖庫先取 K，評分時排除 ignore，不為 ignored 候選補足 K，實際 cross 數照實記。
- 本機七張 3090 共用，使用既有 `resources.py queue` 等空卡；GPU 動作由 root 登記，不干預其他工作。
- 固定判準與執行版本 commit 後才產生或讀取主結果；結果由非實作者獨立核對。

## 當前環境

- 本機 `acvlab`；2026-11-16 20:00 實查七張 RTX 3090 皆繁忙。
- `/ssd4` 約 1.1 TiB 可用；本機無叢集 quota，依 CLAUDE.md 本機適用規則。
- 沿用既有 `dl/retrieval3090/envs/retrieval3090-torch214`，官方要求不同時另設隔離環境並記錄差異。
- 模型、原始資料、向量快取留在 ignored `dl/`／`hf_home/`；暫存放 repo 的 `hf_home/tmp`。
