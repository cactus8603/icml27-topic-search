# 物件連言檢索面板（dev／test）建置規格

[claude] 2026-11-16 台北。使用者同意此設計後凍結：先 commit 本規格與 `build_panel.py`，再產生資料。產生後不改規則；要改就開新版本（v2），不覆寫 v1。計畫背景見[報告 64](../../../../../reports/2026-11-16-64-retrieval-baseline-matrix-plan.md)。

## 目的

建立能做「完整多正例圖庫檢索」的多條件查詢面板：查詢要求 k 個物件類別同時出現。COCO 對 80 類物件是逐一標完的人工標註，所以每題的正例、負例都能從標註完整判定，不需要新的人工標註。

屬性綁定與空間關係不在本面板：前者用 COLA，後者用 SugarCrepe++ relation 子集與左右面板（理由見報告 64）。

## 來源

- 標註：COCO 2017 `annotations_trainval2017.zip`，SHA-256 `113a836d90195ee1f884e704da6304dfaaecff1f023f49b6ca93c4aaae470268`（與 TFDS 公布值相同）。只用 `instances_train2017.json`。
- 影像：COCO train2017。只下載被選入圖庫的影像，另行記錄逐檔 SHA。
- val2017 不用：只有 5,000 張，k 均衡且正例不重疊時只能得到約 520 個獨立群，不足 1,000。val2017 保留給 NegBench、COCO 5k 等公開 benchmark。

## 已曝光影像（test 圖庫必須排除）

以下 train2017 影像曾用於開發，`build_panel.py exposed` 取聯集後寫成 `exposed.json`，記錄每個來源的 SHA 或 ID 清單摘要：

| 來源 | 內容 | train2017 影像數 |
|---|---|---|
| `coco2017-anchor-rerank` 影像與 manifest | 8,191 圖開發圖庫（M1、D1–D3） | 8,192（含 1 張被排除的重複圖） |
| `coco2017-rank-and` 影像 | 舊面板 | 12,288 |
| `coco2017-rerank` 影像 | 舊面板 | 4,096 |
| `lcse-calibration-20261116/calibration_design.json` | LCSE 校準選用 512 張與其排除清單 | 512＋12,286 |
| `d3-nested-20261116/design.json` | D3 圖庫 | 8,191（與第一列相同） |
| Half-Truths `coco_qwen.parquet`、`coco_mistral.parquet`（revision `ddcd448c…`） | `source_image_id` 的 COCO ID | 789 |

前五列聯集為 24,576 張。封存的獨立面板（8,192 張 train2017）從未用於開發，使用者已決定暫不做，不列入排除。

## 建置規則

1. **影像狀態**（每張圖、每個類別）：
   - **出現**：至少一個非 crowd 實例，分割面積 ≥ 1% 影像面積（τ＝0.01）。
   - **確定沒有**：該類別在此圖沒有任何標註。
   - **忽略**：有標註，但只有小於 τ 的實例或 crowd。
2. **兩個圖庫**：train2017 全部影像依 `sha256("conj-panel-2026111601:pool:<id>")` 排序。
   - **test 圖庫**：依序取前 20,000 張**不在已曝光清單**的影像。
   - **dev 圖庫**：其餘影像依同一順序取前 20,000 張（已曝光影像可在 dev 側）。
   - 兩者不重疊；程式內有 assert。
3. **候選查詢**（每個圖庫分開）：k ∈ {2, 3, 4} 個類別的組合，條件：
   - 正例（k 類全部「出現」）1–20 張；K＝100 一定裝得下。
   - 困難負例 ≥ 5 張：其中 k−1 類出現、剩下那一類「確定沒有」。
4. **選題**：各 k 的候選依 `sha256(seed:split:k:組合)` 排序，k＝2、3、4 輪流取題；**所有已選題目的正例互不重疊**，取到候選用完為止。每題因此是獨立群。
5. **忽略集**：某題的忽略集＝不是正例、沒有任何條件「確定沒有」、但至少一個條件是「忽略」的圖。評估該題時從圖庫移除這些圖，既不算正例也不算負例。
6. **查詢文字**：單一模板 `A photo that contains a dog, a cat, and an egg.`；類別順序依固定雜湊打散，名稱用 COCO 原名加冠詞。各模型需要的指令前綴由模型包裝程式加，不改查詢本文。
7. **dev-small**：自 dev-full 依 `sha256("conj-panel-dev-small-2026111601:<query id>")` 排序，k＝2、3、4 各取前 84 題，共 252 題。

可行性（CPU 估算，只讀標註）：20,000 張圖庫約可得 1,127 題（k＝2／3／4 約 339／409／379）。實際數字以產生結果為準。

## 輸出

`dl/claude-panel/conj-panel-v1/`（大檔不進版控；manifest 與 SHA 回寫到本資料夾的 `MANIFEST.md`）：

- `exposed.json`：排除來源與 ID 聯集。
- `manifest.json`：所有參數、輸入與輸出 SHA、各 split 題數。
- `dev/gallery.json`、`dev/queries.jsonl`、`dev/dev_small.json`。
- `test/gallery.json`、`test/queries.jsonl`。

每題欄位：`id`、`k`、`category_ids`、`names`、`text`、`positives`、`ignore`、`n_hard_negatives`、`group`。

## 使用規則

- **開發只用 dev**：dev-small 用於除錯與快速篩選，正式判斷用 dev-full。
- **test 不能拿來開發。** 建好後只記錄題數與 SHA；除基線照論文方法量測外，我們的方法要等最後算分數才在 test 上跑，且只跑一次。
- **評估約定**：
  - 每題對「圖庫減去忽略集」做完整排序，算 mAP，分母為全部正例。
  - 另報 Recall@K。
  - 重排主設定 K＝100；K＝50、200 為深度變化實驗。
  - 不確定性以題為單位做 bootstrap，因為每題就是一個獨立群。

## 已知限制

- 只測物件連言，不含屬性、關係、否定。
- COCO 標註仍有漏標與錯標；τ 是人為門檻，小物件標為忽略而非負例。
- 正例不重疊的要求讓多數題目只有 1–2 個正例。
- 單一查詢模板；模板敏感度若要測，另開實驗。
- 選題偏向較少見的類別組合，因為常見組合的正例常超過 20 張。
