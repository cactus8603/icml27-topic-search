# 多條件檢索：強基線 × 資料集的基線矩陣計畫

[claude] 2026-11-16 18:08（台北）。這份是規劃，給 Codex 執行，沒有新的實驗結果。數字取自已 commit 的報告與論文原文，每個都附來源。

[題目首頁](../topics/compositional-retrieval/README.md) · [報告 62：3090 開發計畫](2026-11-16-62-retrieval-3090-dev-plan.md) · [報告 63：資源分配](2026-11-16-63-full-paper-resources-and-allocation.md) · [Codex 基線套件 v1](../codex/reports/2026-11-16-retrieval-baseline-suite-review-codex.md) · [協作規則](../COORDINATION.md)

## 目的

使用者 2026-11-16 的決定：

- 題目繼續做。現象、訊號、改善空間都已有證據；2B→8B 只回收 23% 的排序差距，8B／8B 仍剩 **15.13 pp**。[來源](../codex/reports/2026-11-16-core-scaling-headroom-codex.md)
- 叢集數字對不上不重要。要確認的是 **3090 上的基線本身是準的**。
- 136 題、50 候選太小，沒有說服力。
- 先建立「近年強方法 × 資料集」的基線矩陣，之後新方法才知道是更好還是更壞。

這份計畫回答四件事：用哪些方法、用哪些資料集、怎麼確認基線正確、執行順序。Codex 另外在做 SOTA 調查；兩份清單要合併，以較完整的為準。

## 原則

1. **基線正確性以論文自報數字為準，不以叢集為準。** 每個模型至少在一個它論文報過的 benchmark 上重現，誤差在 ±1 pp 內，才進矩陣。
2. **test 不能拿來開發，但可以照論文的方法測試**（使用者 2026-11-16 確認）。基線不調參，照各論文的方法在官方 test 上跑，和論文數字對照。我們自己的方法只能在開發資料上選設定與超參數，test 等最後實際算分數時才跑，只跑一次。
3. **沒有公開排行榜。** 組合式的文字找圖目前沒有評分伺服器或排行榜；各論文都是在作者釋出的 test 集上自報分數。我們照同樣做法，並把分數和論文表格對照。
4. **重排主設定 K＝100**（使用者 2026-11-16 定案），和 Qwen3-VL-Reranker（K＝100，自 Qwen3-VL-Emb-2B 取）、CORE 開發集（K＝100）一致；BLIP、mPLUG 用 COCO 256／Flickr 128，MCMR 用 50。K＝50、200 做為深度變化實驗。[CORE](https://arxiv.org/html/2609.04083) · [Qwen3-VL](https://arxiv.org/html/2601.04720v2) · [MCMR](https://arxiv.org/html/2603.01082) · [BLIP](https://arxiv.org/pdf/2201.12086)
5. **每個資料集單獨報，不做跨集平均。** 指標不同的不合併。（沿用基線套件 v1 的規定）

## 方法清單

### A. 第一階段 embedding（全部只跑推論）

| 方法 | 規模 | 為什麼要 | 程式／權重 | 3090 |
|---|---|---|---|---|
| CORE-Embed | 2B、8B | 目前組合式最強的 MLLM embedding（Table 2 總平均 0.641／0.666） | `Alibaba-NLP/core-emb` | 可（8B 實測 17.1 GiB） |
| Qwen3-VL-Embedding | 2B、8B | MMEB-V2 第一（8B 為 77.8）；CORE 的基底 | HF `Qwen/Qwen3-VL-Embedding-*` | 可 |
| GME | 2B、7B | 非 CORE 系的 MLLM embedding | `Alibaba-NLP/gme-Qwen2-VL-*` | 可 |
| VLM2Vec-V2.0 | 2B | 另一個訓練路線，MCMR R@1 0.288 | TIGER-Lab | 可 |
| UniME | 2B、7B | CORE Table 2／5 有報 | 公開 | 可 |
| SigLIP2 so400m | ~1B | 最強的通用雙塔 | `google/siglip2-*` | 可 |
| CS-CLIP | B/32 | 組合式 CLIP（NeurIPS 2026），Half-Truths 作者 | `kbora/CS-CLIP` | 可 |
| CLIC | B/32、L/14 | 組合式 CLIP（NeurIPS 2025） | `nmndeep/CLIC-*` | 可 |
| NegCLIP、TripletCLIP | B/32 | 否定／硬負例訓練的經典對照 | 公開 | 可 |

CORE 與 Qwen3-VL 共用 Qwen backbone，不算兩種獨立架構（[基線套件審查](../codex/reports/2026-11-16-retrieval-baseline-suite-review-codex.md)）。所以 GME、VLM2Vec、SigLIP2、CLIP 系至少要各有一個。

### B. 重排器（固定第一階段，取前 K）

| 方法 | 規模 | 備註 |
|---|---|---|
| CORE-Reranker | 2B、8B | 論文只在選擇題評估，**沒有在圖庫檢索測過**（Table 1 註）。Codex 量到它在 NegBench 否定檢索 MRR −9.16 pp，可能是論文沒覆蓋的情境，不一定是 bug，但要先排除模板錯誤 |
| Qwen3-VL-Reranker | 2B、8B | 官方重排器 |
| jina-reranker-m0 | 2B | CORE Table 1 的對照（總平均 0.720） |
| 通用 MLLM 逐點評分 | Qwen3-VL-8B-Instruct | 問「這張圖是否符合描述」取 yes 機率。MCMR 論文報告 MLLM 逐點重排能明顯改善 |

第一階段固定用兩個：最強的 MLLM embedding（依 P1 結果，預計 CORE-Embed-2B 或 Qwen3-VL-Emb-8B）和一個 CLIP 系，避免重排結論綁定單一召回模型。

### C. 不需訓練的組合式修正

| 方法 | 狀態 | 做法 |
|---|---|---|
| LCSE（ICML 2026） | 作者有程式；Codex 已重實作，校準流程和作者不同 | 依作者協定重跑；和 Codex 的版本分開標示 |
| ConstraintMatch（2611.00784） | 不需訓練，跑在 CORE-2B／Qwen-2B 上，**沒有公開程式** | 先只在它報過的 ARO、SugarCrepe++ 上對照論文數字；重實作列為選做 |
| Flow-CLIP（2610.21937） | 需要訓練，**沒有公開程式** | 只引用，不進矩陣 |

## 資料集

### 主表：圖庫檢索（能回答「多條件查詢能不能排對」）

| 資料集 | 條件型態 | 官方 test | 開發資料從哪來 | 指標 | 論文對照數字 |
|---|---|---|---|---|---|
| **COLA 單物件**（GQA、PACO） | 單物件、多屬性綁定 | GQA 320 題／1,952 圖；PACO 400 題／7,921 圖；分見過／未見過 | COLA 官方 train 切出一份 dev | 困難負例上的 mAP | FLAVA＋MM-Adapter：GQA 48.54、PACO 19.36 [COLA](https://arxiv.org/html/2305.03689v3) |
| **NegBench COCO 否定檢索** | 含否定的描述 | COCO val2017 5,000 圖、25,014 句 | **見下方「必須先解決」** | R@5（CORE 論文用）、MRR | CORE-Emb-2B R@5 **0.766**、8B **0.790** |
| **MCMR** | 商品的視覺＋文字多條件 | 只有 test（3,997 題，候選數以鎖版檔為準） | 無；只當跨領域 test | R@1、MRR | CORE-Emb-2B R@1 **0.306**、MRR **0.397** |
| **COCO 物件連言面板（自建，已建好）** | k＝2–4 個物件類別同時出現 | train2017 的 2 萬張圖庫，1,122 題，排除已曝光影像 | train2017 另一個不重疊的 2 萬張圖庫：dev 1,127 題、dev-small 252 題 | 全庫 mAP、Recall@K | 無（自建）。val2017 只有 5,000 張，只能得到約 520 個獨立群，所以改用 train2017。[規格](../topics/compositional-retrieval/code/experiments/retrieval-conjunction-panel/SPEC.md) · [產生紀錄](../topics/compositional-retrieval/code/experiments/retrieval-conjunction-panel/MANIFEST.md) |

自建面板是為了補足其他 benchmark 沒有的「完整多正例＋條件數 k 可控」。目標 **≥1,000 個獨立查詢群**：現有 110 群時最小可偵測效應約 5.96 pp，偵測 2 pp 約需 1,000 群。[來源](../codex/reports/2026-11-16-retrieval-experiments-codex.md)

### 開發資料分兩層

縮的是查詢數，不縮圖庫。圖庫的 embedding 建一次就能快取，成本主要在重排（查詢數 × K）；縮圖庫會少掉干擾圖，讓 mAP 整體偏高，小版的結論可能搬不到大版。

| 層級 | 查詢 | 圖庫 | 用途 |
|---|---|---|---|
| **dev-small** | 約 200–300 群 | 完整 | 除錯、快速試想法；只看大效應（約 10 pp 以上） |
| **dev-full** | ≥1,000 群 | 完整 | 決定方法是否繼續；檢定力足以看到約 2 pp |
| test | 官方 | 完整 | 最後算分數，只跑一次 |

- dev-small 是 dev-full 的子集，看任何結果之前用固定 hash 抽定。
- 正式判斷一律用 dev-full。
- 估計一輪 K＝50 的 2B 重排：dev-small 約 20 分鐘，dev-full 約 2 小時。

### 配對診斷（不是圖庫檢索，報分但不當主結論）

| 資料集 | 內容 | 對照數字（CORE-Emb-2B／Reranker-2B） |
|---|---|---|
| COLA 多物件 | 兩物件屬性互換，四選對 | 0.462／0.724 |
| SugarCrepe++ | replace／swap，五種擾動 | 平均 0.696／0.839 |
| NegBench MCQ | 否定選擇題 | 0.531／0.646 |
| FACTOR-Bench | 1,695 對邏輯運算子 | 見 LCSE 論文 |
| Half-Truths COR | 同圖三句排序 | 見 Half-Truths 論文 |

### 一般檢索不退步

COCO 5k、Flickr30K 1k 的官方 test，文字找圖與圖找文字的 R@5、R@10。CORE 論文用的是 NegBench 提供的 COCO 版本，重現時要用同一版。

## 必須先解決：NegBench 的開發與測試重疊

Codex 的開發集 NegBench dev1000 是從同一份 COCO val2017 否定檢索檔抽出的 1,000 張圖。[來源](../codex/reports/2026-11-16-retrieval-scale-data-survey-codex.md) 基線套件 v1 同時把「完整 negated」列為評估格。如果在 dev1000 上選方法，再報完整 NegBench 的分數，測試集就被開發污染了。

**決定（依「test 不能拿來開發」）：** NegBench 官方檔只當 test，不再用 dev1000 開發。否定條件的開發資料改由 COCO train2017 自建，影像與官方檔不重疊；自建否定描述要抽查品質。

到目前為止，dev1000 上只跑過基線（cheap 索引與 CORE50），沒有用來選我們的方法，所以官方檔仍可當 test。論文要寫明這 1,000 張圖曾用於基線量測。

## 基線正確性關卡（P0）

用 CORE 論文的 Table 2、Table 5 當目標。只需要 embedding 推論，成本低，又同時驗證評分程式和 3090 平台。

| 模型 | COCO T→I R@5 | Flickr T→I R@5 | MCMR R@1／MRR | NegBench 否定 R@5 |
|---|---|---|---|---|
| CORE-Emb-2B | 0.793 | 0.962 | 0.306／0.397 | 0.766 |
| CORE-Emb-8B | 0.820 | 0.970 | 0.412／0.506 | 0.790 |
| Qwen3-VL-Emb-2B | 0.754 | 0.944 | 0.273／0.360 | 0.668 |
| Qwen3-VL-Emb-8B | 0.785 | 0.947 | 0.375／0.469 | 0.723 |
| GME-2B | 0.758 | 0.930 | 0.139／0.199 | 0.658 |
| VLM2Vec-V2.0 | 0.716 | 0.926 | 0.288／0.369 | 0.682 |

來源：[CORE 論文 Table 2、5](https://arxiv.org/html/2609.04083)。

- **通過**：每個模型至少兩欄在 ±1 pp 內。
- **不通過**：先查模板、指令前綴、影像解析度、正規化，和作者程式逐項比；查完仍不符就照實記錄差距，不調到對上為止。
- CLIP 系用各自論文的 COCO／Flickr 數字對照。
- 重排器沒有圖庫檢索的論文數字，改用選擇題對照（COLA 多物件、SugarCrepe++、NegBench MCQ，見上表）。

## 執行順序

| 階段 | 內容 | 驗收 |
|---|---|---|
| **P0** 基線正確性 | 六個 embedding 跑上表四個 benchmark；重排器跑三個選擇題 benchmark | 依上一節的通過標準；每格附模型版本、模板、batch、環境收據 |
| **P1** 資料 | 下載 COLA（含官方 train，切 dev）；從 train2017 建否定句開發集；從 train2017／val2017 建自建面板；抽定 dev-small ⊂ dev-full。建法和群定義先凍結，再產生 | 每份資料記 SHA、題數、群數；開發與測試影像 ID 不重疊的檢查 |
| **P2** embedding 矩陣 | 方法 A 全部 × 主表與一般檢索資料集 | 和 P0 同一套程式；結果由非實作者獨立重算 |
| **P3** 重排矩陣 | 方法 B × 兩個第一階段 × K＝100；K＝50、200 的深度變化實驗先在自建面板做 | 計入實際成本（秒／題、峰值 VRAM），50 次重排不等於相同算力 |
| **P4** 彙整 | 每個資料集標出最強基線組合，作為新方法的比較對象 | 表格只放已通過 P0 的模型 |

P0 不通過的模型不進 P2。P3 依 P2 選第一階段，不另外看測試結果挑。

## 算力估算（3090，依已實測吞吐推算，非保證）

- **embedding**：8B 建 8,191 圖的索引約 21 分鐘。[來源](../codex/reports/2026-11-16-core-scaling-headroom-codex.md) 六個模型 × 四個資料集，估計一到兩天，可分卡平行。
- **重排**：實測 2B 約 0.074 秒／對、8B 約 0.17 秒／對（批次攤提，[來源](../codex/reports/2026-11-16-core-scaling-review-codex.md)）。1,000 題 × K＝100＝10 萬對，2B 約 2 小時、8B 約 5 小時。四個重排器 × 兩個第一階段 × 兩個 K × 四個資料集，總量約 **100–200 GPU 小時**，要排優先順序：先自建面板與 NegBench，COLA 和 MCMR 其次。
- 8B 模型峰值約 17–18 GiB，要等 ≥18 GiB 空閒的卡；用 `codex/retrieval3090/resources.py` 排隊。

## 已定與待定

已定（使用者 2026-11-16）：

- test 不能拿來開發；照論文方法測試可以。NegBench 官方檔只當 test。
- 方法名單：和 Codex 的 SOTA 清單合併後，由 Claude 審定，Codex 執行。
- 開發資料分 dev-small／dev-full 兩層。

- 只用已有人工標註的資料，不另做人工標註（使用者一人）。封存的獨立面板沒有人工核對、影像也未取得，暫不做；先前規劃的人工 gold 標籤與錯例人工核對也取消。
- 重排主設定 K＝100；K＝50、200 做深度變化實驗。
- Codex 先前切的開發資料（如 NegBench dev1000）可以重切。
- 自建面板只做物件連言（k＝2–4），已依凍結規格建好。屬性綁定用 COLA；空間關係用 SugarCrepe++ relation 子集與左右面板，和其他論文一樣只在配對任務上評估。

尚未完成：dev／test 圖庫的 40,000 張 train2017 影像下載。
