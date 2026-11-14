# ICLR 2027 升分與三會選題

2026-11-14，台北時間。選題以約十週可做成完整方法與實驗為準，優先看有公開資料、能用小規模實驗驗證的缺口。下列實驗和停損條件是提案，尚無實測結果。

## ICLR 2027 升分

比較 Radar 11/06 21:54:47 與 11/14 20:12:35 的快照。以兩期都有分、評審數相同的投稿計算均分差，依升幅排序，同升幅並列。當期有 37,514 篇有分，其中 37,115 篇符合比較條件，965 篇升分、154 篇降分。85 篇升至少 2/3 分，8 篇升至少 1 分。目前 1,905 篇均分至少 3.00/4，占 5.08%，45 篇均分至少 3.67（精確門檻 11/3），占 0.12%。公開討論到官方日程所列 11/18，決定預計 12/16 公布。 [固定快照](https://github.com/dipta007/iclr-2027-rebuttal-radar/tree/a67aefa2c4c2d231e471abab4c36cb771c672ec9/docs/data) · [比較方法](https://github.com/dipta007/iclr-2027-rebuttal-radar#how-it-is-counted) · [ICLR 時程](https://iclr.cc/Conferences/2027/Dates)

| OpenReview 投稿 | 初始 → 11/14 均分 | 升幅 |
|---|---:|---:|
| [#18295 OpenART](https://openreview.net/forum?id=ZPk8kVArJP) | 2.00 → 3.33 | +1.33 |
| [#3740 Focused Omni-Modal GUI Agents](https://openreview.net/forum?id=LaOUMzqRa7) | 2.33 → 3.33 | +1.00 |
| [#10419 From Knowing to Doing](https://openreview.net/forum?id=FKIWSV1kSL) | 2.33 → 3.33 | +1.00 |
| [#28436 Learning without Overwriting](https://openreview.net/forum?id=zQnQL2xLJ9) | 2.33 → 3.33 | +1.00 |
| [#30643 Feature Extraction Advantage of Graph Transformers](https://openreview.net/forum?id=UjLT8kHce8) | 2.20 → 3.20 | +1.00 |
| [#18010 FiRE](https://openreview.net/forum?id=CIn8XwY0G4) | 2.00 → 3.00 | +1.00 |
| [#19864 One Adapter, Every Resolution](https://openreview.net/forum?id=Tq9jZdZ0rt) | 2.00 → 3.00 | +1.00 |
| [#54570 Learning Environment Distributions](https://openreview.net/forum?id=CLNPWplLvB) | 2.00 → 3.00 | +1.00 |

三篇並列升幅第 9，分別是 [#36265 HapBO](https://openreview.net/forum?id=45k9l8zWbL)（3.00 → 3.75）、[#32446 Beyond Endpoints](https://openreview.net/forum?id=CVTsVBLRIi) 和 [#38679 Stabilizing Recursive Retrieval in Irregular Time-Series Forecasting](https://openreview.net/forum?id=zM7UJvW8dP)（兩篇均 2.25 → 3.00）。本表保留全站升幅名次，另附 [964 篇升分清單](iclr2027-rating-gains-2026-11-14.csv)。

Radar 只公開由高到低排序的分數。OpenReview 這次擋下公開評審紀錄的自動讀取，個別改分歷程和 reviewer 的理由未能核實。升分屬兩期差值。

高分另看 [#22968 Context Language Models](https://arxiv.org/abs/2609.37725) 和 [#45284 Latent Taxonomies](https://arxiv.org/html/2610.15836v1#A1)，均為 `4/4/4`。後者附錄點名原實驗的分類樹小、平衡、無例外，且合成事實曝光過密。可小規模試等 token 預算下，依實體覆蓋和分類例外挑合成知識，量未見實體的正確歸納率及例外誤推率。需要打過同曝光量的均勻與頻率加權抽樣。

## 優先候選

### 小批次下如何可靠地學資料權重

ICML 2026 Oral〈On the Difficulty of Learning a Meta-network for Training Data Selection〉表 2 顯示，batch 1024 平均準確率 73.75%，batch 64 降到 70.96%。小批次各自正規化選樣權重後再累積，一般無法重現共用分母的大批次更新。先固定模型和樣本，算精確共用分母的更新作參考，對照普通 gradient accumulation、精確兩遍式微批次、權重截斷或有效樣本數下限。只有分層抽樣或交叉估計在同反傳預算下改善超梯度誤差與下游損失，才擴到小型語言模型。一般的隨機超梯度估計與低成本資料策展已有人做，需把主張落在權重集中時的一步正規化估計。 [ICML Oral](https://icml.cc/virtual/2026/oral/71144) · [原文 §3／§5](https://arxiv.org/html/2606.00571v1#S3) · [SABA](https://arxiv.org/abs/2201.13409) · [LOOM](https://arxiv.org/abs/2610.00436) · [Your Hypergradient Is Skewed](https://arxiv.org/abs/2611.15895)

### 在整張感測網上縮小重要聚合量的預測區間

NeurIPS 2026 Oral〈Filtered Conformal Ellipsoids for Graph-Native Time Series〉在全圖 PEMS-BAY（325 個感測器）、相同圖感知骨幹下，GNF+ACI 平均座標投影寬度 9.01，factor baseline 為 5.64，聯合覆蓋率分別為 89.9% 和 90.1%。這個差距隨骨幹改變。先固定預測骨幹，再用低秩全域因子加圖局部殘差估計區間形狀，以近 90% 的相同聯合覆蓋比較事先指定、且未參與形狀訓練的路段聚合量。POET 已做低秩加稀疏協方差，SCALE 已用圖頻譜做校準。第一輪必須和 factor、copula、conditional POET 及直接校準聚合量比較。如果直接校準已達到相同覆蓋與寬度，這個題目就缺少方法貢獻。 [NeurIPS Oral](https://neurips.cc/virtual/2026/poster/155800) · [原文表 3 和附錄 E.9](https://arxiv.org/html/2606.17014v2#S6.SS4) · [程式](https://github.com/YannickLimmer/filter-cp) · [POET](https://arxiv.org/abs/1201.0175) · [Conditional POET](https://eprints.whiterose.ac.uk/id/eprint/167151/) · [SCALE](https://arxiv.org/abs/2605.04957)

### 資料內容影響延遲時，AdamW 的二階矩怎麼估

NeurIPS 2026 Oral〈Do We Need Asynchronous SGD?〉的分析假設 worker 梯度對同一目標無偏。資料相關延遲已有 ordered SGD 加 momentum 的解法，AdamW 二階矩是後者明列的缺口。先用兩至四個 worker 的實測長短批次延遲重播，對比同步 AdamW、固定 token packing、ordered SGD。若延遲確實令少數類樣本的二階矩估計失真，再測依派發次序調整更新的規則。比較同驗證損失的實際訓練時間。 [NeurIPS Oral](https://neurips.cc/virtual/2026/poster/155336) · [異步 SGD 原文](https://arxiv.org/html/2602.03802v2#S6) · [Bringing Order §6](https://arxiv.org/html/2605.02043v1#S6) · [固定一步延遲的 Muon](https://arxiv.org/abs/2606.30634)

### 保持 KV 預算，改善分塊壓縮最差位置的召回

ICLR 2027 #23443，均分 3.67。〈Periodic Weak Spots〉在 DeepSeek-V4 base 的 128K 檢索看到最高 40.2 個百分點的窗內位置差。原文輪替 gate 仍未救回邊界。可固定證據內容、長度和查詢距離，測跨 head 或 layer 錯開分塊邊界是否提高最差位置的召回率。相同 KV 大小下必比原方法與 gate 輪替，並量平均召回及延遲。優先確認這個差距能延伸到自然文件檢索，且小模型能跑出足以驗證的介入。 [OpenReview](https://openreview.net/forum?id=2dhoYgHiC2) · [原文](https://arxiv.org/pdf/2609.36322#page=10)

## 其他可接的缺口

- **多代理評論的漏檢率。** ICML 2026 Oral〈Benchmarking at the Edge of Comprehension〉以預算內未出現可驗證反例作評估。Spotlight〈When to Trust the Cheap Check〉已解強驗證可用時的路由，PP-Rank 已做自適應收集人類評分。可用機器能檢查的錯誤作校準錨點，估計多個相關評論模型的漏檢範圍。先固定答案，在可機器驗證的數學題確認是否能校準未見錯誤類型並減少排名反轉。 [ICML Oral](https://icml.cc/virtual/2026/oral/71031) · [原文 §9](https://arxiv.org/html/2602.14307v1#S9) · [ICML Spotlight](https://icml.cc/virtual/2026/poster/62754) · [PP-Rank](https://arxiv.org/abs/2611.13186)
- **參照模型也可能失準時的少標註校準。** NeurIPS 2026 Spotlight〈Calibration without Ground Truth〉在 LLaMA-3.1-8B 的 MMLU-Redux 上將 Brier 從 0.2450 降到 0.2153，監督式 temperature scaling 是 0.1874。原文附錄 B 已處理單一參照失準。可用多個可能失準的軟機率參照，僅在保守的改善下界為正時採用校準更新，用小批真標註稽核漂移。在相同 32 或 128 個標註下，必比最佳單參照與監督式校準，並做領域留出。 [NeurIPS Spotlight](https://neurips.cc/virtual/2026/poster/152525) · [原文](https://arxiv.org/html/2601.19862v1) · [多不可靠硬標籤校準](https://arxiv.org/abs/2610.19543) · [多提示線上重校準](https://arxiv.org/abs/2607.19689)
- **合成資料的類別內條件偏移。** ICLR 2026 Oral〈High-dimensional Analysis of Synthetic Data Selection〉的分析假設真實與合成資料共享條件標籤模型。後續已分析 model shift 的 SGD，以及同時有 covariance/model shift 的來源級加權 ridge。仍可測窄一點的問題，在同一個合成池內，不同子群的錯標程度不同時，能不能從少量真實標籤辨認哪些樣本值得留，以聯合改善對齊與標籤忠實度。先做合成分類與公開文字分類，對照 covariance matching、來源／群組加權和信心過濾。 [ICLR Oral 論文頁](https://rezaei-parham.github.io/Synthetic-Selection/) · [原文 §6](https://arxiv.org/html/2510.08123#S6) · [SGD 下的 model shift](https://arxiv.org/abs/2609.09572) · [多來源加權 ridge](https://arxiv.org/abs/2610.02578)
- **持續學習後，從不同正確中間步續解。** ICLR 2027 #28436 升 1.00。圖模型結果顯示從起點能解題，仍可能無法從罕見的正確中間狀態續解。可在小模型連續學可驗證的任務後，以固定預算選中間步做自蒸餾，測新題、舊題和替代解法的續解成功率。既有方法已做前綴蒸餾和舊狀態重播，需打過等預算的兩者。 [原文 §3.3](https://arxiv.org/html/2610.05200v1#S3.SS3) · [Prefix Distillation](https://aclanthology.org/2026.findings-acl.1276/) · [Teaching a Moving Student](https://arxiv.org/abs/2609.25048)
- **多輪分離量化已出現直接近作。** ICLR 2027 #30295 從 3.33 到 3.67，原文把同歷史重用與重建 KV 的差異列為未測。〈Same Request, Different Answer〉已驗證多輪 cache 差異及低位元放大效應，BridgeKV 已有跨模型翻譯，ReKV 有部分重建。只有能在相同延遲下，隔離並修正不同 prefill/decode 權重或精度帶來的額外品質差，才考慮投入。 [原文 §5](https://arxiv.org/html/2609.26333v1#S5) · [Cache divergence](https://arxiv.org/abs/2609.04748) · [BridgeKV](https://arxiv.org/abs/2611.11761) · [ReKV](https://arxiv.org/abs/2611.08770)

場次依 ICLR 論文頁、ICML 官網議程資料及 NeurIPS 官方議程的本機快取核對。ICML 與 NeurIPS 的原始審稿均分未能核實，這兩會用 Oral／Spotlight 當篩選依據。
