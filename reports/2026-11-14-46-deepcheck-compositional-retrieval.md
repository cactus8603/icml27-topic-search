# 深查：多條件／組合式圖文檢索的「正確細節升、錯誤細節降」（外部分析候選）

日期：2026-11-14。範圍：只查文獻、授權與 ICML 2026 會場資料，沒有用 GPU，也沒有寫實驗程式。
網路請求約 38 次：arXiv PDF 5 篇、arXiv API 約 12 次、Semantic Scholar 7 次（含 429 重試）、HuggingFace／GitHub API 約 10 次、網路搜尋 3 次。
來源層級標記：〔全文〕下載 PDF 抽文字後讀了相關段落並做關鍵字搜尋，不是逐頁精讀；〔摘要〕只讀 arXiv 摘要；〔搜尋摘要〕只看到搜尋結果；〔API〕HuggingFace／GitHub／S2 的 API 回傳；〔本機〕本機 icml26.json；〔推估〕我自己的估計。

## 結論

**信心：中偏低。** 照外部分析的寫法（量「正確細節升、錯誤細節降」能否同時成立，再把錯誤拆成辨識、綁定、聚合），核心量尺和主要方法都已經有人發表：

1. 「正確補充 ＞ 原描述 ＞ 錯誤補充」這個量尺，Half-Truths／CS-CLIP（NeurIPS 2026）已經定義成 Correct Ordering Rate（COR），也已公開含影像的人工核驗評估資料。外部分析寫的 58.2／50.4／42.2 就是這篇的數字。
2. 「部分條件成立被當成全部成立」：CORE 的五級分類裡就有「L4 Partial Presence」，並用 Rank-KL 依 L5 ＞ L4 ＞ L3 ＞ L2 ＞ L1 做蒸餾訓練。GradCIR（2609.24152）也用四級分級相關性訓練。所以「依符合程度排序」已經有人當訓練目標。
3. 「錯在聚合，不在表徵」：LCSE（ICML 2026）已經做了這個拆解與 training-free 的修正。綁定那一層有 ICML 2026 spotlight（2605.31503）和 2608.15971 的理論。

剩下的空間比較窄，但具體存在：
- 目前沒有人說明「正確細節要升、錯誤細節要降」對評分函數的要求是什麼，也沒有人證明單向量內積（固定索引）在條件數 k 增加時一定會把「部分符合」與「全部符合」混在一起。
- LCSE 的 AND 是 power mean（約等於 min）。依它的式子推論，正確細節不會讓分數上升；沒有偵測到運算子時修正項為 0，所以對 half-truth 類句子不起作用。〔全文，推論〕

若要把信心推到「中」，需要兩件事：(a) 一個能寫成定理的主張（見第 5 節的聚合公理與線性評分的分離命題），(b) 第一週的便宜實驗證明 training-free 的對數概似比聚合在 Half-Truths 上的 COR 明顯超過 CS-CLIP。兩件都成立，我估計可到「中」；要到「中偏高」，還需要第二週的訓練版在 MCMR 或 SugarCrepe++ 上不輸 CLIC／CORE。目前沒有證據支持中偏高。

另外提醒：使用者先前說過「電腦視覺方向可以停掉，專心 ICML 的方向」（criteria.md）。這題本質是圖文檢索，只有用理論與表示學習的框架寫，才比較像 ICML 題目。

---

## 1. 外部分析的說法逐條核實

| 外部說法 | 核實結果 | 來源層級 |
|---|---|---|
| CORE：SugarCrepe++ 上 embedding 74.5%、reranker 87.5% | **正確。** CORE = arXiv 2609.04083（2026-09-03，CASIA／阿里巴巴）。Table 2：CORE-EMBED-8B 的 SugarCrepe++ Avg = 0.745（Replace 0.818、Swap 0.636）；Table 1：CORE-RERANKER-8B = 0.875。這裡的 Avg 是五個子集的平均，不是 Replace 與 Swap 兩組的平均；按兩組平均算，CORE-EMBED-8B 是 (0.818+0.636)/2 = 72.7%〔推估，自行計算〕。 | 全文 |
| CORE 沒納入 CLIC | **正確。** CORE 全文搜不到 "CLIC"。CLIP 系基線只有 SigLIP2、NegCLIP、Triplet-CLIP。 | 全文 |
| CLIC（NeurIPS 2025）ViT-L/14 依 Replace／Swap 分組重算約 81.0% | **數字正確，但不必重算，原表就有。** CLIC = arXiv 2505.24424，NeurIPS 2025。Table 5：「CLIPS + CLIC」（427M，CLIPS ViT-L/14 規模）SugarCrepe++ ITT 的 Replace 84.9、Swap 75.1、Average 81.0。注意：這是以 CLIPS 為底的版本；以 OpenAI CLIP ViT-L/14 為底的 CLIC-LAION 是 Replace 79.0、Swap 59.8（Table 3）。另外兩篇的評測協定（ITT 的定義、子集平均方式）是否一致，我沒有逐項對齊；CORE 表中 NegCLIP 0.486、SigLIP2 0.274 偏低，可能是協定不同所致〔推估〕。 | 全文 |
| CS-CLIP（稱 NeurIPS 2026）ViT-B/32 三組人工核驗資料上「正確補充 ＞ 原描述 ＞ 錯誤補充」比例 58.2%、50.4%、42.2% | **數字正確，但論文名稱要更正。** 出處是 *Half-Truths Break Similarity-Based Retrieval*（arXiv 2602.23906 v2，2026-10-23，Tübingen；首頁標 NeurIPS 2026）。CS-CLIP 是該文方法名（Component-Supervised CLIP）。COR 定義為 Pr[s(I,A+) ＞ s(I,A) ＞ s(I,A−)]，Table 8：COCO(Qwen) 58.2、COCO(Mistral) 50.4、CC3M(Qwen) 42.2，樣本數 509／454／474，ViT-B/32。基線 COR：CLIP 18.9／19.2／24.5，NegCLIP 32.2／25.8／29.1。CS-CLIP 的 ViT-L/14 只報 half-truth 準確率（80.4），沒報 COR。**另有一篇同名但不同的 CS-CLIP**（arXiv 2609.08242，首爾大學，Scene Graph-guided），與本題的單調性量尺無關，容易混淆。另外，v1 的數字不同（CLIP 40.6%、CS-CLIP 69.3%）〔搜尋摘要〕，引用時要用 v2。 | 全文 |
| 已有公開權重、含影像的評估資料、訓練／評測程式 | **正確。** HF `kbora/CS-CLIP`（MIT，open_clip 格式，2026-10-16 更新）；HF `kbora/Half-Truths`（CC-BY-4.0，三個 parquet 共約 200 MB，README 寫明嵌入原始影像位元組，共 1,437 組 anchor／full_truth／half_truth）；HF `kbora/CS-CLIP-Training`（CC-BY-4.0）；GitHub `kargibora/CS-CLIP`（MIT，2026-10-21 更新）。注意：論文自承訓練用 foil 經 Qwen3-VL 稽核約 24.6% 其實是真的（附錄 C）。 | API＋全文 |
| LCSE（稱 ICML 2026）做「拆概念、邏輯聚合」，SigLIP2＋LCSE 在其 Boolean benchmark 90.7% | **正確。** LCSE = *Similarity Is Not Logic: Factored Inference for Dual-Encoder VLMs*（arXiv 2607.23052，CMU），首頁標 PMLR 306（ICML 2026），本機 icml26.json 也有這個標題（一般 poster，非 spotlight）。FACTOR-Bench（1,100 筆運算子題）上 SigLIP 2＋LCSE 90.7%、CLIP＋LCSE 85.5%、最佳微調基線 73.2%。 | 全文＋本機 |
| LCSE 主要評估提供正確語句解析 | **部分正確，但這個弱點被原文擋掉大半。** FACTOR-Bench 的主結果確實用 oracle parse；不過附錄 Table 6 改用 GPT-4.1-mini 的解析後，只差 −0.3pp（85.5 → 85.2）。NegBench 與 COCO 用的本來就是 LLM 解析。真正的空缺是另外兩點：(1) 只做 Boolean 運算子，原文明寫「attribute binding, relations, counting … leave to future work」；(2) AND 用 power mean（γ＜0，約等於 min），沒有運算子時修正項為 0。 | 全文 |
| LCSE 程式公開 | 只看到專案頁連結（sultanmo.github.io/factored-vlm），沒有確認程式是否已放出，也沒有確認授權。 | 全文（連結） |
| CORE 權重與程式 | HF `Alibaba-NLP/core-emb-2b/8b`、`core-reranker-2b/8b`，card 授權 CC-BY-4.0；GitHub `Alibaba-NLP/core-emb`（API 回傳沒有授權欄位）。 | API |
| CLIC 權重與程式 | 論文寫「All our models and code are available」（clic-compositional-clip.github.io）；權重實際位置與授權**沒有查**。 | 全文 |

## 2. 引用者與 2026 年 8–11 月擁擠度

Semantic Scholar 引用（2026-11-14 查；S2 收錄不完整，CLIC 只有 8 筆，偏少）〔API〕：
- Half-Truths 2602.23906：1 筆，Flow-CLIP（2610.21937）。
- LCSE 2607.23052：3 筆，ConstraintMatch（2611.00784）、Hob-VL（2610.01605）、Seeing Is Not Addressing（2609.37230）。
- CORE 2609.04083：2 筆，FactEmbed（2611.05193）、ConstraintMatch。
- CLIC 2505.24424：8 筆，包含 Half-Truths、LCSE、Flow-CLIP、RoFLIP（2026-08）等。

2026 年 8–11 月的相鄰工作（arXiv 搜尋加引用追蹤）〔摘要〕：

| 論文 | 和本題的關係 |
|---|---|
| ConstraintMatch 2611.00784（10-31） | training-free 推論方法，把「同一實體的屬性要落在同一區域」當成成本，從 global similarity 中扣掉；用在 Qwen-2B 與 CORE-2B 上。屬於綁定層的修正，與本題最近。 |
| Flow-CLIP 2610.21937（10-15） | 用影像條件化的 token–patch 證據修正 global score，另有跨影像 margin ranking。屬於綁定層。 |
| The Limits of Binding in Dual Encoders 2608.15971（08-16） | 理論：遞迴綁定的 swap margin 為 m(D)=2b^−D；對比學習目標給綁定的獎勵受限於「caption 與自己的 swap 一起出現」的比率。**是本題理論版最直接的鄰居**，但談的是 swap，不是新增細節或條件數。 |
| GradCIR 2609.24152（09-21） | 組合式檢索用 4 級分級相關性訓練，明確處理「部分符合」。屬於電商 CIR。 |
| CS-CLIP（SNU）2609.08242（09-08） | scene graph 引導的結構化負例。 |
| Hob-VL 2610.01605、Seeing Is Not Addressing 2609.37230、FactEmbed 2611.05193、HN-CLIP 2608.18521 | 較遠：Boolean 推理基準、文字可達性、CIR、dense caption 的負例 margin。 |

**擁擠度判斷**〔推估〕：3.5 個月內約有 8–10 篇相鄰工作，其中 2–3 篇（ConstraintMatch、Flow-CLIP、Limits of Binding）與本題的「方法」或「理論」切面直接相鄰。熱門，但沒有被佔滿。

**「正確細節升／錯誤細節降」的單調一致性有沒有人當訓練目標？**
- 明確用三方排序 A+ ＞ A ＞ A− 當損失：**沒找到**。
- 但非常接近的已經有三個：CS-CLIP 的 unit 對 foil 對比（只壓錯誤那一邊，COR 是評估指標）；CORE 的 Rank-KL 五級排序（L5 全符合 ＞ L4 部分出現，等於「全部 ＞ 部分」這個方向）；GradCIR 的分級損失。
- 所以「把 COR 直接做成損失」只是小幅延伸，單獨撐不起一篇 ICML。

## 3. ICML 2026 冷門度（本機 icml26.json，只看標題）

資料沒有摘要欄位，只能做標題關鍵詞計數。共 6,628 篇 poster，其中 169 篇 oral、742 篇 spotlight。以下不列任何作者資訊〔本機〕。

- 標題同時含 compositional／binding／fine-grained／logic 與 CLIP／embedding／vision-language／retrieval，人工剔除不相關的之後，**直接相關約 8–10 篇，0 oral，2 spotlight**：
  - spotlight：*How can embedding models bind concepts?*、*Necessary Conditions for Compositional Generalization of Embedding Models*。兩篇都偏理論或表示學習。
  - 一般 poster：*Similarity Is Not Logic*（LCSE）、*FG-CLIP 2*、*PixCLIP*、*Left–Right Symmetry Breaking in CLIP-Style VLMs…*、*Seeing to Generalize: How Visual Data Corrects Binding Shortcuts*、*Locate then Correct: Debiasing Attention Heads in CLIP*、*Pix2Key*（開放詞彙檢索）。另有 *Compositional Generalization Requires Linear, Orthogonal Representations in Vision Embedding Models*（arXiv 2602.24264 標 ICML 2026，標題用詞不同，沒被關鍵詞抓到）。
- 標題含 retrieval 且含 image／multimodal／cross-modal：16 篇，0 oral，1 spotlight（雜湊類）。標題含 composed image retrieval：0 篇。
- 對照 ICML 2025（3,339 篇）：同類約 3–4 篇（*When and How Does CLIP Enable Domain and Compositional Generalization?*、*FG-CLIP*、*QuRe*（CIR）等）。

**判斷**〔推估〕：ICML 收這類題目，但量少（約 0.15%），而且上 spotlight 的是**理論或表示學習框架**的論文（綁定函數複雜度、組合泛化必要條件），不是 benchmark 或 CV 工程方法。純方法加 benchmark 的版本，比較像 CVPR 的題目（MCMR 就在 CVPR 2026）。不算太冷門，但必須用 ICML 的寫法。

## 4. 最強反駁與回應

| 反駁 | 強度 | 能否回應 |
|---|---|---|
| **R1：已經做掉了。** Half-Truths 已定義 COR、提出 CS-CLIP、公開資料；LCSE 已做「拆概念、外部聚合」。 | 最強 | 部分能。只能主打這些工作都沒有的東西：(a) 對評分函數提出「兩個方向都單調」的公理，並證明 mean pooling（LCSE §3 指出 holistic similarity 近似 mean pooling）與 min／product（LCSE 的 AND）各自違反哪一條；(b) 單向量線性評分在 k 個條件下的「部分 vs 全部」分離界。若只拿出「LCSE 換一個聚合器」，審稿人會判為增量。 |
| **R2：換大模型或 reranker 就解掉。** CORE-RERANKER-8B 在 SugarCrepe++ 上 87.5%。 | 強 | 能回應一半。(1) 縮放不是解法：CORE 表中 8B embedding 的兩組平均 72.7% 低於 427M 的 CLIPS+CLIC 81.0%（協定未完全對齊，要自己重跑確認）。(2) reranker 只看第一階段的 top-K；如果第一階段因為 k 很大而把「全部符合」的圖排在 K 名以外，reranker 救不回來。這要在實驗裡量第一階段的 recall@K 隨 k 的變化。(3) 但如果 reranker 在 Half-Truths 上的 COR 已經接近 100%，那論文只能限定在「固定索引、無 cross-attention」的情境，賣點會變窄。**這點必須第一週就量。** |
| **R3：只是評測或分析，缺方法。** | 強（使用者自己的門檻） | 外部分析的原版（辨識、綁定、聚合的歸因）本身就是分析。必須附上方法（見第 5 節的 training-free 評分器與訓練版蒸餾），否則不符使用者條件。 |
| **R4：是 CV 會議題目。** | 中 | 第 3 節的資料顯示 ICML 會收，但偏理論框架。回應方式是把主張寫成可證明的評分函數性質，並把實驗限縮到能驗證定理的設定。 |
| **R5：「正確細節要升」本身不合邏輯。** P(描述為真 \| 影像) 對合取只會下降。 | 中（審稿人可能提） | 能回應，而且可以變成賣點：用對數概似比（PMI）語意，s = Σ_c [log p(c\|I) − log p(c)]。正確細節滿足 p(c\|I) ＞ p(c)，分數上升；錯誤細節滿足 p(c\|I) ＜ p(c)，分數下降。兩個方向同時成立，正好說明為什麼機率（product）與 min 都不對。 |
| **R6：relation 單元的原子證據本來就不可靠。** LCSE 自己說失敗會轉移到證據抽取。 | 中 | 證據抽取改用 CS-CLIP 權重（MIT，relation 的 half-truth 準確率 79.2%），或用綁定修正（ConstraintMatch）當證據來源。但這會讓貢獻變成「組合別人的部件」，要小心定位。 |

## 5. 能撐 ICML 的正面版本（建議切入角度）

**暫定題目**：*Monotone Evidence Aggregation for Single-Vector Retrieval: Why Similarity Confuses Partial with Full Matches*〔推估〕

1. **理論**
   - 定義兩條公理：A1，加入正確條件，分數上升；A2，加入錯誤條件，分數下降。另加 A3，部分符合的影像嚴格低於全部符合的影像。
   - 命題一：mean pooling（holistic 相似度的近似）違反 A1 或 A2 的條件，取決於新條件的證據是高於還是低於目前的平均。這可以直接解釋 Half-Truths 觀察到的「plausible 錯誤細節會升分」。
   - 命題二：min、product、power mean（γ＜0）違反 A1。
   - 命題三：在「先校準、再取 log 概似比、再相加」這一族評分函數中，A1–A3 成立，條件是每個條件的證據有正確的方向。
   - 命題四（分離界）：若影像嵌入可做加法分解（ICML 2026 spotlight 2605.31503 的實證發現），任何對固定影像嵌入是線性的評分（也就是單向量 MIPS），其「全部 vs 缺一個條件」的 margin 與雜訊比會以 1/√(k−1) 衰減。分解式評分因為每個條件先經過飽和函數（log σ），已滿足條件的雜訊被壓縮，這個比值與 k 無關。代價是 m 次內積。
   - 這四條需要和 2608.15971（swap 的理論）及 Weller 等的 LIMIT（embedding 的 top-k 組合維度下界）區隔：本題談的是「新增」與「條件數」，不是 swap 或 sign-rank。
2. **方法**
   - (a) training-free：保持索引不變，查詢端把描述拆成單元，每個單元和影像嵌入做內積，用 sigmoid 校準，減去由索引抽樣估出的 log p(c)，加總後只對第一階段 top-K 重排。
   - (b) 訓練版：把 (a) 的分數蒸餾回單向量（類似 CORE 的 Rank-KL，但教師是有理論保證的分解式評分，不是 reranker），並讓 COR 三方排序成為損失的一項。
3. **實驗**
   - Half-Truths（COR，三組）；SugarCrepe++；MCMR（CVPR 2026，多條件，GitHub EIT-NLP/MCMR，授權 CC BY-NC-SA 4.0〔搜尋摘要〕）。
   - 自建 k＝1…6 的多條件查詢，來源可用 COCO 或 Visual Genome 標註，驗證命題四。
   - 保留 COCO 與 Flickr30K 的 R@K，並量測延遲。

## 6. 前兩週的便宜判別實驗與停損判準

GPU 時數都是推估：ViT-B/32、ViT-L/14 推論很便宜，2B／8B embedding 模型推論較貴。

| 週 | 實驗 | 估計 H200 小時 | 停損判準（事先寫定） |
|---|---|---|---|
| 1 | **E0 基線重現**：在 Half-Truths 三組上量 COR，模型包括 CLIP B/32、L/14、SigLIP2、CS-CLIP、CLIC-L、CORE-EMBED-2B/8B、CORE-RERANKER-2B。加上的細節可以直接用 anchor 和 full_truth、half_truth 的字串差分取得，不需要 LLM 解析。 | 1–3 | CS-CLIP B/32 的 COR 重現值和論文 58.2 差距超過 3pt，先查原因再繼續。**如果 CORE-RERANKER 的 COR ≥ 90%，本題只能限定為「固定索引」情境；這時需要和使用者確認是否仍要做。** |
| 1 | **E1 training-free 聚合器**：在凍結的 CLIP B/32 和 CS-CLIP 上比較 mean、min／LCSE 式、log 概似比加總三種聚合。 | 1–2 | **主停損**：凍結 CLIP B/32 加 LLR 的 COR 若 ＜ 45%（COCO-Qwen），而且套在 CS-CLIP 上也沒有比 CS-CLIP 本身的 58.2 高 ≥ 5pt，就停。 |
| 1 | **E2 條件數稀釋**：用 COCO 或 VG 建 k＝1…6 的查詢，量 holistic 評分在 P(s(全部) ＞ s(缺一)) 與第一階段 recall@50 隨 k 的變化。 | 2–4 | 若 holistic 的錯誤率沒有隨 k 單調上升（k＝1 到 6 的差距 ＜ 5pt），命題四沒有實證支撐，理論主軸不成立，停。 |
| 2 | **E3 保持率與延遲**：COCO 與 Flickr30K 的 R@1 和 R@5；在 top-K＝100 重排下量額外延遲。 | 1 | R@1 下降超過 1pt，或延遲超過 holistic 的 1.5 倍，修正後仍無法改善就停。 |
| 2 | **E4 蒸餾訓練版**：用 CS-CLIP 的程式碼與 COCO 訓練 B/32。CS-CLIP 原本是 8×A100、25 epoch；這裡先跑 5 epoch 判斷方向。 | 8–20 | 在 SugarCrepe++ 和 MCMR 上，若 COR 比 CS-CLIP 低，或 SugarCrepe++ 平均比 CS-CLIP 低 ≥ 2pt，就降級為只做 training-free 版。那樣信心最多到「中」。 |
| 2 | **理論**：寫出命題一到四的證明草稿（不花 GPU）。 | 0 | 命題四若需要不合理的假設（例如影像嵌入嚴格可加），就降級為「經驗定律」。那樣會失去 ICML 框架，停。 |

前兩週合計約 15–30 H200 小時〔推估〕。

## 7. 信心評級

- 外部分析原版（量尺加錯誤歸因）：**低到中偏低**。量尺已被 Half-Truths 佔走；歸因屬於純分析，不符合使用者的門檻。
- 第 5 節的重構版（公理、分離界、分解式評分、蒸餾）：**中偏低**。E1 與 E2 都通過可以升到「中」；E4 也通過，才有機會接近「中偏高」。
- 主要理由：
  - 有利：真實的 headroom 很大（最佳 COR 只有 58.2／50.4／42.2，CLIP 只有 18.9）；資料、權重、程式都公開且授權寬鬆；ICML 2026 收過兩篇同主題的 spotlight，都是理論框架。
  - 不利：Half-Truths、LCSE、CORE 三篇已經佔住量尺、方法、蒸餾三個位置；8–11 月還有 ConstraintMatch、Flow-CLIP、Limits of Binding 緊鄰；reranker 可能已經解掉問題；這題偏 CV，和使用者「停掉 CV、專心 ICML 方向」的取向有些衝突。

## 主要來源

- CORE：https://arxiv.org/abs/2609.04083 〔全文〕
- CLIC：https://arxiv.org/abs/2505.24424 〔全文〕
- Half-Truths／CS-CLIP：https://arxiv.org/abs/2602.23906 〔全文 v2〕；https://huggingface.co/datasets/kbora/Half-Truths 〔API〕
- LCSE：https://arxiv.org/abs/2607.23052 〔全文〕
- 同名 CS-CLIP（SNU）：https://arxiv.org/abs/2609.08242 〔全文前段〕
- MCMR：https://arxiv.org/abs/2603.01082 〔搜尋摘要〕
- 2608.15971、2605.31503、2602.24264、2611.00784、2610.21937、2609.24152、2610.01605、2609.37230、2611.05193、2608.18521 〔摘要〕
