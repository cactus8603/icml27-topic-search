# 放寬條件後的 ICLR 掃描：ICLR 2026 錄取論文重挑＋ICLR 2027 投稿熱度

日期：2026-11-14。沒有用 GPU，沒有寫實驗程式。
適用條件：中途收到兩次更新，都已納入。（1）不再排除電腦視覺應用與機器人；（2）「只要有研究價值、信心高就可以，不必打贏 SOTA」，但純負面分析仍然不行。
網路操作約 50 次：OpenReview API 16 次、Semantic Scholar 約 12 次（大多被 429 限流）、arXiv API 2 次（都被 429 擋下）、網路搜尋 8 次、arXiv HTML 全文 1 篇。
來源層級標記：〔全文〕抓了全文，只讀相關段落；〔摘要〕只讀摘要；〔搜尋摘要〕只看到搜尋引擎或第三方整理；〔OR〕OpenReview API 回傳的標題與摘要；〔S2〕Semantic Scholar 引用清單；〔推估〕我自己的估計。
論文用 arXiv 編號或 OpenReview id 標示，不列作者。

---

## 0. 結論先講

- **ICLR 2027 投稿可以查。** `api2.openreview.net/notes/search?term=…&group=ICLR.cc/2027/Conference` 能用；`/notes?invitation=…` 和 `content.venueid=…` 這兩種查法會被 challenge 擋下（403）。我抓到 8,693 篇投稿（含 643 篇撤稿、320 篇 desk reject），另外還抓到 5,552 則回覆，可見審稿意見與作者回覆已經公開。這只是樣本，不是全部投稿，偏誤說明見 §1。
- **這輪查完，沒有一個候選能直接評到「中偏高」。** 有一個候選評「中」，第一週停損實驗過關的話可以升到中偏高：§2 的候選 1，「合成資料選擇理論：從單一類別的線性模型推廣到多類別加條件偏移」。另外兩個是中偏低。
- 加入電腦視覺以後，我先查了四個看起來最有希望的方向，結果都已經擁擠或被做掉，詳見 §3：RAE 表徵空間擴散的「可擴散性預測」、影片模型的時序 SAE、多輪影像編輯漂移、LLM 多輪對話迷失。
- **回報 35 刪掉的題目，這次不建議重提。** 理由見 §4：條件放寬只解決「撞題」這一關，那些題目被刪的主因是缺口太小或改進空間已被吃掉，放寬救不回來。

---

## 1. ICLR 2027 投稿的主題熱度（樣本統計）

**樣本怎麼來的**〔OR〕：
- 用 search 端點查 `term=model`，offset 從 0 到 9000，每頁 1000 筆。端點回傳的 count 上限是 10000，所以 offset 9000 以後拿不到。
- 另外查了 image、reinforcement、learning、network、diffusion 五個詞，各抓 1000 筆，和上面合併去重。
- **偏誤：** 樣本偏向摘要裡有 "model" 一詞的投稿。下面的百分比是在樣本內算的，不是全體投稿的比例。全體投稿數我拿不到；以「樣本約占三到四成」推回全體數量，只是〔推估〕。
- 數字是用正規表示式比對標題加摘要得到的，算是粗分類。「calibration」那一列會把量化校準也算進去，數字偏高。

**primary_area 分布**（8,693 篇中有填的 3,141 篇〔OR〕）：
- foundation/frontier models（含 LLM）1,328
- generative models 976
- CV／audio／language 應用 965
- RL 785
- alignment／safety 707
- interpretability 551
- robotics 415
- representation learning 414
- physical sciences 364
- datasets & benchmarks 334

其餘領域都在 250 以下。

**關鍵字熱度**（樣本 8,693 篇，數字為篇數與樣本內百分比〔OR〕）：

| 熱度 | 主題（篇數，樣本內百分比） |
|---|---|
| 非常熱（≥5%） | agent（1,221，14.0%）、VLM／多模態（731，8.4%）、world model（521，6.0%）、RLVR／GRPO（465，5.3%）、機器人／具身（441，5.1%） |
| 熱（2–5%） | LLM-as-judge（343）、pass@k／exploration（316）、continual（268）、reasoning（258）、LoRA／PEFT（243，修正 regex 後）、steering（245）、one-step／few-step 生成（230）、影片生成（227）、**on-policy／self-distillation（215，2.5%，2026 年新起的一波）**、擴散 LM（212） |
| 中（1–2%） | model merging（192）、Muon／optimizer（164）、flow matching（162）、量化（158）、shortcut／spurious（154）、compositional（152）、SSM／linear attention（138）、hallucination（129）、MoE（116）、tabular（112）、tokenizer（111）、time series（111）、KV cache（103）、自回歸／串流影片（101） |
| 較少（<1%） | memorization（99）、unlearning（92）、CFG（77）、ICL（75）、SAE（64）、speculative decoding（61）、3DGS（54）、REPA（44）、PRM（42）、scaling law（38）、reward hacking（38）、latent reasoning（26）、data attribution（18）、dataset distillation（5） |

**幾個子題的撞題密度**（樣本內篇數〔OR〕）：
- 自回歸影片的 forcing／漂移：33 篇，例如 RepForcing、Mask Forcing、Rolling Sink、Clean Forcing、Revise by Appending。極擁擠。
- MoE 路由：約 40 篇。
- LVLM 幻覺：56 篇。
- model merging：83 篇。
- RAE 系列：13 篇，例如 Pooling RAE、VideoRAE、AlphaRAE、Taming Outlier Tokens、Distilling Drifting Transformers with RAE。
- 特徵上採樣：1 篇（DiveUp）。
- 合成資料選擇與理論：幾乎沒有。只有一篇自駕合成資料估值（`l3xRXkG0Z2`）。

**對 ICML 2027 的含義**〔推估〕：
- agent、VLM、RLVR、world model、自蒸餾、one-step 生成，ICML 一定會收很多。
- 但這些主題的撞題機率也最高。10 週、1–4 張 H200 的條件下要做出站得住的區隔，很難。
- 「理論加資料中心」以及「表徵／生成理論」這兩類，投稿量中等或偏少，ICML 錄取的比例又高，比較適合這次的限制。

---

## 2. 候選

### 候選 1：合成資料選擇理論，從單一類別的線性模型推廣到「多類別加條件偏移」，並導出新的選擇準則

**信心：中。** 第一週停損實驗過關的話可以升到中偏高。

**一句話題目：** 合成訓練資料除了共變異數偏移，還有「標籤／語意不忠實」造成的條件偏移。我們把 ICLR 2026 oral 的高維線性分析推廣到多類別與條件偏移，證明此時只做共變異數匹配不再最佳，並據此導出兼顧共變異數與忠實度的選擇準則，再用影像分類實測這個理論預測的轉折點。

**來源與缺口**：
- 來源是 2510.08123《High-dimensional Analysis of Synthetic Data Selection》，ICLR 2026 oral〔全文，前半〕。
- 模型設定：高維 ridgeless 線性迴歸，每個資料集只有一個類別，合成資料與真實資料共用 β。換句話說，論文**假設沒有條件偏移**。
- 結論段原文列了兩個未來方向：「extend the analysis to multiple Gaussian mixtures」和「introduce a model shift (different β between synthetic and real samples)」。也自承「our theory ignores the interactions between classes」〔全文〕。
- 方法：在 CLIP 特徵的 PCA 32 維空間裡，逐類貪婪挑選樣本，讓合成資料的共變異數接近真實資料。
- 撞題檢查：
  - 一次網路搜尋沒找到 2026 年的直接後續〔搜尋摘要〕。
  - S2 引用清單這次被限流，沒查到。
  - ICLR 2027 樣本裡沒有同題的投稿〔OR〕。

**切入角度（可以寫成貢獻的部分）**：
1. **理論。** 推廣到 K 類高斯混合，並允許合成資料的條件偏移（β_s ≠ β_t，或合成樣本有一定比例的語意錯標）。把測試誤差拆成「共變異數項」加「條件偏移項」，再求出兩項之間的交換關係隨真實／合成樣本比 n_s/n_t 怎麼變。
2. **可驗證的預測。** 理論會給出一個轉折點：條件偏移超過某個門檻後，只做共變異數匹配會比隨機挑選還差，或者最佳合成數量會變少。
3. **方法。** 在選擇目標裡加入忠實度懲罰，用真實資料訓練的 probe 估計每個樣本的一致性。懲罰權重由理論給定，不靠網格搜尋。
4. **實證。** 用 CFG 強度、prompt 錯置等方式控制條件偏移的大小，在 CIFAR-10、ImageNet-100 上驗證理論預測的轉折點。這屬於「新理解加可用的方法」，不以 SOTA 為目標，符合放寬後的條件。

**最接近的工作與區隔**：
- 2510.08123：本題的出發點。它只有一個類別、沒有條件偏移，本題把這兩點補上〔全文〕。
- 2512.14230，以及同組的「filtered contrastive learning」理論：分析的是多模態對比學習下的過濾，資料模型是「部分配對錯誤」。本題是監督式下游訓練裡的合成資料選擇，不同〔搜尋摘要〕。
- 模型崩潰理論的 ICLR 2027 投稿，例如 `MNFqHDmqU6`（迭代高斯估計的崩潰判準）、`MIWQusH2rq`：處理的是遞迴自我訓練，不是從固定合成池裡挑資料〔OR〕。

**最強的反駁與回應**：
- **反駁一：原論文的實證差距本身就在雜訊內。** CIFAR-10 scratch 設定下，共變異數匹配 54.00±1.89，DS3 52.83±2.19；換成 T2I 合成池時，它和 DS3、center matching 不相上下〔全文，表 1、2〕。照這個量級，新準則可能也量不出差別。
  - 回應：主貢獻放在理論與它預測的轉折點，不放在平均準確率。實驗刻意做「條件偏移大」的設定，例如低 CFG 或 prompt 錯置，讓效應量變大；每個設定跑 5 個以上的種子。
- **反駁二：線性理論套不到深度網路。** 原論文也承認這點。
  - 回應：沿用原論文的做法，在 CLIP 特徵上做 linear probe 或 distillation，這個設定最接近理論。scratch 訓練只當作附加結果。
- **反駁三：忠實度過濾早就有人做**，例如用 CLIP score 或分類器信心篩合成影像。
  - 回應：本題的貢獻不是「要過濾」，而是**何時該過濾、過濾多少**由理論給定，並且和共變異數匹配是聯合最佳化。實驗要把這些既有的啟發式過濾法列為 baseline。

**前兩週的停損實驗**：
- 第 1 週：
  - 用原論文的程式（搜尋摘要提到 GitHub 上有 Rezaei-Parham/Synthetic-Selection，我還沒有核實）重現 CIFAR-10 的 pretrained／distillation 設定。
  - 用公開的 T2I 模型，在兩個 CFG 強度下各生成每類 5k 張影像，另外做一組 prompt 錯置 10–30% 的合成池。
  - 量測「只做共變異數匹配」相對隨機挑選的增益，看它是否隨條件偏移變大而縮小甚至反轉。
- 第 2 週：
  - 在線性模擬中驗證推導出的最佳懲罰權重。
  - 在 CIFAR-10 上比較新準則和共變異數匹配、DS3 類 baseline、CLIP-score 過濾。
- **事先寫好的判準：**
  1. 條件偏移最大的那組設定下，共變異數匹配的增益相對最小的那組至少降 2 個百分點，且超過 5 個種子的標準差。
  2. 新準則在該設定下贏共變異數匹配至少 1.5 個百分點，且超過雜訊。
  - 兩條都不成立就停。只成立第 1 條的話，題目會變成偏負面的分析，不建議繼續。
- **GPU 小時〔推估〕：** 生成約 10 萬張 512px 影像約 10–20 GPU 小時（H200，少步數採樣器）；小模型訓練約 100 次、每次 0.1–0.3 GPU 小時；合計約 40–70 GPU 小時。
- **10 週全套〔推估〕：** 加 ImageNet-100、一個非影像設定（例如文字分類用 LLM 生成的合成資料），以及完整證明，約 300–500 GPU 小時。

**主要風險**：
- 理論推導的難度：多類別加條件偏移的高維極限，可能需要隨機矩陣的工具。學生如果沒有這方面背景，時間會很緊。
- 退路：先只處理「單類別加 β 偏移」這個較簡單的推廣，它本身就是原論文點名的未來工作。

---

### 候選 2：用對比學習的「困難樣本」理論做圖文對比預訓練的資料篩選

**信心：中偏低。**

**一句話題目：** 把 ICLR 2026 oral「困難樣本傷害無監督對比學習」的譜圖理論，從單一模態的增強圖推廣到圖文雙模態，從中導出可以在網路資料上計算的困難度分數，以及配套的溫度和 margin 調整，並在 DataComp small／medium 上檢驗。

**來源**：
- 2501.01317，ICLR 2026 oral〔搜尋摘要〕。
- 主要發現：移除 20–40% 的困難樣本，CIFAR-100、STL-10 的線性探測準確率反而上升；論文也證明溫度與 margin 調整可以收緊誤差上界。
- 論文自承：困難樣本的定義依賴監督標籤，真實資料上無法精確得知哪些是困難樣本。

**切入角度**：
1. 雙模態的增強圖推廣，加上無標籤的困難度代理分數，例如跨模態最近鄰的語意混淆度。
2. 理論預測「移除比例」與「溫度」之間是可替代的關係。
3. 在 DataComp 上和 CLIP-score、DFN 類方法、Sieve 比較，目標不是打贏，而是驗證理論的預測。

**最接近的工作**：
- 2512.14230，以及同組的過濾對比學習理論：已有「過濾如何改變雙模態對比學習速率」的分析〔搜尋摘要〕。
- Sieve（2310.02110）：指出 CLIP-score 會把「困難但對齊」的配對誤刪〔搜尋摘要〕。這和本題的方向直接衝突，需要正面回答。
- 2405.19547（CLIPLoss／norm-based 資料選擇）〔搜尋摘要〕。

**最強的反駁：** 網路圖文資料的主要問題是錯配，不是「跨類相似」。而且已經有雙模態過濾理論，本題剩下的區隔就只有「困難」與「錯配」的分辨。
- 回應：這恰好可以當成核心問題，也就是理論上區分「該刪的困難樣本」和「該留的困難樣本」。但這需要新的理論，不確定 10 週做得完。

**停損實驗**：
- 在 DataComp small（12.8M 樣本，ViT-B/32）上，比較「CLIP-score 過濾」與「CLIP-score 過濾後再刪掉困難度分數最高的 x%」，x 取 10、20、30。
- **事先判準：** ImageNet 零樣本準確率與 38 項平均分數，至少有一項穩定提升 0.5 個百分點以上，且超過 3 個種子的雜訊。
- GPU 約 30–60 GPU 小時〔推估：small 規模單次訓練只要幾個 GPU 小時，但下載與前處理 12.8M 筆資料很花時間〕。

**為什麼只評中偏低：** 雙模態過濾理論已經存在；DataComp 的資料下載和清理成本高；而且 ICLR 2027 樣本裡 CLIP 資料策展類只有 1 篇，主題熱度在下降〔OR〕。

---

### 候選 3：特徵上採樣的忠實度：量尺加不確定性感知上採樣器（電腦視覺表徵）

**信心：中偏低。** 這題的驗證不足，只做了引用清單掃描。

**一句話題目：** 現有通用特徵上採樣器（AnyUp 等）只用下游 probe 的分數評估。我們定義上採樣特徵對原始 VFM 特徵的「忠實度」量尺，包括在高解析度輸入下量到的真實高解析度特徵，以及多尺度一致性；再提出輸出逐像素不確定度的上採樣器，讓下游在不確定的區域退回低解析度特徵。

**來源：** AnyUp（2510.12764，ICLR 2026 oral）。

**撞題**〔S2，2026 年引用者 37 篇〕：直接做上採樣方法的至少有 9 篇：
- 2511.16301 Upsample Anything
- 2511.18452 NAF
- 2601.17950 UPLiFT
- 2603.13571 DiveUp（也是 ICLR 2027 投稿）
- 2606.14024 ViT-Up（標題就有 "Faithful"）
- 2606.22749 RaysUp
- 2605.17472
- 2611.02854（task-guided）
- 2603.05787（spectral probing，量尺性質）

ViT-Up 和 spectral probing 很可能已經佔走「忠實度」這個角度。這兩篇我沒有讀全文，需要先讀才能判斷。

**最強的反駁：** 方法很多，增量空間小；而且 ICML 對純視覺 dense feature 的題目收得比 CVPR、ICLR 少〔推估〕。

**停損：** 先讀 2606.14024 和 2603.05787 的全文（不用 GPU）。若兩篇都已提出忠實度量尺，這題就停。之後在 DINOv2／v3 上重現 AnyUp 並量測忠實度，約 10–20 GPU 小時。

---

## 3. 這輪查過、不建議的方向（加入電腦視覺後新查的）

| 方向 | 來源 | 不建議的理由 |
|---|---|---|
| RAE 表徵空間擴散：哪種編碼器「可擴散」、預測量尺加修正 | RAE（2510.11690，ICLR 2026 poster） | 2026 年的 S2 引用者 363 篇；與表徵／潛空間相關的約 190 篇。已有 2609.28473《On the Diffusibility of High-Dimensional Latents》、2608.00626《Where Does Generative Difficulty Reside?》、2609.34732《What Visual Generators Need from Teachers》、2609.31620 FuseReg（處理重建與生成的落差）、2605.21981 RiT。ICLR 2027 樣本另有 13 篇 RAE 系列。我原本設想的切入點已被佔走〔S2／OR〕 |
| 影片模型的時序 SAE（延伸 Temporal SAE，ICLR 2026 oral） | — | 2604.03919《Interpreting Video Representations with Spatio-Temporal SAEs》已自稱是第一個系統性研究，有 contrastive SAE，也做了時間一致性〔搜尋摘要〕 |
| 多輪影像編輯的漂移 | — | 2605.08250 VAE-LFA（NeurIPS 2026）、FreqEdit（CVPR 2026）、TailSteer、MT-EditFlow（2606.01985）、2603.29736（漂移上界）；ICLR 2027 另有 MTEdit、Flow-LQR〔搜尋摘要／OR〕 |
| LLM 多輪對話迷失（Lost in Conversation，ICLR 2026 oral）的訓練修正 | — | 2605.24432 FiC 已用 view-asymmetric 自蒸餾，在 3B–14B 模型上回到單輪表現的 92% 以上，兩個 Llama 骨幹達 100%〔搜尋摘要〕；理論面也已有 ICLR 2027 投稿 `YotPdYyAkS`。改進空間所剩不多 |
| Agent 的推測式動作（Speculative Actions，ICLR 2026 oral，2510.04371） | — | 27 篇引用者裡，系統面的後續超過 15 篇，例如 2609.38201 TomasuLLM、2608.00881 AOSpec、2610.15246 ObsSpec〔S2〕 |
| MoE 路由與專家耦合（ERC loss，2512.23447，ICLR 2026 oral） | — | 引用者只有 15 篇，不擠，但已有 2605.12476（路由學到專家的幾何）和 2604.07030（小規模路由測試平台）。ICLR 2027 樣本的 MoE 路由類有約 40 篇。我想不出有足夠區隔的具體切入點，所以沒有深查 |
| 自回歸／串流影片（Stable Video Infinity、Self-Forcing++） | — | ICLR 2027 樣本 33 篇，極擁擠〔OR〕 |
| LVLM 幻覺（Hallucination Begins Where Saliency Drops，oral） | — | ICLR 2026 錄取的同類 poster 超過 15 篇，ICLR 2027 樣本 56 篇 |

---

## 4. 回報 35 刪掉的題目，能不能重提

逐項判斷如下。條件放寬只解決撞題，但這些題目大多是因為缺口小或改進空間已被吃掉才刪的：

- **無限算力預訓練的 ensemble：** 不建議重提。ICLR 2027 樣本有 `Cp8OdsnZz1`（跨規模選重複次數）、MoE 在資料稀少時過擬合等投稿，加上回報 35 列的 2610.37842 等，至少 8 篇。我想不出新的切入點。
- **Coverage principle、學習率衰減與課程、微型評測、Train-before-test、因果介入偏離、FIRE、HSD、低精度 FlashAttention、logits 低秩：** 維持原判。理由同回報 35：已有直接的後續，或理論上的上限本身就小。
- **Neon 搬到 LLM：** 已在候選清單上，這裡不重提。ICLR 2027 樣本另有 `XWru3nYzAE`（用幾何修改過的輸出做生成模型自我訓練），和影像版的 Neon 相鄰，需要留意〔OR〕。

---

## 5. 沒能確認的事項

- arXiv API 整輪被 429 擋下，S2 後半也被限流。所以候選 1 的 2026 年後續只靠一次網路搜尋加 ICLR 2027 樣本確認，**可能有遺漏**。
- ICLR 2027 樣本約 8.7k 篇，不是全部投稿，而且偏向含 "model" 一詞的摘要。
- 2510.08123 只讀了前 10 萬字元，附錄沒有讀。程式庫是否公開，只有搜尋摘要提到，我沒有核實。
- 候選 2、3 的最接近工作只讀到搜尋摘要，沒有讀全文。
- 我沒有抓 ICLR 2027 投稿的審稿分數。回覆筆記雖然在樣本裡，但沒有分析。

## 6. 暫存資料

- ICLR 2027 樣本（已整理）：`/tmp/claude-1047/-work-ChihChungHsu-cactus8603-icml27-topic-search/4f80774d-118c-4d50-aa0e-d24ce12f76e5/scratchpad/iclr27.json`
- 原始頁面：`/tmp/claude-1047/-work-ChihChungHsu-cactus8603-icml27-topic-search/4f80774d-118c-4d50-aa0e-d24ce12f76e5/scratchpad/or/`
- RAE 引用者清單：`/tmp/claude-1047/-work-ChihChungHsu-cactus8603-icml27-topic-search/4f80774d-118c-4d50-aa0e-d24ce12f76e5/scratchpad/iclr48/rae_cites.txt`
