# 深查：組合泛化／屬性綁定（凍結視覺表徵 vs. 真實屬性參照的差距）

日期：2026-11-14｜範圍：不動 GPU、不寫程式｜網路搜尋與開頁約 25 次
來源層級標記：〔全文〕＝讀了論文全文或表格；〔摘要〕＝arXiv/官方摘要；〔搜尋摘要〕＝搜尋引擎或第三方摘要；〔推估〕＝我自己的估算；〔本機〕＝icml26.json／icml25.json 的標題計數。
本報告不列任何作者姓名或聯絡資訊，論文一律以 arXiv 編號或標題稱呼。

---

## 0. 結論先講

- 外部分析指的論文找到了：**"Are Object-Centric Representations Better at Compositional Generalization?"**，arXiv 2602.16689（v1，2026-02-18），ICML 2026 一般 poster（icml26.json：`Accept (regular)`，OpenReview id `cgLZYrBHh6`）。
- 三組數字**全部對得上**，但在 arXiv v1 是 **Table 13**（TF 5 下游模型、COOD 測試集）。v1 只有 14 張表，沒有表 15，可能是相機版改了編號；相機版 PDF 我拿不到（OpenReview 擋下載），所以「表 15」這個編號**查不到，無法核實**。
- 新發現（外部分析沒提）：差距**不只是感知誤差**。即使把同一表徵在分佈內（ID）的落後也扣掉，仍有約 12–15 pp 是「組合」特有的落差（見 §1.3）。這比外部分析的說法強，是這個題目能站住的主要依據。
- 程式、固定 splits、渲染資料、預訓練權重：**全部查不到公開版本**。要重現就得自己渲染，或改用現成的公開資料集。
- 冷門度：不冷門。ICML 2026 有 4 篇組合泛化／綁定類的 spotlight，但物件中心（slot）架構類 9 篇全是一般 poster，0 篇 spotlight。
- **信心：中**，還不到使用者要的「中偏高」。要升級，條件是前兩週的門檻實驗通過：新骨幹上差距仍在，而且方法原型能追回一部分差距。

---

## 1. 核實那篇 ICML 2026 論文（arXiv 2602.16689）

### 1.1 設定〔全文〕
- 三個合成世界：CLEVRTex、Super-CLEVR、MOVi-C 風格，用 Blender 自行渲染。CLEVRTex 每個物件由（形狀、大小、材質）三元組定義，共 192 種物件。
- 先保留 20% 的物件組合給 COOD 測試；訓練集分 easy / medium / hard 三級，每升一級可用物件約減半。每個訓練集 48k 張（40k 訓練、4k 驗證、4k ID 測試），COOD 測試集 4k 張。每張圖平均 42 個問答對，每個訓練集約 170 萬個 QA。
- 表徵：DINOv2 **ViT-S/14**、SigLIP2 **ViT-B/16**（都是小模型），兩者的物件中心版 DINOSAURv2、SigLIPSAUR2（Slot Attention 重建凍結特徵，7 個 slot），另有 k-means 與 cross-attention 調整大小的對照組。
- 下游 VQA 模型：從零訓練的小 transformer（2 或 5 層、hidden 128），問題用 T5-base 編碼；batch 128，lr 1e-4。
- 參照組（GT Oracle）：直接把所有可見物件的真實屬性當作「影像表徵」輸入。

### 1.2 數字核實（arXiv v1 Table 13：TF 5、COOD 準確率 %）〔全文〕

| 設定 | 外部分析說法 | 全文實際 | 文內最佳表徵是哪個 | 差距 |
|---|---|---|---|---|
| CLEVRTex medium | 76.9 vs 97.5（20.6） | 76.9 vs 97.5 | SigLIPSAUR2（SigLIP2 是 76.7） | 20.6 ✔ |
| Super-CLEVR medium | 66.8 vs 91.6（24.8） | 66.8 vs 91.6 | SigLIP2（dense） | 24.8 ✔ |
| MOVi-C hard | 55.3 vs 78.8（23.5） | 55.3 vs 78.8 | SigLIPSAUR2 | 23.5 ✔ |

附註：
- 用小下游模型 TF 2 時，參照組本身較低（Table 11：CLEVRTex-M 89.0、Super-CLEVR-M 86.8、MOVi-C-H 68.9），差距會小一些。外部分析挑的是 TF 5，算是比較有利的版本，但不算錯。
- MOVi-C medium 的差距更大，外部分析沒提：最佳 57.9 對參照 84.2，差 26.3 pp〔全文 Table 13〕。
- hard 設定下連參照組都掉到 67–79，而 medium 是「參照組幾乎能解、表徵不能解」的區段，最適合當主戰場。

### 1.3 差距是不是全來自「參照組拿真值」？把 ID 落後拆出來〔全文表格數字，相減為推估〕

ID 準確率取自 Table 12（TF 5）。用 SigLIP2（dense）當表徵：

| 設定 | 表徵 ID | 表徵 COOD | 表徵 ID−COOD | 參照 ID | 參照 COOD | 參照 ID−COOD | **組合特有落差** |
|---|---|---|---|---|---|---|---|
| CLEVRTex-M | 93.9 | 76.7 | 17.2 | 99.7 | 97.5 | 2.2 | **≈15.0** |
| Super-CLEVR-M | 84.9 | 66.8 | 18.1 | 97.0 | 91.6 | 5.4 | **≈12.7** |
| MOVi-C-H | 85.1 | 54.6 | 30.5 | 98.7 | 78.8 | 19.9 | **≈10.6** |

解讀：表徵在 ID 也落後參照組 6–14 pp（這部分是感知或讀出誤差，外部分析說「只證明表徵可能是瓶頸」是對的）。但把 ID 落後扣掉後，仍有 10–15 pp 只在未見組合才出現。這是「讀得出個別屬性，但遇到新組合就綁錯」這個說法目前最直接的證據。

但要注意：論文**沒有**做「對單一屬性做線性探測」這類實驗。所以「模型能讀出個別屬性」這半句，在這篇論文裡**沒有被直接驗證**，要我們自己做。

### 1.4 「60 萬步」與計算量
- 〔全文 App. C〕最貴的組合（DINOv2 + TF 5）訓練 600k 步到收斂（沿用 ICLR 2025 那篇 VQA 物件中心研究 arXiv 2407.15589 的協定）。其他表徵依 FLOPs 比例換算步數，例如 SigLIP2 的 TF 2 / TF 5 分別是 641,123 / 688,733 步（Table 6），並在學習飽和後設上限。外部分析的「60 萬步」✔。
- 在 H200 上跑多久〔推估〕：下游模型約 1M 參數，序列約 300 token，batch 128。先預抽特徵：40k 張 × 256×384 fp16 ≈ 8 GB，可整包放進 H200 顯存。每步大概受 kernel 啟動延遲限制，約 10–20 ms，所以 **600k 步 ≈ 2–3.5 小時／run**。模型很小，一張 H200 可以同時跑 4–8 個 run。照原文規模全部重做（9 個 split × 約 14 種表徵 × 2 種下游 ≈ 250 runs）約 500–900 GPU-h 的序列時間，並行後牆鐘時間約 1 週（4 卡）。不過門檻實驗不需要全做，見 §5。
- DINOSAUR／SigLIPSAUR 的 slot 預訓練：每個資料變體一次。論文沒給時數；我推估每次 10–20 GPU-h，論文規模 18 次約 200–350 GPU-h〔推估〕。
- 論文的算力來自歐洲超算中心（JUWELS／JUPITER 與 NHR@KIT）〔全文致謝〕，原作者算力遠大於 1–4 張 H200。

### 1.5 程式、splits、資料、權重是否公開
- arXiv v1 全文：唯一的 GitHub 連結是 fvcore（算 FLOPs 用的工具），沒有自家程式連結〔全文〕。
- ICML 2026 虛擬頁面：只連到 OpenReview，沒有程式、資料或權重〔開頁〕。
- GitHub 以 "object-centric compositional"、"SigLIPSAUR" 等關鍵字搜尋：沒有對應 repo〔開頁〕。第三方彙整站也說找不到維護中的 repo〔搜尋摘要，可信度低〕。
- 它沿用的 VQA 協定（arXiv 2407.15589）在 arXiv HTML 裡也找不到程式連結〔開頁〕。
- **結論：官方程式、固定 splits、渲染資料、預訓練權重全部查不到，無法核實是否存在。** 要重現只能自己用 Blender 渲染：CLEVRTex 生成器、Super-CLEVR、Kubric 都公開。三個世界合計約 44 萬張圖，GPU Cycles 每張 2–5 秒，約 250–600 GPU-h，**這是最大的可行性風險**〔推估〕。替代做法見 §5。

### 1.6 ComPA
- 52.8% 出自 arXiv 2505.19406〈Unveiling the Compositional Ability Gap in Vision-Language Reasoning Model〉（benchmark 名為 ComPABench）。這是它們提出的 RL-Ground 方法在組合任務上的**最佳整體分數**（52.8%）〔全文〕，不是「基準模型只有 52.8%」。
- 這篇研究的是 VLM 在 RL／SFT 後訓練下，跨模態、跨任務的**技能組合**，不是物件屬性綁定。
- NeurIPS 2025 收錄：arXiv HTML 沒有寫 venue，**查不到，無法核實**。
- 同意外部分析：它不足以證明這個題目有空間，和本題只是間接相關。

---

## 2. 新骨幹是否測過、擁擠程度

### 2.1 新骨幹
- 該論文只測了 DINOv2 ViT-S/14 與 SigLIP2 ViT-B/16〔全文〕。**DINOv3、更大的 SigLIP2（L／So400m）、DINOv2-L/g 都沒測**。搜尋也找不到用 DINOv3 做 slot／組合泛化的結果〔搜尋摘要〕。
- 所以「換大骨幹後差距還在不在」是開放問題，同時也是風險：如果大骨幹把組合特有落差縮到 5 pp 以下，題目就沒了。這是門檻實驗第一關。

### 2.2 引用者
- Semantic Scholar `arXiv:2602.16689/citations` 回傳 **0 筆**（2026-11-14 查詢；也可能只是索引延遲）。
- 這篇有一個 ICLR 2025 workshop 前身〈Object-Centric Representations Generalize Better Compositionally with Less Compute〉，結論相同：物件中心表徵在未見組合上較好，且下游算力少 4–8 倍〔搜尋摘要〕。

### 2.3 直接相關、會被審稿人拿來比的工作

| 工作 | 層級 | 關係 |
|---|---|---|
| arXiv 2602.24264〈Compositional Generalization Requires Linear, Orthogonal Representations in Vision Embedding Models〉，ICML 2026 **spotlight**，有程式 | 摘要 | 理論上證明組合泛化**需要**線性、正交的逐概念分解；在 CLIP／SigLIP／DINO 上量到部分線性分解，且分解程度與未見組合的表現相關。**這是本題最重要的理論支點，也是最大的撞題來源**。 |
| arXiv 2605.31503〈How can embedding models bind concepts?〉，ICML 2026 **spotlight**，有程式 | 摘要 | CLIP 的場景嵌入可加性地分解成物件表徵，但綁定函數複雜度高，所以泛化不到未見組合；從零訓練的 transformer 在資料覆蓋足夠時會學到低複雜度（乘法型）綁定。和上一篇可能出自同一團隊，代表這個團隊正在追這條線。 |
| ICML 2026〈Necessary Conditions for Compositional Generalization of Embedding Models〉（spotlight）、〈Local Mechanisms of Compositional Generalization〉（spotlight）、〈Formalizing the Binding Problem〉、〈Seeing to Generalize: How Visual Data Corrects Binding Shortcuts〉 | 本機標題 | 只看了標題，內容未核實。 |
| arXiv 2510.24709（NeurIPS 2025）〈Does Object Binding Naturally Emerge in Large Pretrained ViTs?〉 | 搜尋摘要 | ViT 中層可用二次探測器讀出「兩個 patch 是否屬於同一物件」，準確率約 90%；但它研究的是物件身分綁定，不是屬性組合。 |
| arXiv 2412.18743（NeurIPS 2024 workshop）物件中心模型在屬性組合上的成功與限制 | 摘要 | 單物件簡單影像；物件中心模型在屬性組合上有一定能力。 |
| arXiv 2607.22705 EditCLEVR（2026-07） | 摘要 | 成對場景介入，評估物件中心表徵在屬性編輯下的組合忠實度，有 COOD 測試組。評估面相近。 |
| arXiv 2610.22602 SABER〈Stuck on Shuffle〉（2026-10） | 摘要 | CLIP 與 MLLM 嵌入模型的屬性－物件綁定基準，並用純文字微調改善綁定。屬於 VLM 檢索這一側。 |
| arXiv 2609.31456 COMPASS（2026-09） | 摘要 | 把 VLM 的組合失敗拆成「整合成本」與「單一技能負載」，發現整合成本是真的，但只佔一部分。 |
| arXiv 2609.08886 FRAME（2026-09） | 摘要 | 用逐屬性讀出做物件記憶的屬性組合檢索，屬於機器人應用。 |
| 舊方法線：SysBinder／block-slot（ICLR 2023）、Learning to Compose（ICLR 2024）、Provable Compositional Generalization for OCL（ICLR 2024） | 記憶與搜尋結果 | 「逐因子分塊 slot」的方法已經存在，必須當基線。 |

**擁擠評估**：
- 「綁定／組合泛化」這個大題 2026 年明顯升溫：ICML 有 2 篇以上 spotlight，arXiv 8–11 月約 4–5 篇相關。
- 但「**凍結的基礎模型特徵 → 物件中心或 dense 表徵 → 小下游模型**」這個設定下，以「追回到參照組」為目標的方法論文，我沒找到。
- 撞題風險中等。最可能的撞題來源是上面兩篇 ICML spotlight 的團隊推出後續作。

---

## 3. 冷門度（icml26.json，只做標題關鍵詞計數）〔本機〕

ICML 2026 有 6,797 筆，其中 spotlight 742 筆，基準 spotlight 率約 11%。

| 類別（標題 regex） | 篇數 | spotlight | 備註 |
|---|---|---|---|
| object-centric／slot | 9 | 0 | 其中 1 篇是擴散語言模型的 slot filling，與本題無關；實際相關約 8 篇 |
| compositional / systematic generalization | 8 | 3 | 含 2602.24264 等 |
| binding | 9 | 2 | 實際相關約 5 篇，其餘是藥物分子結合、KV binding 等 |
| compositional（任何用法） | 37 | 6 | |
| disentangl* | 46 | 8 | |

對照 ICML 2025（3,459 篇）：object-centric／slot 2 篇、組合泛化 5 篇（1 篇 spotlight）、binding 3 篇。

判斷：
- **ICML 會收這類題目，而且組合泛化／綁定的理論或機制類工作拿 spotlight 的比例偏高**（約 4/15，高於 11% 的基準）。
- 但純物件中心架構類在 ICML 只拿到一般 poster。
- 包裝建議：以「組合泛化／綁定的表徵幾何」為主軸，物件中心只當其中一種工具，不要寫成 CV 架構論文。

---

## 4. 最強反駁與回應

1. **「這只是 CLEVR 類玩具資料」**（最強）
   - 回應：能定義「未見組合」的受控世界本來就是必要條件；ICML 2026 的兩篇 spotlight 也用受控設定。
   - 但必須補一個真實影像的遷移實驗，例如組合式零樣本（compositional zero-shot）基準 MIT-States／C-GQA（真實影像的屬性－物件未見組合），或 SABER 一類的綁定基準。只要在真實資料上有一致方向的增益，就能大幅緩解。
   - 可回應程度：**部分**。

2. **「差距來自參照組拿到真值，本質上追不回」**
   - 回應：§1.3 的拆解顯示，扣掉 ID 感知落差後仍有 10–15 pp 只在 COOD 出現；參照組本身的 ID−COOD 落差只有 2–5 pp（MOVi-C-H 例外，約 20）。
   - 所以目標要定成「追回**組合特有落差**」，不是追到參照組。這樣就把「追不回」的批評轉成可量測的目標。
   - 可回應程度：**大致可以**。

3. **「這是 CV 題目」**
   - 回應：用表徵幾何（線性、正交分解，2602.24264）和綁定函數複雜度（2605.31503）當理論框架，以下游泛化當檢驗；ICML 對這條線給過 spotlight。
   - 可回應程度：**可以**。

4. **「方法已經有了（SysBinder、分塊 slot、因子化 VQ）」**
   - 回應：舊方法都是從像素端到端訓練。在凍結的基礎模型特徵上，還沒有人以組合 VQA 的參照組差距來系統比較這些方法。新意要放在「凍結特徵上的組合結構重組」加上理論預測的驗證。
   - 可回應程度：**中**。若結果只是「SysBinder 加 DINOv3 就好了」，貢獻會偏薄。

5. **「大骨幹自己就解了」**
   - 回應：無法預先回應，只能用門檻實驗回答。如果真的解了，題目不成立。這是最需要先測的未知數。

6. **「方法用了屬性標籤，對 dense 基線不公平」**
   - 回應：參照組本身就用了全部真值，所以「少量標籤」本來就在兩者之間，可以當連續譜來報告。但至少要有一個無標籤或只用 QA 監督的版本。

---

## 5. 建議的正面貢獻版本與前兩週計畫

### 5.1 題目雛形
暫名：「把未見組合拉回可組合的幾何：凍結視覺特徵上的因子化重組」

- **分析（支撐方法，不是主貢獻）**：在 DINOv2／DINOv3／SigLIP2（S/B/L）及其 slot 版本上，把組合泛化落差拆成三項：
  - (a) 感知落差：ID 與參照組之差；
  - (b) 單屬性讀出落差：逐屬性線性探測在未見組合上的準確率；
  - (c) 綁定落差：屬性能讀出，但配錯物件。
  - 同時量 2602.24264 定義的線性與正交程度，檢驗「正交程度可預測 COOD」這個理論預測在 slot 表徵上是否成立。
- **方法（主貢獻）**：一個輕量的「因子化 slot 讀出＋子空間重組」模組，放在凍結特徵或 slot 上：
  1. 用 ID 資料學出每個屬性的低秩子空間，並加正交約束；
  2. 在 slot 之間互換屬性子空間的分量，合成「未見組合」的潛在 slot；
  3. 用重組出的 slot 加上由問題程式重算的答案，增強下游訓練。CLEVR 類資料的問題由程式生成，答案可以直接重算。
  - 無標籤版本：只用 QA 監督訓練子空間，或用跨 slot 的一致性正則取代重算答案。
  - 基線：SysBinder 類分塊 slot、cross-attention 調整大小、k-means、單純加大下游模型。
- **遷移**：在 MIT-States／C-GQA 等真實影像的組合零樣本基準上，加同一個讀出模組。

### 5.2 前兩週門檻實驗（避開自己渲染）
- **資料**：
  - CLEVR-CoGenT：官方公開，條件 A／B 對調了顏色與形狀的組合，本身就是組合 split；
  - 公開的 CLEVRTex（約 5 萬張，含逐物件標註）：依物件三元組保留 20% 組合，把含保留組合的場景移到 COOD 集。保留後約剩 25–30% 的場景可用於訓練〔推估：0.8^6 ≈ 0.26〕，約 1.3 萬張；
  - Super-CLEVR（公開，約 3 萬張）：同樣過濾。
  - 問答用 CLEVR 的程式化問題生成器。
  - 代價：數字不能和原論文直接對照，只能比較趨勢。
- **第 1 週**：
  1. 抽取凍結特徵：DINOv2-S/B/L、DINOv3-B/L、SigLIP2-B/L，約 5–10 GPU-h；
  2. 參照組與 dense 表徵各訓練下游 TF 5，每個 run 先跑 150k 步並檢查是否飽和，約 1 GPU-h；3 個資料集 × 7 種表徵 × 3 個 seed ≈ 60 runs，每卡並行 6 個，約 60 GPU-h；
  3. 逐屬性線性探測，可忽略不計。
- **第 2 週**：
  1. 在 2 個資料集上訓練 DINOSAUR 式 slot（凍結 DINOv2-B／DINOv3-B），約 4 × 15 ≈ 60 GPU-h；
  2. 子空間重組原型與 SysBinder 類基線，約 40 runs，40–60 GPU-h。
- **兩週合計約 170–200 GPU-h**〔推估〕。4 張 H200 兩週可用約 1,300 GPU-h，1 張約 330 GPU-h，都夠。

### 5.3 停損判準
- **第 1 週末（任一條成立就停）**：
  - 最大骨幹的組合特有落差（表徵的 ID−COOD 減去參照組的 ID−COOD）在 3 個資料集中有 2 個 < 5 pp：差距已被規模化解決；
  - 參照組本身的 COOD < 85%：資料切法不適合，換切法最多一次，再不行就停；
  - 不同 seed 的 COOD 標準差 > 2 pp，而且無法靠增加 seed 壓下來。
- **第 2 週末**：
  - 重組方法在至少 2 個資料集上追回 **≥ 25%** 的組合特有落差，而且增益大於 seed 標準差的 3 倍；
  - 同時要勝過 SysBinder 類基線至少 2 pp。
  - 沒達到就停：只剩分析，屬於使用者不收的類型。
- **第 4 週末**：真實影像（C-GQA 或 MIT-States）上有一致方向的增益（未見組合準確率至少 +1.5 pp，或 HM／AUC 有提升）。沒有的話，論文只能定位成受控研究，ICML 中稿機率明顯下降，要重新評估。

---

## 6. 信心評級：**中**

**加分**：
- 外部分析的數字全部核實，差距真實存在；
- 拆出 ID 後仍有 10–15 pp 的組合特有落差，題目有實質空間；
- ICML 2026 對組合泛化／綁定給了多篇 spotlight，不冷門；
- 新骨幹還沒有人測；
- 有可以直接借用的理論框架（線性、正交分解）；
- 下游模型很小，算力門檻低。

**扣分**：
- 原論文的程式、資料、splits、權重都查不到，原設定無法便宜重現，只能改用公開資料集；
- 同一條線上有一組 ICML spotlight 團隊在推進，撞題風險中等；
- 方法線上已有 SysBinder 等前作，新意要靠實驗結果撐；
- 「合成玩具」這個批評只能部分回應，真實影像遷移的結果未知；
- 10 週很緊；
- 「大骨幹自己就解了」是未知的致命風險。

**升到「中偏高」的條件**：第 2 週停損判準通過（大骨幹上落差仍 ≥ 8 pp，且原型追回 ≥ 25%）。
**降到「中偏低」的條件**：DINOv3-L 或 SigLIP2-L 讓組合特有落差小於 5 pp，或方法贏不過 SysBinder 類基線。

---

## 7. 外部分析逐條核實總表

| 說法 | 結果 | 層級 |
|---|---|---|
| ICML 2026 物件中心表徵研究 | ✔ arXiv 2602.16689，ICML 2026 一般 poster | 全文＋本機 |
| 表 15 | 數字 ✔；arXiv v1 位於 Table 13，「表 15」編號查不到，無法核實（相機版未取得） | 全文 |
| CLEVRTex-M 76.9 / 97.5 / 20.6 | ✔（TF 5） | 全文 |
| Super-CLEVR-M 66.8 / 91.6 / 24.8 | ✔（TF 5） | 全文 |
| MOVi-C-H 55.3 / 78.8 / 23.5 | ✔（TF 5） | 全文 |
| 參照組直接拿真實屬性 | ✔ | 全文 |
| 凍結特徵加小下游模型 | ✔（2／5 層，hidden 128） | 全文 |
| 600k 步 | ✔（DINOv2 設定；SigLIP2 為 641k／689k） | 全文 |
| 官方程式、splits、資料、權重 | 查不到，無法核實（推定未公開） | 開頁與搜尋 |
| ComPA 52.8% | 數字存在，但是 RL-Ground 方法的最佳組合分數（arXiv 2505.19406）；NeurIPS 2025 收錄查不到，無法核實；題目是技能組合，不是屬性綁定 | 全文 |

主要來源：
- https://arxiv.org/abs/2602.16689
- https://icml.cc/virtual/2026/poster/62839
- https://arxiv.org/abs/2505.19406
- https://arxiv.org/abs/2602.24264
- https://arxiv.org/abs/2605.31503
- https://arxiv.org/abs/2510.24709
- https://arxiv.org/abs/2412.18743
- https://arxiv.org/abs/2607.22705
- https://arxiv.org/abs/2610.22602
- https://arxiv.org/abs/2609.31456
- https://arxiv.org/abs/2609.08886
