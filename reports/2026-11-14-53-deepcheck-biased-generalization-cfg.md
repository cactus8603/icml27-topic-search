# 深查：擴散模型「偏差泛化」在條件生成與 CFG 下的行為（候選 2，來自 52 號掃描）

日期：2026-11-14｜範圍：不動 GPU、不寫程式｜網路操作約 22 次（arXiv HTML 全文 3、arXiv API 7、Semantic Scholar 3（2 次 429）、OpenReview 4、GitHub 1，其餘為重試）
來源層級標記：〔全文〕讀過 arXiv HTML 原文相關段落；〔摘要〕arXiv／OpenReview 摘要；〔標題〕只核實了 arXiv 編號與標題，內容描述依既有知識；〔API〕Semantic Scholar 回傳；〔推估〕我的估計，沒有查證。

---

## 0. 結論先講

- **信心：中**（52 號掃描給的「中偏高（偏下緣）」要下修）。原因不是原論文有問題，而是 2026 年 3–11 月這個方向變得很擠，而且掃描提的主打方法（單模型訓練／留出去噪誤差差距）**已經被至少三篇工作直接或間接做掉**。
- 三個關鍵發現：
  1. **偵測量撞題比預期嚴重。**「訓練與留出樣本在中間噪聲水平的去噪誤差差距」就是 2603.13419（Beyond Sample Copying，原論文的引用者之一，52 號誤標為「解析理論」）摘要裡的核心觀察；單模型、跨擴散時間校正的版本是 2610.40616（TSC，2026-10-29）；「記憶出現前，用 score divergence 在單模型上分開訓練／留出樣本，並轉成無標籤 MIA」是 2610.11670（2026-10-08，有引用原論文）。所以「單模型偏差偵測量」作為主貢獻已站不住，只能當工具或基線比較。
  2. **CFG × 偏差泛化的「訓練動態」仍是空的。**目前查到碰 CFG 的都是推論期、逐字記憶（2411.16738、2509.25705、Wen 等 2407.21720），或條件模型的隨機特徵理論（2610.11288，沒有 CFG、沒有訓練時間軸）。沒有人用 sample-split 量「條件分支／無條件分支／不同 guidance 強度」的偏差起點如何隨訓練移動。原論文結論段明確把這件事列為 open direction，限制段也明說「primarily diagnostic，不提出 mitigation」〔全文〕。
  3. **原作設定極小、程式公開、可直接延伸到條件版。**CelebA 32×32 灰階、每份 n=1024、U-Net base width 32、batch 512、5000 epochs（約 1 萬步）、15 個模型〔全文〕；CelebA 程式基於 Bonnaire 等的公開 repo，階層資料、BP 精確分數程式在 github.com/davide-beltrame/biased-generalization（HTTP 200，本輪確認存在）。階層資料的「根節點符號」天然就是類別條件，BP 能算精確的條件與無條件分數，所以**精確的 CFG oracle 可得**，這是本題最強的差異化資產。
- **建議切入角度（要升到中偏高時）**：把題目從「CFG 版＋單模型偵測量」改成「**條件與無條件分支各有自己的偏差時鐘，CFG 外插放大的正是條件分支提早出現的樣本特定成分**」。主貢獻是機制＋可驗證的定量預測（偏差起點隨每類樣本數 n/K 縮放、幅度隨 guidance 權重放大），偵測與緩解是次要貢獻，並且明確以既有 MIA／記憶偵測為基線。詳見 §5。

---

## 1. 原論文 2603.03469 核實（Biased Generalization in Diffusion Models，ICML 2026 spotlight）

### 1.1 實驗設定與算力〔全文〕
- 真實資料：CelebA，置中裁切、灰階、縮到 32×32，無資料增強；每個模型用 n=1024 的不相交子集；共 15 個模型、105 對（sample-split）。
- 模型：標準 U-Net，base width 32、channel multiplier (1,2,4)，16×16 與 8×8 有 self-attention；DDPM，T=1000、線性 β 1e-4→0.02、ε-prediction。
- 訓練：Adam lr 1e-4、batch 512、5000 epochs（n=1024 時約 1 萬步）。
- 指標：(a) 樣本層級：固定起始噪聲與反向噪聲，比較兩個模型生成影像的 cosine 距離；(b) 分數層級：同一個加噪輸入下兩模型 ε 預測的 cosine 距離，t=50、150 最明顯；並分「輸入來自 A 的訓練集」與「來自第三份資料」兩組。
- 關鍵結果：樣本層級 cosine 距離的 U 形最低點明顯早於 DSM test loss 最低點；Eq.(3) 的標準複製指標在所有 checkpoint 平均都低於 1e-3（也就是沒有逐字記憶）。
- 受控實驗：階層樹資料（離散序列，one-hot 後做連續擴散），8 層 transformer、hidden 512；n=5000／12000／70000 分別訓練 20000／30000／35000 epochs；BP 給精確後驗平均，BP_k 給濾掉高層關聯的粗化 oracle。
- 算力：論文沒寫 GPU 小時。依上述規模，CelebA 單一模型應在 1 GPU 小時內〔推估〕。
- 程式：CelebA 部分基於 github.com/tbonnair/Why-Diffusion-Models-Don-t-Memorize，作者修改版與階層資料、BP 實作在 github.com/davide-beltrame/biased-generalization〔全文〕；repo 存在〔本輪 HTTP 檢查〕。

### 1.2 限制段與 future work〔全文〕
- 限制：只研究標準 DDPM；資料與架構是中等規模，沒看大型高維模型；**「分析主要是診斷性的，不提出 mitigation 或替代訓練目標」**；偏差只用距離與分數指標，抓不到語意相似。
- 結論段 open direction：條件生成與 guidance（明點 CFG）可能「選擇性地把生成導向對特定訓練資料過擬合的特徵」，並把「偏差泛化、條件化與資料萃取的交互作用」列為 future work。
- 52 號掃描這兩點描述正確。

### 1.3 52 號掃描需要更正的地方
- 「原文 Figure 2 已顯示中間時間點差異最明顯」：Figure 2 量的是**兩個模型之間**的分數 cosine 距離，再依輸入來源分訓練／測試；不是單一模型的訓練／留出 loss 差距〔全文〕。原文真正接近單模型訊號的是 §4.3 的 excess loss 分解（需要 BP oracle，只在合成資料可用）。
- 引用者「兩篇解析理論」：其中 2603.13419 不是純解析，它的主要實證就是「中間噪聲水平的訓練／驗證去噪 loss 差距」〔摘要〕，正好是本題原本的偵測量。
- Semantic Scholar 引用者清單不完整：至少 2610.11670 在參考文獻與相關工作段引用原論文〔全文〕，但 S2 沒列。

---

## 2. 2411.16738 與記憶／成員推論文獻：本題偵測量的區隔

### 2.1 2411.16738（CFG inside the Attraction Basin May Cause Memorization）〔全文〕
- 設定：全部是**已訓練好**的文字生圖模型（SDv2.1 在 LAION-10k、Imagenette 上微調；SDv1.4 在重複提示資料上微調；SDv1.4 預訓練的 500 個記憶提示）。
- 觀點：反向軌跡早期有「吸引盆」，在盆內套 CFG 會被拉向記憶影像；盆的出口（transition point）可由 ‖ε(x,c)−ε(x,∅)‖ 的驟降偵測。方法：先不套 CFG，過 transition point 再套；另提 opposite guidance。Imagenette 情境是固定 transition point（t=700／600）。
- 程式公開（github.com/SonyResearch/mitigating_memorization）。
- 與本題的關係：它處理的是**逐字複製、推論期**；完全沒有訓練時間軸，也沒有「還沒複製之前的偏差」。但「只在部分時間區間套 guidance」作為緩解手段已被它與 2404.07724（limited guidance interval）佔走，所以 52 號提的「依時間步調整的 guidance」**不能當新方法**，只能當基線。

### 2.2 記憶偵測與成員推論主要文獻（編號已核實）
| 工作 | 編號 | 訊號 | 與本題偵測量的關係 |
|---|---|---|---|
| Extracting Training Data from Diffusion Models | 2301.13188〔標題〕 | 生成＋過濾萃取；loss 型 MIA、LiRA | 原論文的動機來源；loss-threshold MIA 就是「訓練 loss 比留出低」 |
| Membership Inference Attacks against Diffusion Models | 2302.03262〔標題〕 | 單一 t 的去噪 loss 門檻 | **幾乎等於本題原偵測量的逐樣本版** |
| Are Diffusion Models Vulnerable to MIA?（SecMI） | 2302.01316〔標題〕 | 確定性前向／反向的 step-wise 後驗估計誤差 | 可直接當基線 |
| PIA（Proximal Initialization Attack） | 2305.18355〔摘要〕 | 用 t=0 的 ε 初始化軌跡，兩次查詢 | 可直接當基線，便宜 |
| White-box MIA against Diffusion Models | 2308.06405〔標題〕 | 梯度型 | 可選基線 |
| Membership Inference of Diffusion Models | 2301.09956〔標題〕 | loss／likelihood 型 | 同上 |
| Wen 等，Detecting, Explaining, and Mitigating Memorization | 2407.21720〔標題；內容依 2411.16738 全文轉述〕 | ‖ε(x,c)−ε(x,∅)‖ 對記憶提示異常大；訓練期依此門檻剔除 batch 樣本 | **與本題「條件 vs. 無條件分支」切角最接近**，必須當基線，並說清楚差異（它偵測已記憶提示，不是訓練過程中的偏差起點） |
| TSC / RMC（Difficulty Calibration Across Time） | 2610.40616〔摘要〕 | 單模型，跨擴散時間校正 loss；或拿早期 checkpoint 當參照 | **直接撞到「單模型訓練／留出差距」**；RMC 等於「拿早期 checkpoint 當參照」 |
| Early Signatures of Memorization（basin、cyclic denoising） | 2610.11670〔全文部分〕 | 單模型 normalized score divergence，在第一個記憶樣本出現前分開訓練／留出；附錄 G.4 轉成無標籤 MIA；SD v1.4 上的 cond–uncond divergence gap AUC 0.944 | **直接撞到「記憶前的單模型偏差訊號」**；但它是無條件訓練為主，沒有 CFG 強度 × 訓練時間 |
| Beyond Sample Copying（Structural Memorization） | 2603.13419〔摘要〕 | 中間噪聲水平的訓練／驗證 loss 差距 | **原偵測量本身**；也在 ICLR 2027 投稿中 |

### 2.3 判斷
- 52 號提的偵測量（訓練／留出去噪誤差差距）在數學上就是 loss-based MIA 的群體平均版（membership advantage 的期望）。審稿人一眼會指出這點；而且 2603.13419、2610.40616、2610.11670 都在 2026 年做了它或它的改良版。**不能當主貢獻。**
- 可行的處理：把這些當**基線**，題目的方法部分改成只有條件模型才有的訊號（見 §5）。若新訊號在「預測 sample-split 偏差起點」這件事上贏不了 loss gap／TSC／score divergence，就老實報告，方法部分降成附屬。

---

## 3. 撞題檢查（2026 年 9–11 月）

### 3.1 原論文引用者
- Semantic Scholar（第 3 次才成功）列 5 篇〔API〕：2611.12156（動力系統生成模型的特徵學習與 collapse，理論）、2611.02955（長程關聯地形中的擴散，統計物理）、2610.13786（離散圖擴散的「條件記憶」，訓練偏差、早停無法緩解）、2605.21402（記憶、收斂與泛化，理論）、2603.13419（見上）。
- S2 漏列：2610.11670（有引用原論文）〔全文〕。
- 沒有任何引用者做條件生成或 CFG。

### 3.2 arXiv 依日期搜尋（相關且近期）
| 編號 | 日期 | 內容 | 對本題的威脅 |
|---|---|---|---|
| 2610.11288 | 2026-10-08 | 條件擴散的隨機特徵理論：越有資訊的條件，在越小寬度就記憶；「malign generalization」〔摘要〕 | 中：佔了「條件化讓記憶提早」的理論說法（以寬度為軸，不是訓練時間，沒有 CFG） |
| 2610.11670 | 2026-10-08 | 見 §2.2 | 高（偵測量） |
| 2610.02780 | 2026-10-02 | QGD 優化器延後記憶＋Copy-Budgeted Selection 做 checkpoint 選擇，CIFAR-10 2000 張子集，有 flow matching 遷移〔摘要〕 | 中高：佔了「依記憶預算選 checkpoint／早停」 |
| 2611.01775 | 2026-11-01 | 記憶盆的頻譜特徵，涵蓋條件、無條件、rectified flow，訓練期與推論期介入〔摘要〕；也投了 ICLR 2027 | 中：有條件模型但看的是逐字記憶盆 |
| 2610.40616 | 2026-10-29 | TSC／RMC（見 §2.2） | 高（偵測量） |
| 2609.35377、2609.26392 | 2026-09 | NTK 頻譜解釋「先學後記」；雙降與 malign overfitting〔摘要〕 | 低：無條件、理論 |
| 2605.30642 | 2026-05 | 在 RHM 階層資料上：常見子字串先被記，部分記憶期造成「slop」〔摘要〕 | 低中：同類階層資料，但無條件 |

- 沒找到「CFG 強度 × 訓練 checkpoint × sample-split 偏差」的工作，也沒找到「偏差感知早停」用在條件模型的工作。

### 3.3 OpenReview ICLR 2027 投稿（關鍵字搜尋，4 次）
- 直接相關：Early Signatures of Memorization（=2610.11670）、Spectral Signatures of Memorization Basins（=2611.01775）、Beyond Sample Copying（=2603.13419）、**Orthogonality of Generalization and Memorization Trajectories in Diffusion Models**（把訓練軌跡分解成朝 population 與 empirical denoiser 的兩個正交方向，並提出 DAO 正則化緩解記憶〔摘要〕）、Local Coverage Governs Memorization（多類別下，類內多樣性高的類別記憶更強〔摘要〕）、How Excess Latent Dimensionality Delays Memorization、Finite-Sample Training … in Transformer DDPMs、Understanding Mode Collapse … under CFG（CFG 的解析模型，不談記憶）。
- 「Orthogonality …」的「泛化與記憶是正交方向」與原論文的 key takeaway 幾乎同一句話，而且已附緩解方法。這進一步壓縮了「無條件偏差＋緩解」的空間。
- 沒看到條件模型偏差起點 × CFG 的投稿。ICLR 2027 投稿數上萬，關鍵字搜尋有漏的可能。

### 3.4 小結
「無條件模型的偏差／潛在記憶偵測」與「延後記憶／選 checkpoint」在 2026 秋天已經是擠的子領域（至少 6 篇）。「條件化與 CFG 怎麼改變偏差泛化的時間軸」仍空，但鄰近的理論（2610.11288）與局部覆蓋觀點（Local Coverage）已靠近，預期 ICML 2027 會有人同時做。撞題可接受，前提是用下面的切入角度。

---

## 4. 最強反駁與回應

### 反駁 A（最強）：「你的偵測量就是 loss-based 成員推論，2603.13419／2610.40616／2610.11670 都做過。」
- 回應：同意，所以不把它當貢獻。新題的偵測訊號只在條件模型存在：同一網路的條件分支與無條件分支（condition dropout 訓練出來）看到的有效樣本數不同（每類 n/K vs. 全部 n），所以兩者的偏差時鐘不同；「條件分支相對無條件分支」的訓練／留出差距，等於在單一模型內拿到一個內建的低偏差參照，不需要第二個不相交資料模型。這跟 Wen 等的 ‖ε_c−ε_∅‖ 是同一類量，差別在於：(1) 我們看它**沿訓練時間**怎麼變、能否預測 sample-split 偏差起點；(2) 比較的是訓練／留出差距，不是單一提示的幅度。實驗上必須把 loss-gap MIA、PIA、SecMI、TSC、score divergence、Wen 等都放進同一張「預測偏差起點」的表。
- 風險：如果這個內建參照贏不了 TSC，方法部分就只剩「條件版基線比較」，論文重心落在機制。

### 反駁 B：「CFG 放大記憶早就知道了（Carlini 2023、Somepalli、2411.16738、2509.25705）。」
- 回應：那些都是已訓練模型、逐字複製、推論期。本題問的是：(1) 在 test loss 還在下降、還沒有任何逐字複製時，guided sampler 的偏差幅度是否隨 w 單調變大；(2) 偏差起點是否隨每類樣本數 n/K 而不是 n 縮放。這兩個是可被否證的定量預測，原論文結論段只是推測。

### 反駁 C：「32×32 灰階、n=1024 太小，推不到實務。」
- 回應：主實驗用 CIFAR-10 類別條件（每類 100–500 張）與階層資料（精確 oracle）；外部效度用 Imagenette 或小資料 LoRA 微調 SD（2411.16738 的情境 2 已證明這個設定會記憶、程式公開）。不宣稱推到大模型預訓練。

### 反駁 D：「這只是把已知的『記憶時間 ∝ n』套到每類樣本數，很平凡。」
- 回應：若結果真的只是 n→n/K 的換算，那就是一個乾淨、可用的實務規則（「條件模型要依每類樣本數設早停」）；但 CFG 的外插係數 (1+w) 會把條件與無條件分支的差放大，預測上 guided 偏差應**早於**條件分支本身，這不是單純換算。階層資料的 BP 精確條件分數可以直接檢驗這點。

### 失敗時能留下的正面貢獻
- 條件化的階層資料基準：根符號當類別，BP 給精確條件／無條件／CFG 分數，可量「偏差」而非只有「複製」。這是可重用的工具（原作 repo 已有大半）。
- 條件模型上各種 MIA／記憶偵測量預測偏差起點的系統比較（正面的 benchmark 結果，即使新訊號沒贏）。
- 條件模型的早停／guidance 區間實務建議（例如依 n/K 設定 checkpoint 選擇，或依測得的偏差調 condition dropout），對照 2411.16738 與 2404.07724。
- 注意：若第 2 週發現 CFG 對偏差起點與幅度都沒影響，剩下的就是純負面分析，不符合使用者條件，應停損。

---

## 5. 建議切入角度（升到中偏高的條件）

一句話：**條件擴散模型的偏差泛化有兩個時鐘：條件分支依每類樣本數提早進入偏差期，CFG 的外插把這個樣本特定成分放大；據此提出條件模型專用的單模型偏差訊號與早停／guidance 規則。**

1. 機制（主）：在階層資料上把根符號當條件，訓練帶 condition dropout 的 transformer denoiser；用 BP 精確算條件、無條件、CFG oracle，分別量三者相對 oracle 與 sample-split 的偏差起點。預測：條件分支起點 ∝ n/K；guided 起點更早、幅度隨 w 增加。
2. 真實資料驗證：CIFAR-10 類別條件（K=10），每類 n_c ∈ {100, 400}，兩份不相交資料，取樣 w ∈ {0, 1, 3}，複製原作的樣本層級與分數層級 sample-split 指標。
3. 方法（次）：條件 vs. 無條件分支的訓練／留出差距當單模型偏差訊號，用來選 checkpoint 或調整訓練後期的 condition dropout；與 loss-gap MIA、PIA、SecMI、TSC（2610.40616）、score divergence（2610.11670）、Wen 等比較「預測 sample-split 偏差起點」的準確度。緩解以 limited-interval guidance（2404.07724）、transition point（2411.16738）為基線。
4. 不要做：flow matching 推廣（2611.01775、2610.02780 已經碰到）、通用無條件偵測量、大型文字生圖預訓練。

---

## 6. 前兩週停損實驗

| 週 | 實驗 | GPU 小時 | 通過判準 |
|---|---|---|---|
| 1 | (a) 用原作 repo 重現 CelebA sample-split，但只訓 6 個模型（15 對），確認偏差最低點早於 test loss 最低點；(b) 階層資料加根符號條件與 condition dropout（p=0.1），n=5000，3 seeds，BP 算條件／無條件 oracle | (a) 約 5–10；(b) 約 10–20（8 層 transformer，2–3 萬 epochs）〔推估〕 | (a) 兩個最低點的 epoch 差 > 2 倍種子標準誤；(b) 程式能算出條件分支與 CFG oracle 的偏差曲線。(a) 失敗代表 pipeline 有問題，先修再往下 |
| 2 | (c) 階層資料：比較條件分支、無條件分支、w∈{0,1,3} 的偏差起點；(d) CIFAR-10 類別條件，n_c∈{100,400}，2 份不相交資料 × 3 seeds（12 個模型），每個模型訓到 test loss 最低點後再多 50% | (c) 約 5；(d) 約 50–80（小 U-Net，32×32，每個 4–7 小時）〔推估〕 | **主判準**：條件分支或 guided 的偏差起點比無條件早，或 test-loss 最低點時的偏差幅度隨 w 單調增加，效果 > 2 倍種子標準誤，且在 (c)(d) 至少一個成立 |
| 3（預告） | 單模型訊號 vs. 基線 | 約 10（只算推論）〔推估〕 | 新訊號與 sample-split 偏差的跨 checkpoint Spearman ρ ≥ 0.7，且比 loss-gap／TSC 高 ≥ 0.1，或起點預測誤差較小 |

- 前兩週合計約 70–115 GPU 小時〔推估〕，1–2 張 H200 可在兩週內完成。
- 停損：
  - 主判準在 (c)(d) 都不成立 → **放棄**（剩下的只是「CFG 不影響偏差」的負面結果）。
  - 主判準成立但第 3 週新訊號輸給 TSC／score divergence → 繼續做，但論文定位改成「機制＋條件模型上的偵測基準」，信心維持中。
  - 主判準成立且新訊號贏 → 信心升到中偏高。

---

## 7. 信心評級

- **目前：中。**
- 下修理由：主打的偵測量撞到 2603.13419、2610.40616、2610.11670；「延後記憶／選 checkpoint」有 2610.02780 與 ICLR 2027 的 DAO；「泛化與記憶正交」也已有 ICLR 2027 投稿附方法。
- 支撐理由：CFG × 訓練時間軸仍空；原作明列為 future work 且自承沒有 mitigation；程式公開、規模極小、BP 精確條件 oracle 是別人沒有的實驗工具；兩週內可判生死。
- 升到中偏高的條件：照 §5 改角度，且第 2 週主判準通過。

## 沒能確認的事項
- 2301.13188、2302.01316、2302.03262、2308.06405、2407.21720 只核實了編號與標題，方法描述依既有知識與 2411.16738 的轉述。
- 2610.11288、2610.02780、2610.40616、2611.01775、2603.13419 只讀摘要；它們是否已含 CFG 強度實驗未確認。2610.11670 只讀了部分全文（相關工作、附錄 G.4、算力）。
- OpenReview 只用關鍵字搜尋，ICLR 2027 投稿數上萬，可能漏掉標題不顯眼的條件／CFG 版本。
- 所有 GPU 小時都是〔推估〕，沒有實測吞吐量；原論文沒報算力。
