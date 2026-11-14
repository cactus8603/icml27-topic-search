# 文獻新穎性查核：ELSA×NCM「尺度樹」理論線（T1／T2／T3）

我只做查核，沒有提出方法。搜尋只用了技術關鍵字。下面每篇文獻都標了查證程度：
- 【開】：用 WebFetch 開過 abstract 或 HTML 全文；
- 【搜】：只看到搜尋摘要，沒有開原文。

---

## 1. 逐篇比對

### 背景兩篇

- **ELSA**（arXiv 2604.23798，CVPR 2026 Findings）【搜】
  - 內容：把 online softmax 寫成 (m,S,W) 上的 monoid prefix scan，區塊內用 Hillis–Steele、區塊間用 Blelloch。
  - 重疊：背景文獻，不適用。
  - 註：補充材料的 monoid 證明我沒有開。
- **NCM**（arXiv 2610.14162，Kumbong 等人，2026-10-09）【開 HTML，含附錄】
  - 同層 code 給定較粗層的累積重建後彼此獨立取樣（§3.2 Eq.5）。
  - 附錄 A.2 有一個 ELBO gap，等於 tokenizer 階層的負 log posterior。這是「把 tokenizer 指派當點質量後驗」造成的 gap，**不是同層 factorization gap**。
  - 層長度表是手調的：
    - 附錄 C 表 5：例如 L=128、K=12。
    - 附錄 F.1：比較了 K∈{4,8,16,32}，以及線性、平方、根號、對數、指數、cosine 六種排程；我讀到的段落沒有寫出哪個最好。
    - 每層 4096 個 code。
  - **論文沒有量測同層 factorization gap。** F.3 只用 code corruption 處理同層錯誤累積。
  - **code 是全域的，不對應文字片段**（例如「coarse levels capture global structure」「refines a representation of the entire board」），也就是說它不是樹。
  - 重疊：T2 與 T3 的空白 NCM 本身沒有補上；T3 的「各層實測 gap」沒做過。
- **NCM 的後續與引用**：用兩次搜尋都**找不到**任何後續或引用它的論文。NCM 才公開約 5 週，可能還沒被收錄；我沒有查 Semantic Scholar 或 Google Scholar 的引用頁。

### 平行可掃描模型

- **Sequential-Parallel Duality in Prefix Scannable Models**（Yau 等人，arXiv 2506.10918，ICLR 2026）【開 HTML】
  - 內容：PSM 把序列切成區塊，用 Blelloch scan 匯合（匯合運算可以不滿足結合律），證明 PSM ∈ SPD(n, log n)。
  - **只談平行訓練與序列推論的狀態計算，完全沒有談取樣、平行解碼或取樣精確性。**
  - 重疊：T1「與 PSM 類別的關係」對方沒做。但兩者是不同的軸（計算條件機率 vs. 取樣平行度），能否寫出有內容的關係定理，本身是個風險。

### Multiresolution Markov tree（Willsky 一系）

- **Willsky, "Multiresolution Markov models for signal and image processing", Proc. IEEE 90(8), 2002**【搜】
  - 內容：塔狀樹上的 MR Markov 過程綜述，連結 1/f、自相似過程、wavelet。
- **Basseville, Benveniste, Chou, Golden, Nikoukhah, Willsky, IEEE Trans. IT 1992；Chou–Willsky–Benveniste–Basseville 1989**【搜】
  - 內容：樹上的 AR／狀態空間模型，尺度間 Markov。
- **Irving & Willsky, "A canonical correlations approach to multiscale stochastic realization", IEEE TAC 46, 2001；Irving–Karl–Willsky 1994**【搜】
  - 內容：在高斯情形下，**選粗層節點狀態，使子節點條件獨立，並把狀態維度降到最低**，方法是 CCA。
  - 這就是 T1「以粗層為分隔集的樹狀 Markov 實現」加上「有界 code 大小」的高斯版本。
- 重疊：
  - T1 的「精確表示條件」與「狀態維度」：**部分已做掉**（高斯、連續狀態）。
  - 離散、有限 code、序列語言的情形：沒看到。

### 樹／張量網路與互資訊 scaling（T1、T2 的最近鄰）

- **Lu, Kanász-Nagy, Kukuljan, Cirac, "Tensor networks and efficient descriptions of classical data"**（arXiv 2103.06872）【開 ar5iv】
  - §V.2–V.3：Lin–Tegmark 式的二元 Markov 樹可以表成 tree tensor network，「預期產生 power-law 相關與 **對數** 互資訊 scaling」，與 WikiText-2 實測不符。
  - 實測文字的互資訊對區塊長度呈冪律 L^ν，指數接近 volume law（約 0.82）。
- **Tensor tree**（arXiv 2408.10669）【搜】
  - 內容：切斷一條 bond，跨切點互資訊 ≤ log(bond dimension)。
- **Pestun & Vlassopoulos, "Tensor network language model"**（arXiv 1710.10248）【搜】
  - 內容：語言互資訊冪律，指數約 0.37。
- **Lin & Tegmark, "Critical behavior in physics and probabilistic formal languages"**（arXiv 1606.06737，Entropy 2017）【搜】
  - 內容：Markov 或正規文法的互資訊指數衰減，PCFG 或深層階層可以呈冪律。
- 重疊：
  - T1 的「有界 code 大小的樹表示類別」與「跨切點互資訊 ≤ 跨越邊數 × log χ」：**物理與張量網路文獻已有，屬部分已做**。
  - T2 的「冪律互資訊 → 固定預算不夠」：**定性結論已有**（對數 vs. 冪律的不匹配）。

### 平行取樣 factorization error（2024–2026）

- **ParallelBench**（Kang 等人，arXiv 2510.04767，ICLR 2026）【開 HTML】
  - Theorem 1：把 Y 分成 T 組平行步驟時，min_θ KL(P_data‖P_θ) ≥ Σ_i E[C(S_i | X, S_<i)]。C 是條件 total correlation，當各 token 邊際正確時等號成立。
  - 重疊：T2 的 KL 骨架**已做掉**。
- **Zhang, Yu, Brekelmans, Tang, Asif, Ver Steeg, "Generation Order and Parallel Decoding in MDMs: An Information-Theoretic Perspective"**（arXiv 2602.00286）【開 HTML】
  - 前向 KL(p‖p_B) 恰好等於 Σ 區塊條件 TC（Eq.16）。
  - 反向 KL 有對應的分解，而且可以任意大（Eq.18）。
  - 驗證成本 ≥ exp(條件 TC)。
  - 條件獨立時沒有偏差（§3.3）。
  - 重疊：T2 的骨架已做掉。
- **Representation-based Masked Diffusion Model, RMDM**（arXiv 2609.12382）【開 HTML】
  - Prop. 3.1：給定 latent z 時，TC 的減少量**恰為** Σ_i I(x_i; z | x_U) − I(x_M; z | x_U)。
  - 只說 latent 容量「決定」殘差相依，沒有給容量界。
  - 重疊：T2 中「粗層 code 吸收多少相依」這一步**部分已做**。由此一行即可推出 TC(X|Z) ≥ TC(X) − (n−1)·H(Z)。
- **Serial Depth of Adaptive Parallel Sampling**（Wen, Liang, Lu, Lin，arXiv 2608.25505，2026-08）【開 abstract】
  - 內容：在前向 KL 預算下，自適應平行取樣所需輪數的下界，以及配套排程；分布族從 log log n 到 n 都有。
  - 不涉及 latent 或階層。
  - 重疊：T1 的 token 層級類比（「哪些分布可以少輪精確或近似取樣」）**部分已做**。
- **Chen & Wang, "Parallelism, critical windows, and separations among diffusion LMs"**（arXiv 2609.20539）【開 alphaxiv】
  - 內容：uniform 與 Gaussian diffusion 需要 Õ(DTC/ε) 次查詢；masked diffusion 有 Ω̃(d) 的下界。
  - 沒有多尺度。
  - 重疊：部分（下界方法論）。
- **Chen, Cong, Li, "Optimal Inference Schedules for MDMs"**（arXiv 2511.04647）【開 abstract】
  - 內容：任意分布與任意排程下期望 divergence 的精確刻畫，以 TC／DTC 給上界，某些情形 O(log n) 步即可。
  - 重疊：部分。
- **Li & Cai**（arXiv 2505.21400，NeurIPS 2025）【搜】
  - 內容：KL 為 O((1/T) Σ I(X_i; X_−i))，宣稱上下界相符。
- **Zhao & Cai**（arXiv 2602.20126）【搜】
  - 內容：Õ(TC/K) 與 Õ(DTC/K)。
- 下列只看到搜尋摘要，**沒開**：
  - DEMASK（2604.02560，只開了 alphaxiv 頁）；
  - Discrete Copula Diffusion（2410.01949）；
  - VMD（2510.23606）；
  - DiLaDiff（2605.23605，以 latent 換取同步條件獨立）；
  - 2603.00045（「factorization barrier」源於結構誤設）。

### VAR／next-scale 理論

- **MVAR**（arXiv 2505.12742，ICLR 2026）【搜】
  - 內容：尺度間 Markov 化，以經驗為主。
- **VAR fine-grained complexity**（arXiv 2501.04377）【搜】
  - 內容：計算複雜度，不是統計誤差。
- **"Multi-scale AR models are Laplacian/discrete/latent diffusion in disguise"**（arXiv 2510.02826）【開 abstract】
  - 內容：abstract 未談同尺度獨立性誤差或排程。
- **Reed 等人, Parallel Multiscale PixelCNN**（ICML 2017）【搜】
- **Locally Hierarchical AR**（NeurIPS 2022）【搜】
- 共同點：這兩篇都**假設**同尺度或底層 code 條件獨立，沒有給界。
- 重疊：VAR 的**同尺度 factorization gap 理論與層長度最佳化，我沒找到任何人做過**。這是 T2、T3 最乾淨的空白。

### Hilberg／互資訊冪律用於模型設計

- **L²M, "Mutual Information Scaling Law for Long-Context LM"**（Chen 等人，arXiv 2503.04725）【開 HTML】
  - 雙分互資訊 ~ L^β（次線性）。
  - Thm 5.2：I^BP ≤ C·dim(z) + log M。
  - Thm 5.4／5.6：模型的歷史狀態必須滿足 dim(z) ≿ L^β。
  - 引用了 Hilberg 與 Dębowski。
  - 重疊：**這就是 T2 的「同構前作」**，只是把「KV／歷史狀態」換成「粗層 code 容量」。T2 的冪律論證**部分已做掉**，差別只在於套用的對象。
- **Dębowski 一系（Hilberg exponents，arXiv 1403.1757 等）**【搜】
  - 內容：Hilberg 猜想的理論與實測。

### 資訊論經典

- **Wyner common information**，以及 **Gastpar & Sula, "Relaxed Wyner's Common Information"**（ITW 2019，arXiv 1912.07083）【搜】
  - 內容：使 A⊥B|Z 所需的最小 I(AB; Z)；relaxed 版本允許 I(A;B|Z) ≤ γ。
- **Op 't Veld & Gastpar**（Allerton 2016）【搜】
  - 內容：推廣到 total correlation。
- 重疊：T2「code 容量 vs. 殘差條件相依」的權衡曲線，**本質上就是 relaxed Wyner／TC 版的 common information**。單一切點的情形是經典結果。
- **Crutchfield 的 E ≤ C_μ**（excess entropy ≤ 統計複雜度）【搜】
  - 內容：預測所需記憶下界等於過去與未來的互資訊。
  - 這就是 T2 單切點不等式在時間序列上的老版本；Hilberg 冪律正是 E 發散的情形。

### 殘差 VQ／階層 VAE 的條件獨立分析

- **HQ-VAE**（arXiv 2401.00365）【搜】
  - 內容：處理 collapse，沒有分析因子化 gap。
- 重疊：沒找到理論分析。

---

## 2. 既有誤差界能否直接推出 T2

**大半可以。**
- 以下兩件事已經是恆等式或定理：
  - 前向 KL 等於各層條件 TC 之和（2602.00286 Eq.16；ParallelBench Thm 1 加上取下確界）。
  - 給定 latent 時 TC 的減少量（RMDM Prop 3.1）。
- 再加上兩個經典不等式：
  - TC ≥ 任一二分割的互資訊；
  - I(A;B|Z) ≥ I(A;B) − H(Z)。
- 合起來幾行就得到「同層 gap ≥ 跨切點互資訊 − 粗層 code 熵」。
- 「固定容量配上冪律互資訊，所以不可避免」也已經有 L²M（以狀態維度表述）、張量網路的 bond dimension 界、E ≤ C_μ 三個先例。

**T2 可能還剩下的新東西：**
- (a) 多層鏈式加總，並以各層 code 數為變數求最佳配置（water-filling 型），然後和 VAR、NCM 的手調排程比較。我沒搜到有人做過。
- (b) 在「encoder 是可學的確定性函數」之下，對所有 tokenizer 取下確界的版本，也就是把 relaxed Wyner／TC common information 接到多尺度 tokenizer 上。
- (c) 這個下界在實務規模下是否非平凡（見第 3 點），以及能否實測驗證。

這些比較像「把已知的不等式組合到新的場景」，數學深度有限。

---

## 3. T2 下界最可能出錯的地方

1. **方向問題，這是最嚴重的一點。**
   - code-level 的 TC 是 code 空間上的 KL，但我們關心的是文字分布的 KL。
   - data processing 只給出 KL(p_X‖q_X) ≤ KL(p_Z‖q_Z)，是**上界而不是下界**。
   - 多組 code 可以 decode 成同一段文字；模型也可能把機率放在「非標準 code」上，卻仍然生成合法文字。
   - 所以由 code TC 推不出文字層的損失下界。
   - 文字層的下界必須直接處理「q_X 是 |Z| 個乘積分布的混合」這個結構，類似 nonnegative rank／latent class 近似，例如 KL ≥ I_p(A;B) − log|Z| 之類。這需要獨立證明，我沒有找到現成定理，也沒有驗證它成立。
2. **NCM 的 code 是全域的，不是樹。**
   - 「樹層 ↔ 尺度層、分隔集 ↔ 粗層」這個對應在 NCM 不成立：每個 code 涵蓋整條序列（論文自述）。
   - 文字的「切點」不對應同層 code 的任何分割，所以「跨切點互資訊」無法直接當作同層 code TC 的下界。
   - 要嘛限定在局部化的樹狀 tokenizer（那就回到 Willsky／張量網路，已有結果）；要嘛另外證明全域 code 的版本。
3. **確定性的殘差 VQ。**
   - Z = f(X) 時 I(X;Z) = H(Z)，I(A;B|Z) ≥ I(A;B) − H(Z) 仍然成立。
   - 但 encoder 可以優化，所以「不可避免」必須對所有 encoder 取下確界，也就是 Wyner 類的量。對確定性 encoder 而言，這個最佳化屬於另一類問題，與隨機 encoder 的 Wyner CI 可能不同。
   - 另外，lossy decoder 的重建誤差會與 factorization gap 混在一起（NCM 的重建率 ≥ 99.95%，不是 100%）。
4. **層間不是嚴格樹狀。**
   - NCM 的每一層都看「所有較粗層的累積重建」。
   - 若再加上 code corruption 或 nested dropout，訓練分布與生成分布本身就不同，前向 KL 等於 Σ TC 的恆等式就不再精確。
   - 該恆等式還假設同層邊際是正確的；學習誤差需要另外一項來處理。
5. **Hilberg 的適用性。**
   - Hilberg 是未證明的猜想，講的是平穩過程裡相鄰區塊的**無條件**互資訊。
   - T2 需要的是**給定上下文或粗層下的條件**互資訊。
   - 各篇估計的指數差很多：約 0.37（Pestun）、次線性（L²M）、約 0.82（Lu 等人）。估計方法有偏差，L²M 自己也說它可能低估了指數。
6. **在實際規模下下界可能是空的。** 以 NCM 的設定粗估：
   - 線性排程 1→128，K=12，碼本 4096 ≈ 12 bit／code。
   - 最後一層之前累積約 600 個以上 code，約 7–9k bit 的容量。
   - 128 個 token 文字的總熵估計只有數百 bit。
   - 所以「互資訊 − 容量」在 NCM 實際的運作區間幾乎必為負，下界等於 0。
   - 這是我自己用論文附錄數字做的粗估，沒有實測。
   - 結論是 T2 只在「每層 code 預算很小」或「序列很長」的極限才有咬合力。
7. **前向與反向 KL。** 2602.00286 指出反向 KL 可以任意大。界要說清楚是哪個方向；兩個方向的意義不同（mode covering 與 coherence）。

---

## 4. 總評：新穎性風險「中偏高」

- T2 的核心不等式由已知結果幾行就能推出：
  - ParallelBench／2602.00286 的 KL = Σ 條件 TC；
  - RMDM 的 TC 減少恆等式；
  - Wyner CI 與 E ≤ C_μ；
  - 「容量 vs. 冪律互資訊」的論證有 L²M 這個幾乎同構的前作。
- T1 在高斯情形有 Irving–Willsky，在樹或張量網路有 bond dimension 界與「Markov 樹只能給對數互資訊」的結果。
- 與 PSM 的連結是類比而不是定理；PSM 根本沒有談取樣。
- 真正空白的部分：
  - VAR／NCM 同尺度 gap 的理論與實測；
  - 各層 code 數的最佳配置；
  - 全域（非樹）code 的下界。
- 但這個空白要以理論為主撐起 ICML，前提是：解決第 3 點的方向問題，並證明下界在實際規模下非平凡。否則審稿人很容易說「這是已知不等式的組合」，或「這個界對 NCM 是空的」。
- 若轉成「理論加實測」：T3 量測每層 gap，再加上依理論設計的排程，在小預算下優於手調表。新穎性風險可以降到「中」。

---

## 沒能確認的事項

- ELSA 的補充材料（monoid 證明）沒開。
- NCM 附錄 F.1.2 的六種排程哪一個最好，文字沒寫，我也沒看 Figure 10。
- NCM 的後續與引用論文搜不到，可能是還沒收錄；沒有查引用資料庫。
- 只有搜尋摘要、沒開原文的：
  - Willsky 2002 綜述與 Irving–Willsky 2001 的內文；
  - Lin–Tegmark；
  - Gastpar–Sula、Op 't Veld–Gastpar；
  - Crutchfield E ≤ C_μ 的原文；
  - Li–Cai、Zhao–Cai；
  - MVAR；
  - HQ-VAE；
  - DiLaDiff、2603.00045、Discrete Copula、VMD；
  - 張量樹 2408.10669、Pestun 1710.10248。
- 2608.25505、2511.04647、2510.02826 只看了 abstract，內文可能有相關內容。
- 「文字層 KL ≥ I − log|Z|」這類混合乘積分布的下界，我沒找到現成定理，也沒驗證它成立。
- 第 3 點第 6 項的容量與熵數字是我的粗估，不是實測。
