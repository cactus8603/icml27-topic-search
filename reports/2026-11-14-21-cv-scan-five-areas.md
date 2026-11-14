## 報告 A：生成模型

範圍 A 探查報告（2026-11-14）

我只找到 2 個值得考慮的候選題目，沒有湊第三個。這個範圍在 2025–2026 年極度擁擠：CFG、one-step/MeanFlow、REPA、tokenizer、pixel-space、VAR、DMD 這幾條線，每條在 2026 下半年都有多篇新 arXiv，而且常是大實驗室在做（見文末不建議清單）。下面標「已開啟」的論文，我實際打開過 arXiv 摘要頁或 HTML 全文；標「據搜尋摘要」的，我只看過搜尋結果的片段。

---

### 候選 1：ImageNet 規模 latent DiT 的「泛化機制」能否用閉式去噪器解釋，殘差是否集中在一個臨界噪聲窗口

**一句話主張**：在 ImageNet-256 類別條件 latent DiT/SiT 上，有一個閉式去噪器（不需訓練網路），由類別條件的二階統計（Wiener）加上局部 patch 後驗組成。它在臨界窗口以外，對訓練網路的 x0 預測能達到接近「兩個獨立訓練模型彼此之間」上限的 r²。網路與閉式模型之間的落差集中在一個窄的中段噪聲窗口。這個窗口的位置也可以預測 CFG 有效區間，以及 2-NFE 取樣時中間時間點的最佳位置。

**為何可能為假、誰相信相反的事**
- Kamb & Ganguli（ICML 2025）明說他們的理論只涵蓋純卷積模型。對帶 self-attention 的 UNet，他們的閉式模型 median r² 只有約 0.77，並認為 attention 會「從 patch mosaic 中雕出語意一致的物體」，也就是非局部的語意作用無法用局部模型解釋。
- Lukoianov 等人（NeurIPS 2025）認為 locality 來自資料統計，與 Kamb 的「架構歸納偏置」說法相反。但他們自己的解析去噪器在 CIFAR-10 也只有 r²=0.589，而兩個不同 seed 的 DDPM 彼此之間是 0.852。資料一複雜，閉式模型就可能失效。
- Bertrand 等人（NeurIPS 2025）主張泛化主要發生在 flow matching 軌跡的早段，和「中段窗口」的說法可能衝突。
- 實驗結果也可能是落差在所有 t 都很大、沒有窗口結構，那主張就不成立。

**最接近的工作**
1. Kamb & Ganguli, "An analytic theory of creativity in convolutional diffusion models", ICML 2025, https://arxiv.org/abs/2412.20292 （已開啟）。提出 ELS 閉式機器，在 MNIST/CIFAR/CelebA 32px 上對 CNN 的 r² 約 0.90–0.96，對 UNet+SA 約 0.77。差別：只做 CNN、小解析度，沒有 latent、沒有條件。
2. Niedoba, Zwartsenberg, Murphy, Wood, "Towards a Mechanistic Explanation of Diffusion Model Generalization", ICML 2025, https://arxiv.org/abs/2411.19339 （據搜尋摘要）。用 patch 經驗去噪器（PSPC）近似網路。差別：小資料集、無條件、沒有 DiT。
3. Lukoianov, Yuan, Solomon, Sitzmann, "Locality in Image Diffusion Models Emerges from Data Statistics", NeurIPS 2025, https://arxiv.org/abs/2509.09672 （已開啟 HTML）。提出 Wiener 式解析去噪器，Table 1 見上。他們把條件生成、高階統計、複雜架構列為 future work，所以原作者自己可能接著做。差別：只有 CIFAR、CelebA-HQ/AFHQ 64px，也有 CIFAR 上的 DiT 但只比 sensitivity field，沒有 latent 和 ImageNet。
4. Zhang, Hu, Liu, Okyay, Gao, "Concurrence of Symmetry Breaking and Nonlocality Phase Transitions in Diffusion Models", arXiv 2605.04830, 2026-05 （已開啟 HTML）。這是最接近的撞題：他們在 DiT-XL（ImageNet 256）和 SD3 上把 attention 截成半徑 r 的窗口，發現 locality gap 在 t≈0.2 附近大、早期和晚期小，也連結到 guidance interval。差別：他們用的是截斷網路自己的 attention，不是閉式或資料驅動的去噪器；結果大多只畫熱圖、只示範一個類別；也沒有回答「是資料統計還是架構」。
5. Hu, Liu, Zhang, Gao, "Local Diffusion Models and Phases of Data Distributions", arXiv 2508.06614（v2 2026-04，已開啟摘要）。提出相變理論：遠離轉變點時局部小網路就夠。Bradley, "Local Mechanisms of Compositional Generalization in Conditional Diffusion", arXiv 2509.16447（v3 2026-03，已開啟）。研究條件 locality，但只用 CLEVR，並分析 SDXL。

**改進空間的證據**
- CIFAR-10 上最好的解析模型 r²=0.589，兩個 seed 的網路互比 0.852，差 0.26。CelebA-HQ 是 0.902 對 0.981（Lukoianov Table 1）。
- Kamb & Ganguli 對 attention 模型只有約 0.77。
- 在 ImageNet、latent、類別條件的設定下，我找不到任何閉式模型的數字，等於完全空白。

**停損實驗（一週，1–2 張 GPU）**
- 用公開的 SiT-XL/2 與 DiT-XL/2 checkpoint（兩者都在 ImageNet-256 的 SD-VAE latent 上訓練）。編碼全部 1.28M 張圖的 latent（4096 維，fp16 約 10GB，放得進一張 H200）。
- 在約 20 個 t 上，對 128 個樣本計算三種閉式去噪器：(a) 經驗最優去噪器，(b) 類別條件 Wiener，(c) 類別條件 Wiener 加 patch 後驗。
- 量測它們對 SiT-XL 輸出的 r²，並以 DiT-XL 與 SiT-XL 互比的 r² 當上限。
- 放棄條件：
  1. 最佳閉式模型在多數 t 的 r² 都低於約 0.5，而且落差在各 t 上均勻、沒有窗口結構；
  2. 或者窗口位置跟 guidance interval（DiT-XL 條件窗口約 t≈0.2–0.4）完全對不上。
- 反過來，如果閉式模型在窗口外已經接近上限，就繼續做。

**需求**
- 模型：SiT-XL/2、DiT-XL/2，可加 REPA SiT 的公開權重。上限也可以用自己訓練的 SiT-B 第二個 seed。
- 資料：ImageNet-1k。
- 指標：r²/MSE 對 t 的曲線；窗口內和窗口外的 FID 干預（把窗口外換成閉式去噪器後的 FID）；2-NFE 中間點的掃描。
- 估計 200–500 GPU 小時（latent 編碼與 patch 檢索加起來幾十小時；若自訓 SiT-B 第二 seed 約 150 小時）。
- 審稿人預期的 baseline：ELS（Kamb）、PSPC（Niedoba）、Lukoianov 的解析去噪器、經驗最優去噪器、2605.04830 的截斷 attention 診斷。

**ICML 適配度**：高。同類論文 Kamb & Ganguli 與 Niedoba 都在 ICML 2025，Lukoianov 與 Bertrand 在 NeurIPS 2025。這是 ML 理論與機制型的題目，不是 CVPR 型。

**主要風險**
- 2605.04830 的團隊（他們已有 2508.06614 的理論）或 Sitzmann 組可能先做到 ImageNet。
- SD-VAE latent 不是像素，二階統計可能很差，導致 r² 全面偏低。
- 純分析型論文要有一個乾淨的「預測可驗證」結果才過得了審稿，例如用窗口預測 CFG 區間或 2-NFE 中間點，否則容易被評為「測量」。

**信心：中低**。改進空間有明確數字，ImageNet 規模也確實沒人做；但結果可能是負面的，而且這條線在 2026 年有至少 5 篇相鄰工作。

---

### 候選 2：影像自回歸模型（IAR）比 diffusion 記憶更多，原因是「乾淨 context 的 teacher forcing」而不是離散 token 或 loss 型態

**一句話主張**：在相同資料、相同參數量、相同 tokenizer 的控制實驗中，IAR 遠比 diffusion 更早開始記憶（τ_mem），主因是訓練時 context 沒有被破壞，而不是 categorical cross-entropy 或模型大小。只要在 AR 訓練中加入 diffusion 或 mask 式的 context 破壞，就能恢復 diffusion 那種 τ_mem ∝ n 的延遲記憶，而生成品質不變。

**為何可能為假、誰相信相反的事**
- Kowalczuk 等人（ICML 2025）把 IAR 洩漏較多歸因於幾個混雜因素：輸出端可取得 likelihood、AR 每次更新的訓練訊號較多、模型大小、以及「是不是 IAR」這個指標本身。
- 他們把 MAR 洩漏低歸因於 diffusion loss 取代了 CE。但 MAR 同時也是 masked context，所以這兩個因素是混在一起的。我的主張若成立，代表他們的歸因有錯；若不成立，代表答案就是 loss 型態。
- MIA 的數字差異也可能只是攻擊者能取得 likelihood 造成的，不代表真實的記憶差異。必須用不依賴攻擊的指標（生成樣本的最近鄰比例、train/test loss gap）才能判斷。

**最接近的工作**
1. Kowalczuk, Dubiński, Boenisch, Dziedzic, "Privacy Attacks on Image AutoRegressive Models", ICML 2025, https://arxiv.org/abs/2502.02514 （已開啟，含 HTML）。量測 IAR 與 DM 的 MIA、dataset inference、資料萃取。差別：只量測和推測原因，沒有控制實驗。
2. Kasliwal, Boenisch, Dziedzic, "Localizing and Mitigating Memorization in Image Autoregressive Models", arXiv 2509.00488，ICML 2025 workshop （已開啟）。在 VAR/RAR 中定位記憶神經元並剪除，VAR-d30 可萃取圖從 672 降到 110（據搜尋摘要）。差別：做事後緩解，不談機制起源。
3. Bonnaire 等人, "Why Diffusion Models Don't Memorize: The Role of Implicit Dynamical Regularization in Training", arXiv 2505.17638（發表處據搜尋為 NeurIPS 2025，未確認；另有 J. Stat. Mech. 版）。結論是 τ_mem ∝ n、τ_gen 為常數，實驗用 CelebA 32px 灰階、n 從 128 到 32768（據搜尋摘要）。差別：只有 diffusion，作者也承認分析用 SGD、沒有涵蓋 Adam。
4. Favero, Sclocchi, Wyart, "Bigger Isn't Always Memorizing: Early Stopping Overparameterized Diffusion Models", arXiv 2505.16959 （已開啟）。記憶時間隨 n 增加，涵蓋影像與語言 diffusion。差別：沒有自回歸對照。
5. Luo, Yu, Li, Bjerva, "Characterizing Memorization in Diffusion Language Models: Generalized Extraction and Sampling Effects", arXiv 2603.02333, 2026-03 （已開啟，arXiv 頁沒寫會議；搜尋結果說是 ICML 2026，未確認）。文字領域，把 DLM 記憶較少歸因於取樣解析度，AR 解碼是極端。差別：文字、歸因於取樣而非訓練，也沒說有沒有控制訓練條件。這篇是潛在的相反立場：如果影像上差異主要來自取樣而非訓練，我的主張就是假的。

**改進空間 / 公開失敗案例的證據**（全部取自 Kowalczuk 等人，已開啟）
- MIA TPR@1%FPR：VAR-d30 為 94.57%，DM 最高只有 6.38%（SiT-XL/2），而兩者 FID 相近（1.48 對 1.58）。
- 從 VAR-d30 萃取出 698 張訓練圖。
- RAR-XXL 49.80%，MAR-H 只有 3.40%。
- 原因至今沒有控制實驗。

**停損實驗（一週，2–4 張 GPU）**
- 取 ImageNet 子集，n ∈ {2k, 8k}。用同一個離散 tokenizer（例如 LlamaGen VQ）和同一個約 100M 的 Transformer，訓練四種變體：
  1. 因果 AR + CE；
  2. 因果 AR + CE，但 context 隨機破壞約 30%；
  3. MaskGIT 式 masked + CE；
  4. 連續 latent 的 DiT-B 作對照。
- 每約 10k 步量一次記憶比例（最近鄰比值 d1/d2 < 1/3）和 train/test NLL gap。
- 放棄條件：
  - (1) 和 (4) 在相近品質下 τ_mem 沒有明顯差異：代表 IAR 的洩漏來自規模或 likelihood 取得，不是機制。
  - (2)(3) 和 (1) 一樣快記憶：代表「context 破壞」的假說錯誤。此時題目退化成「loss 型態才是原因」，新意大減。
- 注意 VAR-d16 的 TPR 只有 3.05%。小模型可能看不出差異，所以要靠縮小 n 來放大記憶效應。

**需求**
- 模型：自訓約 100–300M 的 AR、masked、DiT。可用公開 VAR/RAR/MAR/SiT checkpoint 驗證外部效度。
- 資料：ImageNet 子集。
- 指標：τ_mem 對 n 的斜率、記憶比例、FID 或 held-out FD、加上不依賴 likelihood 的 MIA。
- 估計：完整的因子設計（4 種目標 × 3 個 n × 2 種大小 × 2 個 seed）約 50 次訓練，每次 10–20 GPU 小時，共約 700–1000 GPU 小時。在 4 張 H200、10 週內做得完。
- 審稿人預期的 baseline：Kowalczuk 的攻擊、Bonnaire 的 τ_mem 量測方法、MAR、標準去重複和 early stopping。

**ICML 適配度**：中高。Kowalczuk 在 ICML 2025，Bonnaire 據搜尋在 NeurIPS 2025，Luo 據搜尋在 ICML 2026（皆未確認）。「訓練目標如何決定記憶」屬於 ML 機制題目。

**主要風險**
- Boenisch/Dziedzic 組持續在這條線上（已有 Kasliwal 的後續），可能先做控制實驗。
- 審稿人可能認為「teacher forcing 會記憶」在 LLM 領域是常識，因此要強調 τ_mem ∝ n 這個定量律的有無。
- 小規模可能複製不出大規模的差距。
- 跨 tokenizer 和連續/離散的公平比較很難設計。

**信心：中**。失敗案例有公開數字，原因仍未解，也有可證偽的對立假說，算力也夠；但效果在小規模可能消失。

---

### 沒有第三個候選的原因

我評估過 masked 影像生成的解碼順序。Halton 排程器（ICLR 2025，https://arxiv.org/abs/2503.17076 ，已開啟）在 32 步時 FID 3.74，Confidence 排程器為 5.93，存在差距。但文字端的依賴感知平行解碼已經很多：DAPD 2603.12996、DEMASK 2604.02560、ParallelBench（ICLR 2026）、2602.00286（皆據搜尋摘要或已開啟摘要）。把這些方法搬到影像只會是遷移型的增量，不符合教訓 1。

---

### 不建議做的方向（各一句理由）

- **Feature caching／特徵重用**：TeaCache、TaylorSeer 等方法數量太多，偏應用調參，多半投 CVPR/ICCV。我沒有逐篇開啟，這是依領域概況判斷。
- **CFG 機制與修補**：2026 年至少有 CACFG（2610.05845）、terminal repair（2607.07665）、manifold projection（2601.21892）等，對成因沒有共識但已非常擁擠（據搜尋摘要）。
- **One-step／MeanFlow 系列**：iMF 1-step FID 1.72、DMF 2.16（據搜尋摘要），與多步模型差距已小，而且訓練 XL 模型需要的算力超出條件。
- **REPA／表徵對齊**：iREPA（ICLR 2026）、sREPA（2605.16949）、PixelDense（2610.00483），以及 RAEv2 已到 gFID 1.06（2605.18324，已開啟），主要由大實驗室在推。
- **Tokenizer 的 diffusability**：已有系統性研究 2606.03578、2605.07915，以及 ECCV 2026 的 2609.28473（已開啟），新的機制主張很難站得住。
- **Pixel-space x-prediction（JiT 後續）**：2610.23854（已開啟）已發現「模型容量增加會讓偏好轉向 velocity prediction」，PixelGen 等後續也很多。
- **VAR 機制與誤差累積**：SRDD（2509.22636）、2510.02826、SAR（CVPR 2026）已把「VAR 等於 discrete diffusion」和 exposure bias 談完，多半投 CVPR。
- **DMD 機制（CFG augmentation 還是 distribution matching）**：Decoupled DMD（2511.22677，已開啟）、1.x-Distill、eDMD（2611.17194，2026-11-12）、critic 研究（2610.42135，Adobe/MIT，已開啟）都在做，撞題風險極高。
- **一步蒸餾的記憶問題**：2604.23552 已做（據搜尋摘要）。
- **DiT 的 attention sink／massive activations**：2605.13974、2605.09313（據搜尋已有 ICML26 程式碼）、2509.22650，已經擁擠。
- **只做 diffusion 的 τ_mem 規模律驗證**：Marion & Wu（2605.06077，已開啟）是一篇以 CIFAR-10 實驗為主的立場論文，認為該往「記憶發生前模型學到什麼」去研究；單純做 ImageNet 規模驗證只是確認性工作。若要做，應併入候選 2 的跨模型族比較。

**本報告引用的主要連結**：
- https://arxiv.org/abs/2412.20292
- https://arxiv.org/abs/2411.19339
- https://arxiv.org/abs/2509.09672
- https://arxiv.org/abs/2605.04830
- https://arxiv.org/abs/2508.06614
- https://arxiv.org/abs/2509.16447
- https://arxiv.org/abs/2502.02514
- https://arxiv.org/abs/2509.00488
- https://arxiv.org/abs/2505.17638
- https://arxiv.org/abs/2505.16959
- https://arxiv.org/abs/2603.02333
- https://arxiv.org/abs/2605.06077
- https://arxiv.org/abs/2503.17076
- https://arxiv.org/abs/2610.23854
- https://arxiv.org/abs/2605.18324
- https://arxiv.org/abs/2511.22677
- https://arxiv.org/abs/2610.42135
- https://arxiv.org/abs/2611.17194
- https://cfeng16.github.io/on_the_diffusibility/

---

## 報告 B：low-level vision 與動態感知

範圍 B 結論

只提 2 個候選，第二個信心偏低。這個範圍內能寫成一般性 ML 問題的題目，幾乎都集中在「diffusion／flow 逆問題求解器的後驗品質」。這一塊 2026 年非常擠：2026 年 6 月以後至少有 6 篇直接相關的 arXiv（見下文）。復原、點追蹤、event camera 的其他方向，多半是 CVPR 型的刷榜或應用。我找不到第三個同時符合 5 條篩選標準的題目，所以不湊數。

---

## 候選 1：免訓練後驗採樣器的「覆蓋率校準」，用模型自己生成的真值做無需 ground truth 的校準

**一句話主張**：以 PSNR 或 MSE 調出的 guidance 權重和雜訊參數，會讓 DPS 類採樣器系統性地過度自信。改用「從 diffusion 先驗自己抽 x、模擬 y=Ax+n，再看 posterior 樣本的 rank 統計」（simulation-based calibration，SBC）來選超參數，可以讓覆蓋率接近名目值，而 PSNR 只損失少量。

這個做法的關鍵點：SBC 對「學到的先驗 p_θ」是精確的，所以可以在 256×256 影像尺度上把演算法本身的誤差單獨拆出來。Zach 等人的精確參考只做到 1D、d=64。

**為什麼可能為假、誰相信相反的事**
- Xu et al., ICLR 2025, "Rethinking Diffusion Posterior Sampling"（arXiv 2501.18913，已開啟）主張 DPS 本質上比較像 MAP，而且樣本多樣性低。如果他們是對的，DPS 這一類採樣器可能**沒有任何超參數設定**能達到校準，本主張就會不成立。
- Zach et al., ICLR 2026（下述）發現：分數模型變準之後，C-DPS 反而在 11 個顯著比較中有 7 個變差。這表示超參數在補償分數誤差，「調超參數就能校準」未必成立。
- Moroy et al. 2026（arXiv 2610.03503）以 MSE 目標最佳化 guidance 權重，並宣稱有 SOTA。他們隱含的假設是 MSE 最佳的權重就夠好；本主張認為 MSE 最佳和校準彼此衝突。
- 還有一種可能：在 p_θ 樣本上調好的參數，到真實影像上不成立（先驗失配）。這也會讓主張為假。

**最接近的工作**（以下皆已開啟確認）
1. Zach, Haouchat, Unser, "A Statistical Benchmark for Diffusion Posterior Sampling Algorithms", ICLR 2026, https://arxiv.org/abs/2509.12821
   - 做了什麼：用 Lévy 過程先驗加 Gibbs 精確後驗，評估 C-DPS、DiffPIR、DPnP 的 MMSE gap 和覆蓋率。
   - 差別：只做 1D、d=64，只做評測，不提校準方法；超參數用 validation set 上的 grid search 選。
2. Moroy et al., "Getting Your Guidance Weights Right in diffusion and flow-matching posterior sampling", arXiv 2610.03503（2026-10），https://arxiv.org/abs/2610.03503
   - 做了什麼：把 guidance 權重化為 2D 最小平方問題離線求解。
   - 差別：需要乾淨影像配對，目標是 MSE；只報 PSNR、SSIM、LPIPS、FID，沒有覆蓋率或多樣性的定量評估。**這是最接近的撞題風險。**
3. Benita, Elad, Keshet, "Analyzing and Guiding Zero-Shot Posterior Sampling in Diffusion Models", arXiv 2602.07715（v2 2026-05），https://arxiv.org/abs/2602.07715
   - 做了什麼：在高斯先驗下推導理想採樣器的閉式解，並提出參數設計框架。
   - 差別：只限高斯先驗，摘要裡沒有提校準或覆蓋率。
4. Burns, Fridovich-Keil, "When, why, and how do diffusion posterior samplers fail? A finite-sample lens", arXiv 2605.30330，https://arxiv.org/abs/2605.30330
   - 做了什麼：用有限樣本經驗先驗當診斷工具，指出 likelihood 近似在中間時間步會高估或低估後驗的散布。
   - 差別：只做診斷，不提修正方法。
5. Séguy, Rodet, "FB-GDM", arXiv 2609.29216（2026-09），https://arxiv.org/abs/2609.29216
   - 做了什麼：用變分推論在每一步推斷兩個 precision 參數，不需要 ground truth。宣稱在 CelebA-HQ 上與用 ground truth 校準的 ΠGDM oracle 相差 0.1 dB 以內。
   - 差別：只限 ΠGDM 型、線性問題；摘要沒有覆蓋率評估。
6. 相鄰工作：
   - "Training Calibrated Conditional Generative Models with a Sample-Only Coverage Loss"，arXiv 2611.15829（2026-11）：訓練期的覆蓋率損失，不是免訓練採樣器。
   - Qiu et al., "Beyond Accuracy: Evaluating Posterior Fidelity of Diffusion Inverse Solvers"，arXiv 2602.04189：提出 score-KSD 這個不需 ground truth 的指標，只做評測。

**改進空間的證據**
- Zach et al. 的 MMSE gap（dB，0 為最佳）：
  - Student-t(1) 先驗、去卷積：最好的 DiffPIR 仍差 10.45 dB，DPnP 7.84 dB，C-DPS 18.30 dB。
  - Bernoulli-Laplace 先驗、去卷積：DiffPIR 1.09 dB，C-DPS 4.30 dB。
  - 這些數字我是從第三方摘要頁 papernotes.org 讀到的。原 PDF 無法解析，標「數字來源為二手摘要，需再核對原文」。
- 覆蓋率：作者部落格說三個演算法「generally overconfident」；摘要頁轉述為覆蓋率「almost always much lower than α」。**我沒有拿到具體覆蓋率數字（未確認）。**
- PosteriorBench（Yao et al., arXiv 2609.20794）在科學逆問題上報告「substantial distribution-matching gaps」，posterior std 對不上；該頁也沒有具體數字。

**第一個停損實驗（約 1 週、1 張 H200）**
- 設定：FFHQ-256 預訓練 ADM 先驗；DPS、DiffPIR、DAPS 三種採樣器；4× SR 與 box inpainting 兩個任務。
- 從先驗抽 100 個 x，每個 x 產生 16 個後驗樣本。在像素、低頻 DCT 係數、PCA 方向上計算 SBC rank 直方圖和 90% 覆蓋率。
- 每個採樣器掃 guidance 權重和雜訊縮放（約 6 個值）。
- 放棄條件（任一成立即放棄）：
  - (a) 預設超參數下覆蓋率已經在名目值 ±5% 以內，代表沒有缺口；
  - (b) 整個掃描範圍內沒有任何設定能讓覆蓋率進到名目值 ±10%，而且 PSNR 也沒有可接受的點。此時只剩「Xu et al. 是對的」這種負面結果；可以考慮，但不是方法論文；
  - (c) 在 Zach 的 Lévy 基準上做轉移檢查失敗：用 SBC 選出的參數，對 Gibbs 精確後驗的覆蓋率沒有比 MSE 選出的參數更好。

**需要的資源**
- 模型：ADM FFHQ-256 與 ImageNet-256（DPS 原 repo 的權重），可再加一個 flow-matching 先驗（PnP-Flow 或 Flower 的 CelebA、AFHQ 權重）。Lévy 實驗沿用 github.com/zacmar/dps-benchmark。
- 資料：FFHQ、ImageNet val、Lévy 1D。
- 指標：SBC rank 均勻性（KS 或 χ²）、HPD 或分位數覆蓋率、MMSE gap、PSNR、LPIPS、FID，可加 score-KSD。
- GPU 時數：以每組設定約 0.5–2 GPU-h（100 NFE 的採樣器）估算，總共約 400–800 H200 小時，4 張卡 10 週可以做完。1000 NFE 的 DPS 很貴，要限縮樣本數。
- 審稿人預期的 baseline：
  - DPS、ΠGDM、DiffPIR、DAPS、DPnP，加一個 SMC 類（FPS-SMC 或 Tempered Guided Diffusion，arXiv 2605.03712）；
  - 超參數選法對照：Moroy et al. 的 MSE 權重、FB-GDM，以及「事後 conformal 修正樣本」，例如 Belhasin et al. TPAMI 2023（arXiv 2305.10124，只看過摘要）。

**ICML 適配度**
- 中高。同類分析／理論型論文近期被 ML 會議收錄的例子：
  - Zach et al. ICLR 2026；
  - Xu et al. ICLR 2025；
  - Jia et al., "Weak Diffusion Priors Can Still Achieve Strong Inverse-Problem Performance", ICML 2026 spotlight（arXiv 2601.22443，已確認）。
- 這類題目通常放在 cs.LG／stat.ML，不是 CVPR 型。

**主要風險**
- 撞題：2026 下半年同主題的論文非常密集（2605.30330、2606.17048、2609.20794、2609.29216、2610.03503、2611.15829）。Moroy et al. 或 Zach 團隊很可能下一步就做「為覆蓋率選權重」。
- 效果可能很小：如果過度自信是結構性的，調參救不回來，就只剩負面結果。
- 審稿人可能質疑「對 p_θ 校準 ≠ 對真實資料校準」，所以 Lévy 精確實驗必須做。
- 10 週可行，但 1000 NFE 的 DPS 會吃掉很多算力。

**信心：中。** 有實際證據顯示缺口存在，而「SBC 用 p_θ 自己的樣本，在影像尺度上做精確診斷並據此調參」這個組合，我搜尋後沒有找到直接撞題。但這一區太熱，撞題機率高，而且「校準與 MSE 衝突」可能只是一個小效應。

---

## 候選 2：弱先驗或失配先驗「點估計看起來沒事，但後驗校準崩潰」

**一句話主張**：diffusion 先驗失配時（換資料集訓練、少步數生成器），逆問題求解器的 PSNR 幾乎不受影響，但後驗覆蓋率與 MMSE 之外的分布誤差會隨失配程度明顯惡化。所以「弱先驗也夠用」只對點估計成立。

**為什麼可能為假、誰相信相反的事**
- Jia et al., ICML 2026 spotlight（arXiv 2601.22443）主張，在量測資訊充分時，弱先驗表現接近強先驗。他們的理論是 Bayesian consistency，也就是後驗會集中到真實訊號附近。
- 如果後驗本來就集中，覆蓋率也會跟著沒事，本主張就不成立。這正是可被否證的地方。

**最接近的工作**（皆已開啟）
- Jia, Yuan, Liu, Shen, Wang, ICML 2026, https://arxiv.org/abs/2601.22443：做的是點估計表現。
- Qiu, Yang, Liu, Wang, Shen, "Beyond Accuracy", arXiv 2602.04189：後驗保真度評測，提出 score-KSD。注意作者與上一篇高度重疊。
- Burns & Fridovich-Keil, arXiv 2605.30330：多峰先驗加上散布錯估會造成幻覺。
- Jin et al., "Hallucination-Aware Diffusion Sampling ... via Robust Prior Updates", arXiv 2606.02331：從先驗端的更新修正幻覺，只看 PSNR、LPIPS 和人評。
- Zach et al., ICLR 2026：提供精確參考後驗的實驗平台。

**改進空間的證據**：薄弱。Jia et al. 的摘要只說他們「identify regimes where they fail」，沒有寫出是哪些情況（未確認具體數字）。目前我沒有任何公開數字，能顯示「失配先驗的覆蓋率」比匹配先驗差多少。

**停損實驗（3–4 天，幾乎不需 GPU）**
- 在 Zach 的 Lévy 基準上，用 Laplace 訓練的 denoiser 求解 Student-t 或 Bernoulli-Laplace 的資料。也就是刻意製造先驗失配，然後比較 MMSE gap 與覆蓋率在「匹配」和「失配」兩種情況下的變化。
- 放棄條件：覆蓋率惡化幅度不比 MMSE gap 的惡化大。也就是兩者同步，沒有「點估計沒事但校準崩潰」的分離現象。

**需要的資源**
- Lévy 1D 實驗加上 FFHQ／ImageNet 交叉先驗：例如用 ImageNet 先驗解 FFHQ，可沿用 Jia et al. 的設定。
- GPU：約 100–300 H200 小時。
- baseline：DPS、DAPS、DiffPIR、DPnP，以及 Jia et al. 的弱先驗設定。

**ICML 適配度**：高。直接回應一篇 ICML 2026 spotlight。

**主要風險**
- **被原作者自己做掉的機率很高**：Shen 與 Wang 兩位作者同時寫了「弱先驗」和「後驗保真度評測」兩篇，把兩者合起來是他們最自然的下一步。
- 在影像尺度上，失配情況下沒有精確參考後驗，只能靠 score-KSD 這類代理指標或 1D 實驗，說服力有限。
- 貢獻可能被看成「評測補充」而不是方法。

**信心：低。** 主張可否證、而且對 ICML 有吸引力，但改進空間沒有公開數字支持，撞題風險又集中在一個特定團隊。

---

## 看過但不建議做的方向
- **改良 DPS 類採樣器以提高 PSNR／LPIPS**：極度擁擠。2026 年就有 EPS（Mammadov, Chung, Teh 等，arXiv 2606.17048，線性高斯下的精確後驗分數）、FB-GDM、guidance 權重最佳化（2610.03503）、DAPS++（2511.17038）、Tempered Guided Diffusion（2605.03712）、RPU（2606.02331）。邊際貢獻很難說服審稿人。
- **DPS 失敗模式的理論診斷**：Burns & Fridovich-Keil（2605.30330）已經用有限樣本觀點做掉；Benita/Elad（2602.07715）已做高斯閉式分析。
- **評測後驗保真度的 benchmark**：已有 Zach ICLR 2026、Beyond Accuracy（2602.04189）、PosteriorBench（2609.20794），再做一個 benchmark 沒有新意。
- **感知與失真取捨、PMRF 延伸**：Flow Map Denoisers（Zilberstein, Mardani, Segarra，arXiv 2606.19802）已證明在高斯目標下能精確走出最佳前緣，並用單一模型涵蓋整條取捨。Latent-PMRF 已是 ICLR 2026 撤稿投稿，DirectFlow 在 CVPRW 2026。剩下的空間偏應用。
- **生成式超解析的幻覺偵測與評分**：擠，而且偏 CV 或依賴多模態大模型。已有 Hallucination Score（2507.14367，看搜尋摘要）、VHA（2610.25770，2026-10，看搜尋摘要）、CHEM（2512.09806，看搜尋摘要）、conformal SR（2502.09664，看搜尋摘要）。理論面已有 Cohen et al.（2405.16475，已開啟）證明的「完美感知品質至少需要 2 倍固有不確定性」。
- **All-in-one 復原**：以挑戰賽和 benchmark 驅動（LoViF 2026、LL-Bench、RealIR-Bench，都只看了搜尋摘要），屬於 CVPR 型刷榜，結論句難以推廣成一般機制。
- **點追蹤與 optical flow**：主要投 CVPR／ICCV。測試時最佳化路線已有 DINO-Tracker（2024）和 Mask-to-Point（2603.17813）。要做出強結果需要大量合成資料訓練，ICML 適配度低。2026 年這一塊我沒有深入查撞題，標「未深入查證」。
- **Event camera 表示學習或脈衝模型**：Neural Events（Scaramuzza 團隊，2606.19835）、event2vec、SpikeCLR 都偏 CV、機器人或類神經形態社群。event2vec 內文出現 ICML 2026 模板，但是否被收未確認。整體 ICML 適配度低，算力門檻也不低（大規模事件資料）。

## 引用可信度註記
- 以下是我實際開啟摘要頁或內文確認過的：2605.30330、2609.20794、2606.17048、2611.15829、2405.16475、2606.19802、2602.04189、2609.29216、2501.18913、2602.07715、2605.03712、2610.03503（含 HTML 內文）、2606.02331、2601.22443，以及 Zach 的部落格。
- Zach 論文的 MMSE gap 數字來自第三方摘要頁 papernotes.org，原 PDF 無法解析，需要再核對。
- 其他只出現在搜尋摘要裡的論文，都標為「看搜尋摘要」或「未確認」。

---

## 報告 C：3D 與幾何

範圍 C（3D 與幾何）文獻探查結果

整體判斷：幾何基礎模型（VGGT／π3 一類）在 2025 年底到 2026 年 11 月之間變得非常擁擠。加速、串流記憶、量化、探針分析、迴圈化、不確定性排序這幾條都已經有 3 到 8 篇工作。符合五條篩選標準的只找到 2 個候選，都是中低信心，沒有湊第三個。

除了特別標「未開啟」或「未確認」的，下面引用的論文我都實際開過 arXiv 頁面或全文。

---

## 候選 A：多視角幾何 Transformer 的「視角數外推失敗」是注意力熵稀釋造成，免訓練的熵不變溫度縮放可以修正

**一句話主張**：VGGT 類模型在輸入遠多於訓練幀數（2–24 幀）時性能下降，主因是全域注意力的 softmax 熵隨總 token 數上升（稀釋），而不是秩崩潰或冗餘本身。只要依總 token 數對 logit 做 log 縮放（InfoScale／SSMax 這類做法），不訓練就能收回大部分退化，而且在 VGGT、π3 等多個模型上都成立。

**為什麼可能為假、誰相信相反的事**：
- 2512.21691（Li et al., arXiv 2025-12）把崩潰建模成跨層的退化擴散，收斂速率 O(1/L)。依這個說法，問題在深度動力學，不在熵。
- FastVGGT 把現象讀成 token collapse（各 token 的注意力圖高度相似），修法是合併冗餘 token。它還說這種坍縮可能是刻意的全域語意蒸餾，不是缺陷。
- 2607.01885 雖然量到熵隨幀數上升，但歸因於「冗餘視角」：視角多樣性才是關鍵，幀數本身不是。照這個說法，溫度縮放修不好冗餘問題。
- 這三種解釋彼此競爭，溫度縮放的實驗可以把它們分開，所以主張確實可能為假。

**最接近的工作**：
1. FastVGGT: Training-Free Acceleration of Visual Geometry Transformer — arXiv 2509.02560。第三方筆記站列為 ICLR 2026，我沒有到 OpenReview 確認。https://arxiv.org/abs/2509.02560
   - 做了什麼：免訓練 token merging，1000 幀時加速 4 倍並減少誤差累積。
   - 差別：它處理的是冗餘與速度，不改注意力的尖銳度。
2. Diversity-aware View Partitioning for Scalable VGGT — Park, Choi, Seo, Cho, Son。arXiv 2607.01885（2026-07）；ECCV 2026 poster 頁出現在搜尋結果，我沒有開。https://arxiv.org/abs/2607.01885
   - 做了什麼：量到熵隨幀數上升，修法是依多樣性把視角切塊。
   - 差別：全文沒有提到注意力溫度或 log-n 縮放（已確認）。
3. Analyzing the Mechanism of Attention Collapse in VGGT from a Dynamics Perspective — Li, Luo, Shi, Gu，arXiv 2512.21691。https://arxiv.org/abs/2512.21691
   - 做了什麼：純理論解釋，沒有提出新方法。
4. Information Entropy Invariance（InfoScale）— Li et al., arXiv 2501.08570。https://arxiv.org/abs/2501.08570
   - Scalable-Softmax — Nakanishi, arXiv 2501.19399。https://arxiv.org/abs/2501.19399
   - 做了什麼：在 LLM 上做長度外推的溫度縮放。
   - 差別：都沒有用在多視角幾何模型。
5. InfiniteVGGT — Yuan et al., arXiv 2601.02281。https://arxiv.org/abs/2601.02281
   - 它的 τ 只用在分配 KV 預算，不是依長度縮放注意力（已確認）。

**撞題查核**：我用多組關鍵字搜「VGGT／geometry transformer + temperature／entropy-invariant／log-n」，沒找到把溫度縮放用在幾何 Transformer 的工作。另外看了 ReDT（arXiv 2610.25517，2026-10，作者未記錄）：它處理的是高解析度（basis tokens），摘要沒有提到視角數外推或溫度縮放。

**改進空間的證據**：
- FastVGGT 表 2／表 5，ScanNet-50，VGGT\*：
  - 100 幀時 ATE 0.140、CD 0.423。
  - 1000 幀時 ATE 0.196（+40%）、CD 0.471（+11%）。
  - FastVGGT 在 1000 幀只回到 ATE 0.164、CD 0.425。
  - 這裡的上限是「多給幀不該變差」，也就是短序列的水準。
- 7Scenes（2607.01885）上，VGGT\* 的 AUC@30 從 300 幀的 0.8269 掉到 1000 幀的 0.8183，退化很小。這是效果可能很小的直接證據。

**第一個停損實驗（約 1 週、1 張 H200）**：
- 設定：用 FastVGGT repo 的 VGGT\*，取 10 個 ScanNet-50 場景，N ∈ {100, 1000}。只在全域注意力上把 q·k 乘以 s = log(N·T)/log(N₀·T)，其中 T 是每幀 token 數、N₀=24；再加 s 的網格掃描，以及「只縮放中段層」的變體。
- 同時量兩件事：
  - 各層的正規化熵。
  - 在固定 N 下，冗餘取樣與稀疏取樣的對照，用來區分「熵」和「冗餘」兩種解釋。
- 放棄條件：
  - 最佳的 s 也收不回 100→1000 幀 ATE 差距（0.056）的三分之一；
  - 或者熵已經回到 N₀ 的水準，誤差卻沒有改善；
  - 或者 π3 上完全無效。
- 估計：1000 幀一次約 12 分鐘（FastVGGT 回報 724 秒），停損實驗大約 10–15 GPU 小時。

**模型、資料、指標與算力**：
- 模型：VGGT、π3、VGGT-Ω／DA3（若權重可取得，未確認）。
- 資料：ScanNet-50、7Scenes、NRGBD、TUM（沿用 FastVGGT／2607.01885 的協定）。
- 指標：ATE／ARE／RPE、Chamfer／Acc／Comp、各層熵。
- 算力：完整實驗約 150–300 GPU 小時，1–4 張 H200 做得完。
- 審稿人預期的 baseline：FastVGGT、2607.01885 的分區、block-sparse／AVGGT（CVPR 2026，只在搜尋結果看到，原文未開）、VGGT-Long 切塊、InfoScale／SSMax 的直接移植。

**ICML 適配度：中**。
- 有利：機制解釋加上與 LLM 長度外推接軌，比較有通用機器學習的味道。ICLR 2026 收過 QuantVGGT（已在 proceedings 確認）和 FastVGGT（未確認）；長度泛化機制的 ASEntmax（arXiv 2506.16640）也是 ICLR 2026（已確認）。
- 不利：多數 VGGT 加速工作去了 CVPR 2026。

**主要風險**：
- 效果可能很小（7Scenes 幾乎不退化）。
- 審稿人可能說「只是把 LLM 技巧搬過來」。
- 溫度縮放不省記憶體，所以必須當成與分區、合併互補的機制來賣。
- 2607.01885 的作者已經量到熵，順手做溫度縮放的機率不低。

**信心：中**。現象和三種互相競爭的解釋都有引用，實驗便宜、一週可以判生死；但改進空間只有中等。

---

## 候選 B：免標籤估計每個場景的不確定性尺度（用自我一致性做 jackknife）

**一句話主張**：前饋式 3D 重建的信心值能排序誤差，但絕對尺度偏小，而且每個場景偏差不同。用模型自身的視角子集或重排預測，經 Sim(3) 對齊後算殘差（jackknife／自我一致性），不需要任何標籤就能預測每個場景的誤差尺度。這樣可以把 90% 區間的「每場景覆蓋率落在目標範圍內」的比例，從資料集層級校正的約 33% 推向逐場景 oracle 的 87%。

**為什麼可能為假、誰相信相反的事**：
- Calibration Audit（Deng, Cheng, Zeller, Cremers；arXiv 2608.29705 v3，2026-09）明確說無法在沒有標籤的情況下預測場景的誤差尺度，並把原因歸於模型不編碼誤差的尺度與分布形狀。
- ReVar3R（arXiv 2610.07883）說：「magnitude calibration requires labels and transfers poorly across domains」。
- 同一篇 Audit 還發現，Marigold 用 10 次取樣的分歧當不確定性時，名目 90% 區間只覆蓋 17–41% 的像素。這表示「分歧」通常會低估誤差。

**最接近的工作**：
1. A Calibration Audit of Confidence in Feed-Forward 3D Reconstruction Models — arXiv 2608.29705。https://arxiv.org/abs/2608.29705
   - 做了什麼：7 個骨幹 × 13 個資料集；用冪律做資料集層級校正。
   - 差別：沒有測多視角分歧、TTA 或重排。
2. ReVar3R: Gauge-Aware Perturbation Uncertainty — Mahdi, Salim, Rabbani, Zereen。arXiv 2610.07883（2026-10）。https://arxiv.org/abs/2610.07883
   - 做了什麼：重排加光度擾動，經 Sim(3) 對齊後取變異數；18 個條件中 15 個 AUSE 勝過內建信心。
   - 差別：只處理排序，尺度要靠標籤。這是最直接的撞題威脅。
3. GeoCond — Ahmedt-Aristizabal et al.，WACV 2027，arXiv 2609.18465。https://arxiv.org/abs/2609.18465
   - 做了什麼：用 frame-permutation orbit variance 當訓練訊號，訓練姿態層級的可靠度 adapter。
   - 差別：需要訓練 head，對象是姿態，不是逐像素尺度。
4. Trust3R — Zhu et al.，ICML 2026，arXiv 2605.19539。https://arxiv.org/abs/2605.19539
   - 做了什麼：監督式 evidential head。
   - 差別：需要訓練。它也主張 conformal 太保守。
5. Free Geometry — Dai, Yang，arXiv 2604.14048。https://arxiv.org/abs/2604.14048
   - 做了什麼：用「視角越多越好」的全集對子集一致性做 LoRA 測試時適應。
   - 差別：目的是精修，不是做不確定性。

**改進空間的證據**（2608.29705 表 2 與正文）：
- 原生信心：覆蓋率 0.570，落在目標範圍內的場景 0%。
- 最佳資料集層級方法（冪律／Mondrian）：0.333。
- 逐場景 oracle（一個用 GT 擬合的純量）：0.87。
- 四種 localized／multivalid conformal 變體都沒有超過 0.333。
- 這個差距有數字，原作者也明說這個問題沒解決。

**第一個停損實驗（約 1 週）**：
- 設定：VGGT 與 π3，在 ScanNet、7Scenes、DTU 上逐場景計算 leave-k-views-out 加 Sim(3) 對齊的殘差尺度。沿用 Audit 的協定（論文說會公開協定與常數，連結未確認），算出逐場景 oracle 純量。
- 放棄條件：
  - 免標籤統計量和 oracle 純量的 Spearman 相關低於約 0.5；
  - 或者用它做逐場景縮放後，落在目標範圍內的比例沒有從 33% 提升到至少 50%。
- 算力：只需前向推論，約 20–40 GPU 小時。

**模型、資料、指標與算力**：
- 模型：VGGT、π3、MASt3R、CUT3R。
- 資料：Audit 用的 13 個資料集的子集，加上 ETH3D、DTU。
- 指標：覆蓋率、落在目標範圍內的比例、誤差／不確定性比（level 與 slope）、AUSE。
- 算力：完整實驗約 100–200 GPU 小時。
- 審稿人預期的 baseline：Audit 的冪律與 Mondrian conformal、ReVar3R、Trust3R、GeoCond、深度集成／TTA。

**ICML 適配度：中到高**。Trust3R 就在 ICML 2026，而且這題可以寫成 jackknife／conformal 理論加上 3D 應用。

**主要風險**：
- 撞題風險很高：2026 年 5 月以來已有 5 篇同類工作。ReVar3R 的作者或 Audit 團隊下一步很可能就是做這題。
- ReVar3R 自己承認的限制：穩定的系統性偏差沒有變異數，而 jackknife 偵測不到偏差，這可能就是上限。
- Marigold 的分歧低估很嚴重，是負面前例。

**信心：低到中**。問題被明確留下，差距也大；但領域擁擠，機制上限也有負面先例。

---

## 為什麼只報 2 個

其他看過的方向都至少違反一條標準：已被做掉、太擠、沒有改進空間的證據，或是偏應用。細節見下節。

---

## 不建議做的方向

- **串流重建的記憶／狀態更新（CUT3R／TTT3R 後繼）**：太擠。TTSA3R（2601.22615）、FILT3R（2603.18493）、Mem3R（2604.07279）、SSR（2603.14765）、LongStream（CVPR 2026, 2602.13172）、InfiniteVGGT、ZipMap、Anchor3R 都已在做。其中 FILT3R、Mem3R、SSR、ZipMap、Anchor3R 只看了搜尋摘要，沒有開全文。
- **串流 KV 快取壓縮與 first-frame attention sink**：LongStream 已點名 sink 是漂移原因並修掉，Anchor3R 也改掉了固定參考幀；π3 則從架構上移除參考幀。
- **免訓練加速（token 合併、稀疏注意力、層剪枝）**：FastVGGT、AVGGT、Block-sparse（CVPR 2026，只看到搜尋結果）、LiteVGGT、2609.36842 層剪枝（3,018 種配置，參數最多減 44%）、VGG-T³。很擠，邊際收益小。
- **量化**：QuantVGGT（ICLR 2026）4-bit 已保留超過 98% 精度，改進空間小。
- **純探針式機制分析**：對應關係與對極幾何（2512.11508）、共視性（2607.09503）、失敗自知（2610.35806）、離群視角拒絕（2512.04012，只看摘要）都已做過。
- **用探針訊號修正失敗**：2610.35806 自己報告，把失敗配對轉給 matcher、或在 rotation averaging 中刪邊都沒有收益；而且多視角下失敗本來就少。
- **層迴圈／測試時算力**：Déjà View（arXiv 2605.30215，NVIDIA）已做出權重共享的迴圈模型。NeurIPS 2026 oral 的標註只出現在搜尋結果，arXiv 頁上沒有寫。
- **高解析度外推**：HD-VGGT（2603.27222，只看到搜尋結果）和 ReDT（2610.25517）已處理，偏 CVPR。候選 A 可以順帶把每幀 token 數納入同一條熵定律，但不建議單獨做。
- **不確定性排序**：ReVar3R、GeoCond、Trust3R 已涵蓋，只剩每場景尺度（即候選 B）。
- **LVSM 類免 3D 的新視角合成模型的隱式幾何探針**：沒有改進空間的證據；2607.19765 已做 saliency 分析；重新訓練所需算力不夠。
- **3DGS 二階最佳化、壓縮、致密化**：只粗略看過。二階最佳化已有 3DGS-LM、3DGS²、matrix-free LM（2504.12905）等，主要投 SIGGRAPH／CVPR，ICML 適配度低。
- **前饋 3DGS、4D 動態、3D 生成**：只粗略看過，判斷是擁擠或算力不足，沒有深入查證。

---

## 已開啟確認的主要連結

2509.02560、2607.01885、2512.21691、2512.11508、2610.35806（含 HTML 全文）、2609.36842、2607.09503、2601.02281、2501.19399、2501.08570、2506.16640、2608.29705（含 HTML 全文）、2610.07883、2609.18465、2605.19539、2604.14048、2602.13172、2605.30215、2503.11651、2610.25517，以及 QuantVGGT 的 ICLR 2026 proceedings 頁。

---

## 報告 D：視覺語言模型與文件理解

範圍 D 調查報告（2026-11-14）

結論：範圍內只找到 2 個勉強過篩的候選，兩個都屬「OCR／文字密集影像裡，資訊卡在 VLM 的哪一段」。第 3 個找不到夠好的，所以不報。token 剪枝、sink/register、光學壓縮、幻覺解碼這四條主線在 2026 年都已經很擠，理由放在最後。

兩個候選共用一套工具：渲染文字影像、在 Qwen3-VL 上取 hidden state、訓練 probe。實務上可以合併成一篇論文，分兩個 setting 來做。

標示原則：下文寫「已開啟」的論文，我實際打開過 arXiv abs 或 HTML 頁面，但不少只讀了摘要或被截斷的全文。只在搜尋結果裡看過的論文標「未確認」。

---

## 候選 1：VLM 在 OCR 中「改寫」字詞，是讀出端的失敗，不是編碼端的失敗；可以在推論時用針對機制的介入修正

一句話主張（假設實驗如願）：開源 VLM 在擾動字詞（打亂、拼錯、無意義字串）上，會輸出語言先驗下的原字。被改寫的實例中，中層 hidden state 仍能線性區分「影像中的真實字串」與「原字」，所以失敗發生在晚層的讀出端（Lee et al. 指出的 final-layer FFN）。一個由此導出的免訓練介入（例如層對比，或抑制晚層 FFN 的先驗方向）能把改寫率壓低一半以上，且不傷害自然文字和降質文字的 OCR。

### 為何可能為假、誰相信相反的事
- Karamolegkou et al. 2026（arXiv 2605.27750，已開啟）的結論是：「visual dependence cannot be reliably strengthened at decode time」。他們的實驗裡 VCD 完全無效，M3ID 只對 Qwen3-VL 有效、對 OlmOCR 無效，而且是相對它自己較差的 no-contrast pass 計算。
- Liang et al. 2026（arXiv 2601.03714，已開啟）認為需要「architectural redesign」，並把問題歸因於解碼器的語言先驗本質，不是可以修的讀出問題。
- 主張也可能在資料上就不成立：如果被改寫實例的中層表徵根本分不出擾動字與原字，「資訊還在」這個前提就倒了。

### 最接近的工作（都已開啟）
1. Lee, Ak, Mohta, Xu, Dimitriadis，"Do VLMs Read or Rewrite? On Transcription Faithfulness in Vision-Language Models"，arXiv 2607.21617（v2，2026-09），https://arxiv.org/abs/2607.21617
   - 做了什麼：15 個系統的改寫率。英文打亂擾動下，olmOCR-2-7B 58.63%、InternVL3.5-4B 46.37%、Qwen3-VL-4B 31.33%、Tesseract/docTR 0%。在 Qwen3-VL-4B 上用 probe 定位到 final-layer FFN 表徵貼近原字，attention 解釋不了改寫。
   - 與本候選的差別：只測了「請不要修正」的 prompt（降 30–50%），作者明說沒做 decoding intervention 或微調。
   - 撞題風險：這是最大的撞題來源，作者自己很可能接著做緩解。
2. Liang et al.，"Visual Merit or Linguistic Crutch? A Close Look at DeepSeek-OCR"，arXiv 2601.03714（2026-01，疑似投 ACL 2026，未確認），https://arxiv.org/abs/2601.03714
   - 做了什麼：自然文字對上零先驗隨機文字，DeepSeek-OCR Tiny 從 88.00 掉到 19.84，Qwen2.5-VL-7B 約掉 51 點，pipeline 的 PaddleOCR-v5 只掉約 4.9 點（94.44→89.53）。
   - 與本候選的差別：只有診斷，沒有方法，也沒有機制定位。
3. Karamolegkou, Angleraud, Sagot, Clérice，"Reading or Guessing? Visual Grounding Failures of VLMs for OCR in Ancient Greek Editions"，arXiv 2605.27750（2026-05），https://arxiv.org/abs/2605.27750
   - 做了什麼：在 OCR 上測 VCD、M3ID、字集遮罩、post-OCR 修正，結果多半無效或依模型而異。
   - 與本候選的差別：用的是通用的對比解碼，不是從機制導出的介入，且只限古希臘文。
4. Seo et al.，"DEGAP: Dynamic Entropy-Guided Attention Perturbation for Contrastive Decoding in LVLMs"，ICML 2026 poster，https://icml.cc/virtual/2026/poster/65456
   - 做了什麼：擾動 visual attention 來做對比解碼，處理物件幻覺。
   - 與本候選的差別：沒碰 OCR 和字元層級的忠實度。它屬於審稿人會要求比較的 baseline 家族。
5. He et al.，"Seeing is Believing? Mitigating OCR Hallucinations in MLLMs"（KIE-HVQA），NeurIPS 2025，arXiv 2506.20168
   - 做了什麼：處理降質文件，用 SFT 加 GRPO 教模型拒答。
   - 與本候選的差別：處理的是遮擋、看不清，不是「看得清卻改寫」，而且需要 RL 訓練。

### 改進空間的證據
- 上限是 Tesseract/docTR/Gemini-3-Flash 的改寫率 0%，開源 4–7B VLM 是 27–59%（2607.21617 Table 5）。最好的既有緩解（prompt）只降 30–50%。
- 零先驗文字上，pipeline OCR 89.53，端到端 VLM 只有約 20–60（2601.03714）。
- 已知的解碼方法有公開失敗案例：VCD 無顯著效果（2605.27750 Table 3）。

### 停損實驗（1 週，1 張 H200）
做法：
1. 用 Qwen3-VL-4B/8B，照 2607.21617 的協定渲染打亂和拼錯的字，約 5k 字，重現改寫率。
2. 在被改寫的實例上，對各層、該字位置的 hidden state 訓練線性 probe，區分「擾動字串」與「原字」。
3. 試一個最便宜的介入（中層對晚層的 logit 對比，或 ablate Lee et al. 指出的晚層 FFN 方向），看改寫率和自然文字 CER 的變化。

放棄條件（任一成立就停）：
- 重現不出 ≥20% 的改寫率。
- 被改寫實例上，所有層的 probe 都分不出擾動字與原字（例如 <70%）。這代表資訊在編碼端就沒了，該轉去候選 2。
- 最便宜的介入加上 VCD/M3ID，改寫率都降不到一半，或者自然文字 CER 明顯惡化。

### 資源
- 模型：Qwen3-VL-2B/4B/8B、InternVL3.5-4B、olmOCR-2-7B（OCR 專用、改寫率最高）。
- 資料：自行渲染的擾動文字（仿 2607.21617），2601.03714 的隨機文字設定，OmniDocBench／OCRBench 用來確認沒有退化，降質自然文字用來測「先驗有用」的那一面。
- 指標：改寫率、CER、自然文字和降質文字 CER 的權衡曲線。
- GPU：約 100–300 H200 小時。
- 審稿人預期的 baseline：「不要修正」prompt、VCD、M3ID、DEGAP 類 attention 擾動對比解碼、DoLa 類層對比，以及合成無意義字串的 LoRA 微調（必比，見風險）。

### ICML 適配度：中偏低
- 「先驗壓過視覺證據」的機制加解碼方法有 ICML 先例（DEGAP，ICML 2026）。
- 但 OCR 忠實度題目多半去 ACL/EMNLP/CVPR/ICDAR。要投 ICML，必須包裝成「先驗與證據在多模態解碼中的仲裁」的一般機制，並在 OCR 以外至少再驗證一個 setting（例如計數或物件屬性）。

### 主要風險
- 2607.21617 原作者在 2026-09 已經點名「decoding intervention 是 future work」，被搶的機率不低。
- 一個簡單的反駁：「用隨機字串做 LoRA 微調就解決了」。如果微調便宜又有效，免訓練方法的價值會被質疑。
- 效果可能依模型而異，就像 2605.27750 看到的那樣。

### 信心：中低
問題真實，數字也清楚，反方有引用。但 ICML 適配度與撞題風險都偏高。

---

## 候選 2：視覺 token「容量撞牆」其實是讀出瓶頸，不是編碼容量極限

一句話主張（假設實驗如願）：文字密度升高時，VLM 的 OCR 會出現「崩潰相」。在崩潰區，凍結的視覺 token 上訓練的淺層讀出器（線性 per-token 或小型 head）仍能恢復大部分字元。所以崩潰主要來自 LLM 解碼端的讀出，而不是 vision encoder 的資訊容量。只調解碼端（LoRA）就能把崩潰點往高密度推。

### 為何可能為假、誰相信相反的事
- Zhuang et al. 2026（arXiv 2602.02539，已開啟）明確把崩潰歸因於「irreversible collapse driven by an information capacity limit」（vision encoder 容量）。他們用 block-wise shuffling 排除語言先驗，但沒有直接 probe 視覺 token。我認為這是他們論證中可以檢驗的漏洞。
- 結果完全可能相反：probe 在同一密度一起崩潰，那就證實了容量說。

### 最接近的工作（都已開啟）
1. Zhuang et al.，"How Much Information Can a Vision Token Hold? A Scaling Law for Recognition Limits in VLMs"，arXiv 2602.02539（2026-01，未見收錄），https://arxiv.org/abs/2602.02539
   - 做了什麼：在 DeepSeek-OCR、InternVL3.5-8B、Qwen2.5-VL 上找到 stable／instability／collapse 三相。stable 在 5,000 字以下，collapse 在 15,000 字以上，兩者比約 2.2 倍且與解析度無關。
   - 與本候選的差別：只看端到端輸出，沒有 probe，也沒有試緩解（作者自列為 limitation）。
2. Liang et al.，arXiv 2601.03714（同候選 1）。
   - 做了什麼：壓縮越強，先驗依賴越重；長文在約 8,000–10,500 token 崩潰。
   - 與本候選的差別：歸因於解碼器先驗，同樣沒有 probe。
3. Shi，"VLMs Read Aloud: The Gap Between Recognizing and Using Visual Text"，arXiv 2611.01517（2026-11-03），https://arxiv.org/abs/2611.01517
   - 做了什麼：Qwen3-VL-8B 的「認得但用不上」差距約 +10.7 點，提出先轉寫再回答的 Aloud。
   - 與本候選的差別：它處理的是「認得之後用不用」，不是「認不認得」的容量問題。但它很新，主題相鄰，審稿人會要求區分。
4. Kawasaki, Tanaka, Nishida，"Responses Fall Short of Understanding"，arXiv 2604.04411（CVPR 2026 workshop），https://arxiv.org/abs/2604.04411
   - 做了什麼：文件理解所需的資訊在中間層比最後一層更線性可解，微調中間層可以縮小差距。
   - 與本候選的差別：沒有談密度或容量，也沒有字元級的讀出。
5. Lee et al.（"Optical Context Compression Is Just (Bad) Autoencoding"，arXiv 2512.03643；作者未核對，只讀過搜尋摘要，未開啟全文）。
   - 做了什麼：指出 DeepSeek-OCR 的視覺壓縮不勝過 mean-pooling 等直接壓縮法。
   - 與本候選的差別：比較的是壓縮法，不是拆解容量與讀出。

### 改進空間的證據
- 2602.02539 本身沒有任何緩解。崩潰區 expected edit distance 大於 0.7，stable 區小於 0.1，同一模型在兩區之間的落差就是可攻的空間。
- 2601.03714：DeepSeek-OCR Tiny 在隨機文字上 precision 只有 19.84%，Base 是 61.70%。
- 缺點：目前還沒有任何「上限」數字。上限要靠本研究自己的 probe 給出，所以改進空間在停損實驗之前並未證實。

### 停損實驗（1 週，1–2 張 H200）
做法：
1. 用 Qwen2.5-VL-7B（或 Qwen3-VL-8B）加 DeepSeek-OCR，固定 token 數，渲染從 stable 到 collapse 的密度階梯，文字用打亂後的版本以去除先驗。
2. 凍結 encoder 加 projector，訓練兩種讀出器，用字型和文本都不重疊的 held-out 集評估：
   - 線性或 2 層 MLP 加 CTC 的 per-token 讀出；
   - 2–4 層、約 5000 萬參數的小 transformer。
3. 比較讀出器的 CER 和 VLM 原生輸出的 CER，看各自從哪個密度開始崩潰。

放棄條件：讀出器在 VLM 崩潰的密度上，CER 也同步崩潰，例如兩者差距小於 10 個百分點。這代表容量說成立，本主張為假。
注意：小 transformer 讀出器容量太大時，會被質疑「等於換一個解碼器」。主要證據要以淺層讀出為準。

### 資源
- 模型：Qwen2.5-VL-7B／Qwen3-VL-8B、DeepSeek-OCR、InternVL3.5-8B。
- 資料：渲染文字（小說、維基、打亂版），OmniDocBench 的高密度頁。
- 指標：CER／edit distance 對密度的曲線、崩潰點位移量。
- GPU：probe 約 50–100 H200 小時；LoRA 推移崩潰點約 200–400 小時。
- 審稿人預期的 baseline：原生輸出、提高解析度（等 token 預算的對照）、mean-pooling 壓縮、只微調 projector、只 LoRA 解碼器。

### ICML 適配度：中
「表徵中有資訊但沒被用」的 probing 加容量論證，在 ICLR/ICML 有市場。容量與 scaling law 式的論述也偏 ML。但負面結果（容量說成立）很難單獨成篇。

### 主要風險
- 結果為假的機率不低：在高密度下，patch 本身可能就無法區分字元。
- 2602.02539 原作者可能自己補 probe。
- 十週內要同時做 probe 和 LoRA 推移，時程偏緊。

### 信心：中低
主張是真正的二選一，反方立場明確。但改進空間還沒有證據，第一週的結果決定一切。

---

## 為何沒有第 3 個候選
其餘子方向，不是最近 6 個月已有 3 篇以上同機制的工作，就是只剩 1–3 點的增益空間，不符合教訓 3 和 4。

---

## 看過但不建議做的方向

- **免訓練視覺 token 剪枝與合併（含文件、OCR 專用）**：極擠。2026 年的 arXiv 就有 ET-Prune 2608.01979、FastOCR 2605.17447、RTPrune 2605.00392、IDPruner 2602.13315、S2Prune 2609.01224、RoRA 2608.07088 等（這些未開啟，只看過搜尋摘要）。增益多半是幾個點，偏工程調參。
- **「簡單降解析度勝過多數壓縮法」的原因**：已經大多被解釋。
  - VTC-Bench（Liao et al.，arXiv 2510.07143，已開啟，搜尋結果顯示收錄於 ACL 2026，未確認）歸因於 benchmark 的 simplicity bias：Qwen2-VL-7B 在 75% 壓縮下，降採樣的 ADR 是 91.0，各方法是 82.7–84.9；但 ChartQA/OCRBench 上各方法反而贏。
  - 位置偏差（ICCV 2025 "Feather the Throttle"，未開啟）和分佈偏移（RCP 2604.04972、"Rescuing Your Pruning" 2610.33688，都未開啟）也已經有人提。剩下的機制空間很小。
- **VLM 的 visual attention sink 和 register**：擠，而且增益小。
  - 已有 Kang et al. ICLR 2025、Luo et al. arXiv 2510.08510（已開啟；ViT sink 有利全域任務、不利細節任務，多數增益小於 1 點）、Choi et al. arXiv 2604.03316（已開啟；LSG 在 LLaVA-1.5 上 MMStar +1.55、CVBench +3.08，學到的 gate 遠不及 oracle）、Đukić et al. arXiv 2611.15448（2026-11-11，已開啟）、Fan et al. CVPR 2026（未開啟）。
  - Jiang et al. 的 test-time registers（NeurIPS 2025，arXiv 2506.08010，只讀過搜尋摘要，未開啟全文）已經涵蓋免訓練去 artifact。
  - 唯一的縫隙是 Choi et al. 的 oracle 與 learned gate 之間的差距，但競爭者已有 5 組以上。
- **光學上下文壓縮（DeepSeek-OCR／Glyph 式「把長文渲染成圖」）的評測與批評**：已被 2512.03643（未開啟）、2601.03714、2611.01517 做掉。2611.01517 在 2026-11 才剛出，還附了方法。
- **物件幻覺的對比解碼和 attention 介入**：極擠，ICML 2026 就有 DEGAP，VCD 家族的變體非常多，難以做出差異。
- **高解析度 zoom-in、crop、「thinking with images」RL**：需要 RL 訓練算力，題目擁擠，偏 CVPR。
- **降質文件的 OCR 幻覺與拒答**：KIE-HVQA（NeurIPS 2025）已經做過，而且需要 GRPO。
- **OCR 錯誤偵測與拒答的 probe**：已有 Yao et al. arXiv 2511.19806（已開啟），但本身偏應用。
- **AnyRes 中全域縮圖與局部 tile 的依賴分析**：這次沒找到 2026 年直接做這件事的論文，但相關的 token 冗餘分析（GlobalCom2、HERO，未開啟）已多，難以導出強方法，增益可能很小。

主要來源：
https://arxiv.org/abs/2607.21617
https://arxiv.org/abs/2601.03714
https://arxiv.org/abs/2605.27750
https://arxiv.org/abs/2602.02539
https://arxiv.org/abs/2611.01517
https://arxiv.org/abs/2604.04411
https://arxiv.org/abs/2510.07143
https://arxiv.org/abs/2604.03316
https://arxiv.org/abs/2611.15448
https://arxiv.org/abs/2510.08510
https://arxiv.org/abs/2506.20168
https://arxiv.org/abs/2511.19806
https://icml.cc/virtual/2026/poster/65456

---

## 報告 E：ICML 實際收什麼

範圍 E 回報：帶視覺成分的論文，什麼樣的進得了 ICML

### 0. 資料來源與抽樣方式
- **官方頁面（已實際開啟）**：icml.cc 的 ICML 2025 oral 列表（https://icml.cc/virtual/2025/events/oral）、ICML 2026 oral 列表（https://icml.cc/virtual/2026/events/oral ，頁面寫共 169 場 oral）、2025 與 2026 的 Spotlight 頁面（2026 頁面寫 574 個 event），以及 ICML 2026 Call for Papers。
- **全量統計**：用 GitHub `papercopilot/paperlists` 的 JSON。這份資料由社群從 OpenReview／官方站抓取，不是官方發布。涵蓋 ICML 2025（main track 錄取 3,269 篇）、ICML 2026（錄取 6,341 篇）、ICLR 2026（含被拒與撤稿，共 18,906 篇有結果）、NeurIPS 2025（main 錄取 5,275 篇）、NeurIPS 2026（main 錄取 5,681 篇）。
  - 「視覺相關」的判定方式：**標題**含 image、video、visual、3D、VLM、MLLM、CLIP、ViT、segmentation、depth、diffusion transformer、tokenizer、world model 等關鍵字。
  - 類型判定也是用標題的正規表示式，可重複歸類，誤差約 ±幾個百分點，只能看相對量級。
  - ICML 2026 的 JSON 沒有標 oral（536 篇 Spotlight 應已含 oral）。oral 名單改用 icml.cc 頁面人工核對。
- **審稿意見**：OpenReview 的網頁與 API（api2.openreview.net）都回傳 403 驗證牆，**沒有讀到任何一篇審稿意見全文**。第 4 節只能用替代資料（ICLR 2026 的子分數），並明確標示。
- **論文內容的確認方式**：
  - Lottery Prior、Compositional Generalization、Are VLMs Seeing、Error Propagation、Kamb & Ganguli、Lukoianov et al. 這幾篇有另外開啟 icml.cc、PMLR 或 arXiv 頁面確認。
  - 其餘論文是讀 paperlists 裡的官方摘要。這些論文的 OpenReview 頁面本身打不開，下文只附 forum 連結，未直接開啟。

### 1. 視覺論文在各會議的比重，以及類型分布
**比重：**
- ICML 2025：482／3,269，14.7%。
- ICML 2026：1,245／6,341，19.6%。
- NeurIPS 2025：24.4%。
- NeurIPS 2026：22.6%。
- 在 ICML 2026 的 primary area 中，`applications->computer_vision` 有 515 篇，排名第二，僅次於 LLM 的 1,155 篇。
- 用標題關鍵字比對 icml.cc 的 2026 oral 名單：169 篇中約 28–30 篇與視覺相關，約 17–18%。

**ICML 2026 視覺論文的類型占比**（可重複歸類）：

| 類型 | 占比 |
|---|---|
| VLM／MLLM／VLA | 22% |
| 擴散／flow／AR 生成 | 22% |
| 影片／world model | 18% |
| 3D | 13% |
| 效率 | 12% |
| 安全／魯棒性 | 11% |
| 可解釋性／機制分析 | 10% |
| 表徵學習 | 7% |
| benchmark | 6% |
| 逆問題／復原 | 3.6% |
| tokenizer | 1.4% |

**重要差異：視覺論文拿得到錄取，但較少被選為重點論文。**
- ICML 2026 的 spotlight（含 oral）比例：視覺 6.3%，非視覺 9.0%。
- ICML 2025 的 oral 比例：視覺 2.3%，非視覺 3.5%。
- ICLR 2026 的 oral 比例：視覺 2.8%（37／1,323），非視覺 4.6%（187／4,028）。
- NeurIPS 兩年沒有這個差距。

**各類代表作**（O 表示 oral；ICML 2026 的 S 表示 spotlight 或 oral）：

- **生成模型理論與取樣**
  - An analytic theory of creativity in convolutional diffusion models（ICML25 **O**，https://proceedings.mlr.press/v267/kamb25a.html）：證明局部性加等變性就能用解析的 ELS machine 預測 CNN 擴散模型的輸出，中位數 r² 約 0.9。
  - Inductive Moment Matching（ICML25 **O**，https://icml.cc/virtual/2025/poster/43977）：單階段訓練的少步生成模型，ImageNet-256 用 8 步達到 FID 2.13。
  - High-Accuracy Sampling for Diffusion Models and Log-Concave Distributions（ICML26 **O**，https://openreview.net/forum?id=GW3umRqsZZ）：取樣步數達到 polylog(1/δ)，是純理論。
  - A Random Matrix Perspective on the Consistency of Diffusion Models（ICML26 **O**，forum id=iPjuUQbkfl）：用隨機矩陣理論解釋為何在不相交資料上訓練的模型會生成相似的圖。
  - Rex: Reversible Exponential (Stochastic) Runge-Kutta Solvers（ICML26 **O**，forum id=7pQIzVNctu）：可精確反演的 ODE／SDE 求解器。
- **生成模型的泛化與記憶化（分析型）**
  - Towards a Mechanistic Explanation of Diffusion Model Generalization（ICML25 S，https://icml.cc/virtual/2025/poster/45759）：免訓練的局部去噪器能重現網路行為。
  - Biased Generalization in Diffusion Models（ICML26 S，forum id=IqTvyp40O6）：test loss 仍在下降時，模型已開始偏向訓練樣本。
  - Locality in Image Diffusion Models Emerges from Data Statistics（NeurIPS 2025，https://arxiv.org/abs/2509.09672）：主張局部性來自資料統計，而非 CNN 架構，直接反駁上面 Kamb & Ganguli 的歸因。
- **Tokenizer**
  - Masked Autoencoders Are Effective Tokenizers for Diffusion Models（ICML25 S，https://icml.cc/virtual/2025/poster/44606）：latent 的 GMM mode 越少、越有判別性，生成越好，也就是說 VAE 的變分形式並非必要。
  - End-to-End AR Image Generation with 1D Semantic Tokenizer（ICML26 S，forum id=fSnxuRhWoC）：tokenizer 與生成器聯合訓練，ImageNet-256 無 guidance 的 FID 為 1.48。
  - VideoFlexTok（ICML26 S，forum id=KyVlaw4BxE）：可變長度、由粗到細的影片 token。
  - ViTok-v2（ICML26 poster，forum id=GkNfX22VXf）：5B 參數；發現小生成器適合激進的通道壓縮。
- **效率**
  - Error Propagation Mechanisms and Compensation Strategies for Quantized Diffusion Models（ICML26 **O**，https://icml.cc/virtual/2026/oral/71084）：推出量化誤差累積的封閉解，加上時間步補償，在 SDXL W4A4 上比 SVDQuant 高 1.2 dB PSNR。
  - Budget-Constrained Step-Level Diffusion Caching（ICML26 poster，forum id=kN0NMoVWXG）：先固定算力預算，再搜尋 cache 策略。
  - LearniBridge（ICML26 poster，forum id=8sD74Krbw7）：最佳校正落在跨 prompt 共享的低秩子空間，只需 3–5 個樣本。
  - 對照：Hybrid Feature Caching for DiT（ICLR26 **O**）；FlashVID 免訓練影片 token 合併（ICLR26 **O**）。
- **機制分析／可解釋性**
  - ConceptAttention（ICML25 **O**，https://icml.cc/virtual/2025/poster/45272）：免訓練，在 DiT attention 輸出空間做投影得到 saliency，零樣本分割達到 SOTA。
  - Are VLMs Seeing or Just Saying?（ICML26 **O**，https://icml.cc/virtual/2026/oral/71047）：把圖片換掉，VLM 的「再看一次」多半沒有真的看，準確率最多掉 60%，thinking 模型更脆弱。
  - Compositional Generalization Requires Linear, Orthogonal Representations in Vision Embedding Models（ICML26 **O**，https://icml.cc/virtual/2026/oral/71155）：證明組合泛化的必要幾何條件，並在 CLIP／SigLIP／DINO 上驗證。
  - A Distributional View for Visual Mechanistic Interpretability（ICML26 S，forum id=TZvHUmClPP）。
- **Benchmark**
  - EMMA 多模態推理 benchmark（ICML25 **O**）、EmbodiedBench（ICML25 **O**）、RoboMME（ICML26 **O**）。
  - 標題含 benchmark／dataset 的比例：ICML 2026 約 6%，NeurIPS 只有約 2%，因為 NeurIPS 另有 D&B track。
- **逆問題**
  - Lottery Prior（ICML26 **O**，https://icml.cc/virtual/2026/oral/71097）：以壓縮為基礎的零樣本逆問題求解器，附非漸近誤差界。
  - An analytic theory of CNN inverse problems solvers（ICML26 poster，forum id=GdiPMYeFoG）：LE-MMSE 解析式能重現 U-Net／ResNet 的輸出，PSNR ≳ 25 dB。
  - On the Guidance of Flow Matching（ICML25 S，https://icml.cc/virtual/2025/poster/44016）。
  - Saving Foundation Flow-Matching Priors for Inverse Problems（ICML26 poster，forum id=s8i2EYzmxt）。
- **3D**
  - ReferSplat（ICML25 **O**）、Holi-Spatial（ICML26 **O**）、3ViewSense（ICML26 **O**）。
  - 3D 在 ICML 的占比（13%）低於 NeurIPS（約 20%）。
- **VLM／VLA**
  - Bad Seeing or Bad Thinking? Rewarding Perception for Multimodal Reasoning（ICML26 **O**，forum id=duzdftKtUA）。
  - Pretrained VLA Models are Surprisingly Resistant to Forgetting（ICML26 **O**）。
  - EMMA（ICML25 **O**）。

### 2. 共同特徵
- **貢獻形式**：視覺 oral 與 spotlight 以兩種形式為主。
  - 「反直覺發現＋可推廣的機制或理論＋小方法」：例如 Kamb、Are VLMs Seeing、Compositional Generalization、Random Matrix、Biased Generalization。
  - 「一般化的方法框架，附誤差或收斂分析」：例如 Error Propagation、Rex、Lottery Prior、IMM。
  - 純工程系統型能拿 oral 的，在 ICML 少見。ICLR 2026 的 oral 則有 SANA-Video、Depth Anything 3、NextStep-1 這類大系統，ICLR 對大型 CV 系統明顯較開放。
- **標題常用字**：rethinking、uncovering、why、mechanism、analytic theory、what makes 等詞頻繁出現，主張多寫成「X 其實是 Y」。
- **實驗規模**：
  - 理論和分析類大多用 CIFAR-10、FFHQ、ImageNet-256 等級的資料，或直接拿開源 7B 級 VLM 做推論分析。
  - 摘要中出現「billion、大規模預訓練」字樣的，在 ICML 視覺錄取論文中約 6%，oral／spotlight 中約 8–11%。
  - 摘要含「免訓練／零樣本／inference-time」的約 18–19%；含理論字樣的約 16%；寫成「我們發現或揭示…」的分析句式約 24–29%。
  - 以上是摘要關鍵字統計，屬粗估。據此推斷，多數 oral 與 spotlight 用 1–8 張 GPU 做得出來，但這不是逐篇核對的結果。

### 3. ICML 少見、多半去 CVPR／ICCV／ECCV 的題目
以下比較 ICML 2026、NeurIPS 2025、ICLR 2026 的錄取篇數（‰ 為佔該會議錄取總數的千分比）：

| 題目類型 | ICML 2026 | NeurIPS 2025 | ICLR 2026 |
|---|---|---|---|
| 復原、超解析、去霧等 low-level 任務 | 32 篇（5.0‰） | 7.8‰ | 9.9‰ |
| optical flow／stereo | 4 篇 | 2.1‰ | 有一篇 oral：WAFT |
| event camera | 5 篇 | — | — |
| 3DGS／NeRF | 2.8‰ | 9.5‰ | 7.7‰ |
| 偵測／分割／追蹤等任務專用模型 | 6.9‰ | 12.9‰ | — |

- **low-level 任務**：ICML 收的這類論文多半有理論包裝，例如 LE-MMSE、Normalization Equivariance 的證明、D–P tradeoff 分析。純架構型的「all-in-one restoration 網路」也有錄取，但都只是 poster。
- **event camera**：ICML 2026 有一篇 oral（SECNet），屬例外。
- **很少見的類型**：單一資料集刷 SOTA 的架構論文、資料集驅動的應用（遙測、醫學影像都在 1–3‰）、純 3D 重建系統。

### 4. 被拒的常見理由（未取得審稿全文，以下為替代證據）
- OpenReview 擋住自動存取，ICML 也只公開錄取論文的審稿，因此**沒有一則審稿原文可引**。
- **替代證據一：ICLR 2026 的子分數**（1–4 分制平均）。視覺論文錄取與被拒之間，差距最大的是 contribution（2.69 對 2.30，差 0.39），其次是 soundness（2.84 對 2.53，差 0.31）與 presentation（2.88 對 2.60）。非視覺論文的差距型態相同。也就是說，主要卡在「貢獻是否夠大」。
- **替代證據二：ICLR 2026 視覺論文的錄取率**：acc/(acc+rej) 為 43.3%，非視覺為 37.9%，視覺並沒有被歧視。分類型看，安全／魯棒性（34.6%）、表徵學習（35.6%）、「免訓練」標題（35.8%）、OCR／文件（約 27%，樣本只有 21 篇）偏低；逆問題／復原約 55%，偏高。
- **官方立場**：ICML 2026 CFP 要求「claims must be supported by reproducible experiments and/or sound theoretical analysis」，並要求把貢獻放在「broader ML literature」的脈絡中。ICML 2012 的 PC 指引則明說 application paper 要特別審視，因為 ICML 主要是 ML 技術會議。
- **推論，未經審稿原文證實**：常見拒稿理由應為貢獻增量、只對特定任務有效、缺乏可推廣的洞見。

### 5. 選題準則（5 條）
1. **結論句要是關於「一類模型」的機制或定理。**例如「擴散模型的局部性來自 X」，不要寫成「我的網路在 Y 資料集上更好」。在 ICML，視覺論文要拿到 spotlight 以上，幾乎都是這種寫法。
2. **要有明確的對立主張可推翻。**最好直接回應一篇高引用論文的歸因，例如 Lukoianov 對 Kamb，或 Are VLMs Seeing 對 VLM「再檢查」的假設。
3. **要有可計算的 oracle 或理論上限做對照**，例如最佳 MMSE、經驗最佳 denoiser、完整計算、精確反演，讓「還差多少」可以量化。
4. **實驗用 CIFAR／FFHQ／ImageNet-256 等級的資料，或開源 ≤8B 模型的推論分析，就能撐起主張。**不要選需要從頭預訓練才能說服審稿人的題目。
5. **避開 ICML 稀有、偏 CV 應用的包裝**（例如純復原架構、單任務 SOTA）。若做 low-level，要包成逆問題或估計理論。

### 6. 符合準則、看起來還有後續空間的方向
以下只是線索，撞題檢查未完整做，需要其他範圍再查。

**(a) 訓練後視覺 denoiser／復原網路的解析理論：從 CNN 推到 attention 與擴散後驗取樣**
- 觸發論文：
  - Kamb & Ganguli（ICML25 oral）。
  - Niedoba et al.（ICML25 spotlight）。
  - Lukoianov et al.（NeurIPS25）：主張局部性來自資料統計，與 Kamb 的架構歸因直接衝突，因此主張可證偽。
  - LE-MMSE（ICML26 poster）：已把解析理論做到 CNN 逆問題求解器。
- 尚未看到的部分：transformer 和 attention 型復原網路，以及「擴散式超解析產生幻覺」的機制解釋。
- 改進空間的證據：據搜尋摘要，Kamb 的理論在含 attention 的 UNet 上，CIFAR10 的中位數 r² 只有約 0.77（未從原文確認）。
- 撞題風險：LE-MMSE 已經佔掉 CNN 這一塊。

**(b) 免訓練擴散加速（量化、cache）的誤差傳播理論與預算最佳策略**
- 觸發論文：Error Propagation（ICML26 oral）、BudCache、LearniBridge（ICML26）、Hybrid Feature Caching（ICLR26 oral）。
- 判斷：ICML 願意收「誤差封閉解＋補償」這種寫法。但 cache 方法本身非常擠，沒有理論貢獻很難差異化。

**(c) VLM 推理時是否真的使用視覺 token：機制分析加上輕量修正**
- 觸發論文：Are VLMs Seeing（ICML26 oral）、Bad Seeing or Bad Thinking（ICML26 oral）、Visual symbolic mechanisms（ICLR26 oral）、Learning to See Before Seeing（ICLR26 oral）。
- 算力：7B 級模型的推論與 attention 分析，用 1–4 張 H200 足夠。
- 風險：熱門題目，撞題機率高。

**(d) 基礎生成先驗在逆問題上的失效與修正**
- 觸發論文：
  - FMPlug（ICML26）摘要直接寫，基礎 flow-matching 先驗「trail behind domain-specific and even untrained priors」，這是公開的失敗陳述。
  - Lottery Prior（ICML26 oral）。
  - Stage-wise D–P Traversal（ICML26）。
- 契合度：與研究者的復原背景最契合。需要包成先驗失配或可辨識性的理論問題。

**(e) Tokenizer 的 latent 幾何如何決定生成品質**
- 觸發論文：MAETok（ICML25 spotlight）、ViTok-v2（ICML26，發現壓縮率的最佳值隨生成器大小改變）、VideoFlexTok（ICML26 spotlight）、InfoTok（ICLR26 oral）。
- 風險：要訓練 tokenizer 加 DiT 才能驗證，算力與時程最吃緊，所以信心最低。

### 7. 看過但不建議的方向
- **純 all-in-one 或真實世界超解析架構**：ICML 每年只有個位數，且多半只是 poster，屬 CV 會議題目。
- **新 VLM benchmark**：ICML 有收，但需要大量標註，且 NeurIPS D&B 和 CVPR 競爭激烈，10 週內做不出有說服力的規模。
- **VLA／機器人 world model**：ICML 2026 收很多，但需要大規模機器人資料或模擬算力，超出條件。
- **大型影片生成系統**（SANA-Video、Depth Anything 3 這類）：ICLR oral 傾向收，但都是大型團隊的訓練規模，不可行。
- **AI 生成圖偵測、浮水印、概念抹除**：數量多，但 ICLR 的錄取率偏低（安全類約 35%），而且容易被評為增量。
- **diffusion cache／token 合併的純經驗方法**：ICLR、ICML 都已大量收錄，沒有理論就很難差異化。

### 8. 限制
- 類型分類是用標題正規表示式，屬粗分類。
- ICML 2026 的 oral 數字是從 icml.cc 頁面人工比對而來。
- 審稿意見原文完全沒有取得。
- 上述統計所用的 JSON 檔，以及統計腳本（scripts 資料夾內的 cls.py、cls2.py、iclr_types.py），放在 /tmp/claude-1047/-work-ChihChungHsu-cactus8603-icml27-s2-gated-sink-kv/bd65f9a7-961a-4b8d-adce-42df28fbb74b/scratchpad/。沒有改動 repo 內任何檔案。

---

5 個子代理各自是否完成：A 完成、B 完成、C 完成、D 完成、E 完成。
