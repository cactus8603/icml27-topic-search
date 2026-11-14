## 結論

這個類比撐不起一篇 ICML 論文的核心想法。ELSA 是一個「數值上精確」的 attention kernel：它不改變模型算的函數，只把計算重新加括號，讓原本的循序遞迴變成平行掃描。它所有的性質，都來自 softmax 歸約滿足一個「可交換、可結合的 monoid」（monoid：一種合併運算，合併的先後分組不影響結果）。NCM 在尺度軸上的循序性有兩個來源：殘差量化是非線性的 argmin 遞迴，生成時同一尺度內是條件獨立取樣。兩者都不具有這種代數結構。NCM 的平行是拿機率模型的近似（同一層內因子分解）換來的；ELSA 的平行是免費的，完全不損失精確度。ELSA 的性質中，唯一能精確搬到 NCM 的是「tokenizer 累積重建 ẑ^(K)=Σ_k up(e_k) 是一個加法前綴和」，但這件事本來就不證自明（trivial），對計算量也沒有意義（K=12）。

---

## 第一步：找出 ELSA 與 HTCS

### ELSA（已確認，信心高）
- **原文資訊**：*ELSA: Exact Linear-Scan Attention for Fast and Memory-Light Vision Transformers*。作者 Chih-Chung Hsu、Xin-Di Ma、Wo-Ting Liao、Chia-Ming Lee，arXiv:2604.23798（2026-04-26，只有 v1），arXiv 註記寫「Accepted to CVPRF2026」，也就是 **CVPR Findings 2026**，不是主會議。程式碼在 github.com/ming053l/ELSA。我讀了 arXiv HTML 全文，包含補充材料 A–J。
- **認定依據**：ACVLab 成員 Chia-Ming Lee 的個人頁（ming053l.github.io）把它列為 CVPR Findings 2026，並寫得到「CVPR 2026 Computational Transparency Award」。CVPR 官方的獎項名稱是「Compute Transparency Champion」，我的搜尋只找到其他論文得獎的公告，沒有官方來源列出 ELSA。這個得獎說法來自作者自己的頁面，我無法核實。
- **其他候選**：
  - 同一個個人頁還列了一篇期刊稿 *ELSA: Two-Phase Standard-Softmax Scan Attention with Path-Specific Precision and Carried Statistics*（Hsu, Ma, Lee, Pan, Liao），標示為「Journal Manuscript」，沒有連結，我讀不到全文。
  - 另一篇同名的 *ELSA: Enhanced Local Self-Attention for Vision Transformer*（Zhou, Wang 等人，arXiv 2112.12786，2021）與 Hsu 無關，排除。

### HTCS（找不到，信心低）
- **試過的查詢**：「HTCS ACVLab Chih-Chung Hsu」；「"HTCS" attention scan OR transformer OR "carried statistics" Hsu」；「"HTCS" arXiv 2026 language model OR diffusion OR hierarchical OR coarse-to-fine」；「Chih-Chung Hsu 2026 … "Two-Phase" OR "HTCS"」。另外在 ELSA 全文與 GitHub README 裡搜過字串，你的 repo 裡也沒有任何地方提到 HTCS 或 ELSA。
- **結果**：完全沒有出現 HTCS 這個縮寫。ELSA 原文裡以 H 開頭的縮寫只有 H2D 和 HMMA。
- **最接近的線索（純屬我的猜測）**：
  - ELSA 的兩層掃描：區塊內用 **H**illis–Steele，區塊間用 Blelloch。
  - 期刊稿標題裡的「Two-phase … **C**arried **S**tatistics」。
  - 也可能是口頭傳述時記錯的名稱。
  - 我無法確認以上任何一個。

---

## 第二步：計算圖

### ELSA（原文內容）
- **輸入／輸出**：一般的 Q、K、V，輸出就是 softmax(QKᵀ)V 本身。不訓練新模型，也不改權重。
- **軸**：批次 b、注意力頭 h、query 位置 i、key 位置 j、特徵 d_v。沒有尺度軸，也沒有迭代或時間步軸。
- **狀態與合併**：對每個 query，在 key 軸 j 上維護狀態 (m, S, W)，分別是執行中的最大值、正規化的累加和、加權的 value 累加。
  - 合併 ⊕（式 4）：當 m_a ≥ m_b 時，u_a⊕u_b = (m_a, S_a+S_b·e^{m_b−m_a}, W_a+W_b·e^{m_b−m_a})，反之兩者對調。單位元是 (−∞, 0, 0)。
  - 依這個定義，⊕ 還是**可交換**的（這是我的推導）。
- **循序與平行**：
  - key 軸原本是循序遞迴（ME-SDPA，深度 O(n)），改成兩層前綴掃描：區塊內 B=128 個 token 用 Hillis–Steele，區塊間用 Blelloch。深度 L(n,B) ≤ 2⌈log₂n⌉+3。
  - query、head、batch 都是平行的。
  - 沒有 residual，也沒有逐級重建。
- **訓練目標**：沒有，這是 kernel，屬於推論端。原文另外用 ELSA 從頭訓練了 ImageNet 模型，只用來展示收斂不受影響。
- **推論流程**：直接替換現有的 attention（drop-in）。

### ELSA 的性質與證據等級（原文）
1. ⊕ 封閉、可結合、有單位元，構成 monoid：**有證明**（補充 A.2）。
2. 區塊組合，即 Proposition 1：u(B₁∪…∪B_K)=⊕u(B_k)，**由第 1 條推得**。另外有實驗驗證（圖 7，B=128）。
3. 在實數運算下輸出與原本的 attention 完全相同：**由第 1、2 條推得**。實驗上與 float64 參考實作比對，argmax 不一致率為 0。
4. FP32 相對誤差 ≤ u·L(n,B)+O(u²) ≤ u(2⌈log₂n⌉+3)+O(u²)：這是 Theorem 1，**有證明（proof sketch）**。依據 Higham 模型、Lemma 1，並需要 Assumption 1（下溢的指數項當作 0；exp 函式庫的相對誤差 ≤ κu）。拿來對照的「循序寫法為 O(nu)」是**宣稱**。
5. 誤差是隨機噪聲、沒有空間結構：**有實驗**（圖 6）。
6. 額外記憶體 O(n)：**推導加實測**（16K token、FP32 時約 0.34 GB）。
7. 加速：A100 上 FP32 1K–16K token 時 1.3–3.5×；Jetson TX2 約 1.6×；LLaMA-13B offload 在 ≥32K token 時吞吐量多 17.8–20.2%。**有實驗**。原文自己承認短序列時輸給 FA2/FA3，而且算術複雜度仍是 O(n²)。
8. 不用重新訓練、可直接替換：由第 3 條推得，並在 CLIP、ViT、Swin、BERT、VGGT、HSIMAE 上**有實驗**。
9. 訓練收斂不受影響：**有實驗**（圖 5，ImageNet）。
10. 不依賴 Tensor Core、不挑硬體：**設計性質加實驗**。

ELSA **沒有**以下這些性質，原文完全沒有對應內容，不能當作要轉移的來源：
- 可提早停止
- 粗到細生成
- 各階段可獨立評估
- 可壓縮性
- 對輸入擾動的穩健性

### NCM
以你提供的事實為準，我沒有另外驗證。我只算了一件事：LM1B 的 Σℓ_k = 641。

---

## 第三步：「轉 90 度」的檢驗

### 這裡的「轉 90 度」是哪兩個軸對調（我的推導）
- **最合理的讀法**：ELSA 的計算網格是 query i × key j，沿 j 掃描、沿 i 平行；NCM 的網格是位置 p × 尺度 k，沿 k 循序、沿 p 平行。所以對應是 **key 軸 j ↔ 尺度軸 k，query 軸 i ↔ 位置 p**。
- **嚴格說，這不是「轉」**：
  - ELSA 沒有把軸對調，它在**同一個軸**上把線性鏈重新加括號成一棵樹。
  - 真正算得上「把網格轉 90 度」的，是「自回歸（沿位置循序）→ NCM（沿尺度循序、沿位置平行）」。
  - 這一點本身就說明，這個直覺混淆了兩件事。

### 對照表

| ELSA | NCM 的對應 | 等級 |
|---|---|---|
| key 軸 j（被掃描的軸） | 尺度 k | 鬆散 |
| query 軸 i（平行的軸） | 位置 p | 鬆散 |
| 前綴狀態 u_{1:j}=(m,S,W) | 累積重建 ẑ^(k) | 只在 tokenizer 解碼端精確 |
| ⊕（可交換、可結合） | 向量加法 ẑ+up(e_k) | 解碼端**精確**同構（加法本身就是 monoid），但不證自明 |
| 各區塊只依賴自己的 token，可獨立計算 | 第 k 層的 code 依賴殘差 z−ẑ^(k−1)（編碼端），也依賴前面各層的條件（生成端） | **對不上** |
| 區塊組合（Proposition 1）的結果與原本的計算完全相同 | 同一層內條件獨立取樣 ≠ 聯合分佈 | **對不上** |
| 平行化不損失任何精確度 | 平行化讓品質輸給 AR，還需要加懲罰 | **相反** |
| 只有浮點捨入誤差（約 u log n） | 量化誤差、取樣誤差、exposure bias | **對不上**（誤差種類不同） |
| 連續、確定性、資訊完全守恆 | 離散、隨機取樣、有損量化 | **對不上** |
| 沒有訓練目標 | VQ 重建、nested dropout、生成器交叉熵 | **對不上** |
| 合併順序無關（可交換） | 粗到細的順序很關鍵（nested dropout 只截斷後綴） | **相反** |

### ELSA 每條性質搬到 NCM 上的判斷
- **monoid 與區塊組合（第 1、2 條）**
  - 在 tokenizer 解碼端屬於 (1)：可由同樣的論證直接推得，Σ_k up(e_k) 可以任意分組相加。但只有 K=12 項，平行化沒有好處。
  - 在編碼端與生成器上屬於 (3)：ẑ^(k)=ẑ^(k−1)+up(Q(down(z−ẑ^(k−1)))) 是非線性遞迴，argmin 沒有結合律。取樣是條件分佈，沒有一個「合併獨立算出的狀態」的運算子。
- **精確等價、不用重新訓練（第 3、8 條）**：屬於 (3)。NCM 的平行本身就是一個不同的模型。
- **誤差界 O(u log n)（第 4 條）**：屬於 (3)。捨入誤差不是 NCM 的瓶頸。如果換成「擾動沿尺度傳播的上界」，就變成 (2)，需要額外假設：生成器對 ẑ 的條件是 Lipschitz 的、codebook 向量的範數有界、而且後面各層用 teacher forcing 固定。在後面各層固定的前提下，ẑ 的變化剛好等於 up(e′_k−e_k)，這是加法結構直接給的；若後面各層重新生成，就沒有任何上界。
- **記憶體與速度（第 6、7 條）**：屬於 (3)。NCM 生成器每一層的 attention 寬度 ≤128 加上 prefix，總共約 641 個位置，正好落在原文承認 ELSA 會輸給 FA2 的區間。
- **隨機噪聲、訓練收斂不受影響（第 5、9 條）**：只在「把 ELSA kernel 塞進 NCM 生成器」這個工程意義下屬於 (1)，對 NCM 的方法本身沒有任何意涵。

### 用現有 LM1B 權重、少量 GPU 可做的實驗（針對 (1)(2) 類）
1. **擾動傳播**：把第 k 層的 code 換成隨機 code 或次近鄰 code，分兩種情況量測解碼結果的 token Hamming 距離與 gen-PPL 隨 k 的變化：(a) 後面各層固定；(b) 後面各層重新生成。這可以直接檢驗「擾動有界／會自我修正 vs 會放大」，並與訓練時的隨機 code 替換對照。
2. **非後綴移除**：只拿掉中間某一層 j（不是截斷後綴），解碼並量測正確率。這是在檢驗「可結合、可交換」在 NCM 上失敗的程度。預期掉得很明顯，因為 nested dropout 只訓練過後綴截斷。
3. **ELSA 式的數值稽核**：固定 RNG，比較 FP32 與 BF16 生成器在各層的 logits argmax 不一致率與取樣分歧。
4. **把 ELSA kernel 換進生成器**：驗證輸出等價（誤差約 1e-6 以內）並量測延遲。預期沒有加速。這只是工程上的健全性檢查（sanity check）。

### 反方向：NCM 的現象在 ELSA 那邊有沒有對應
- **粗層不可預測**：ELSA 沒有任何預測或生成成分，沒有對應。
- **同層獨立取樣需要懲罰補償**：ELSA 因為精確，根本不會碰到這個問題，沒有現成解法可以借。
- 相關的解法屬於其他文獻（例如 MaskGIT 式依信心重新 mask、離散 diffusion 的 remasking、Jacobi／平行定點解碼）。這是我的背景知識，這次沒有開啟確認。
- **我的推測**：如果真的想借「把循序遞迴平行化成掃描」這個精神，對應的工具是把非線性遞迴線性化後做 associative scan 的那類方法（例如 DEER 一類的 Newton／定點迭代平行化），而不是 ELSA；同樣是背景知識，未查證。而且 NCM 只有 K=12，平行化的收益有限。

---

## 第四步：判斷

**撐不起 ICML 論文的核心。** 卡在三個地方：
1. ELSA 的所有保證都來自「計算的函數不變」，這在 NCM 的生成端不成立。
2. 唯一精確的同構（加法前綴和）不證自明，而且沒有計算上的意義。
3. NCM 真正的問題（同層獨立取樣導致品質損失、粗層不可預測），ELSA 完全沒有觸及。

如果硬要寫，一句話主張大概會是「coarse-to-fine 文字生成的尺度遞迴，可視為類似 attention 掃描的前綴累積」。審稿人幾乎一定會問：
- 哪一條定理真的轉移過來了？
- 精確性在哪裡？
- 為什麼不直接引用 RVQ 本身的加法結構就好？

### 「HTCS + ELSA」可能指什麼（猜測）
如果 HTCS 是一個獨立的階層式或粗到細的方法，「HTCS + ELSA」最可能的意思是「在階層式模型裡用 ELSA kernel 加速 attention」。那是工程上的組合，不是理論性質的轉移；而且在 NCM 這種短序列、block-diagonal 的設定下，ELSA 原文自己的數據就顯示不會有收益。要不要繼續追，建議先向提出者問清楚 HTCS 的全名。

---

## 實際開啟過的來源
- https://arxiv.org/abs/2604.23798 與 https://arxiv.org/html/2604.23798v1（全文；HTML 存於 /tmp/claude-1047/-work-ChihChungHsu-cactus8603-icml27-s2-gated-sink-kv/bd65f9a7-961a-4b8d-adce-42df28fbb74b/scratchpad/dl/elsa.html）
- https://ming053l.github.io/ELSA_projectpage/
- https://ming053l.github.io/
- https://raw.githubusercontent.com/ming053l/ELSA/main/README.md

沒有修改任何 repo 檔案。
