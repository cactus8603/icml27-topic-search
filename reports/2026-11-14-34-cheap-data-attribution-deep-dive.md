# 查核報告：低成本資料歸因追上 metagradient 精確值

**結論：有條件做，而且偏向不做。** 切入點 (b) 已經被 2609.15044 做掉。切入點 (a) 還沒找到撞題的論文，但這個題目的動機（便宜方法接近隨機、MAGIC 是可靠的上限）在 LLM 微調的情境下站不住。

## 閱讀程度說明
- **讀了全文**：2609.15044（Spell／Mage，含附錄，主表逐格核對）、2504.16430（MAGIC，含附錄）、2604.07328（Gunn，讀了引言、技術概要和相關研究，第 3、4 節證明沒讀）、2606.05165（STRIDE，讀到第 5 節）。
- **讀了部分內文**：2606.11660（Bergson 論文，讀第 5 節和附錄 A）、2605.18814（只用關鍵字抽讀）。
- **讀了文件**：Bergson 的 README、排行榜和方法說明頁。
- **讀了程式碼**：andrewilyas/simple-magic-jax 的 main.py。
- **只讀摘要和表格標題**：2507.14740（ASTRA）、2509.26544（BIF）。
- **只讀摘要**：2610.40655、2609.33667、2609.38767、2610.24721、2609.32380、2610.27689、2610.14809、2610.15336。
- **沒讀到**：DATE-LM（2507.09424）只下載了 HTML，沒讀內文，只確認它的程式連結是公開的。NeurIPS 2026 那篇「Efficient and Accurate Evaluation of TDA」在 arXiv、Semantic Scholar、OpenReview 都找不到。

---

## 第 1 題：現象或差距是否屬實

### 核心數字總表

| 論文 | 模型與資料 | 指標 | 數字 | 程式與權重 |
|---|---|---|---|---|
| MAGIC 2504.16430 | ResNet-9／CIFAR-10（50 個查詢）；GPT-2 125M／WikiText；Gemma-2B LoRA／LESS 指令資料（32 題 MMLU） | 單模型 LDS，移除比例 0–20% | 移除少量時 LDS 接近 1.0；基線「below 0.4」（圖 4，只讀到圖說，數值無法從圖中取出）。單一範例的數字：CIFAR 上 MAGIC 0.98/0.98、EK-FAC 0.5/0.18、TRAK 0.24/0.7；Gemma 上 MAGIC 0.95/0.99、TRAK 0.06/0.13 | 論文本身沒附程式。官方極簡版 github.com/andrewilyas/simple-magic-jax 於 2026-10-20 建立，只有 CIFAR，README 說可得 LDS >0.9。第三方實作在 github.com/EleutherAI/bergson（PyTorch，支援 FSDP） |
| Spell／Mage 2609.15044 | ResNet-9 10M／CIFAR-10；GPT-2 式 92M／TinyStories 微調（n=2048、256 步） | LDS（Spearman），M=300 個重訓模型，移除 1%／5%；相對 Frobenius 誤差 | 見下方逐格核對 | 找不到程式（論文內沒有連結，GitHub 搜尋也沒有） |
| Bergson 排行榜（EleutherAI） | GPT-2／WikiText 微調 4 epoch，與 MAGIC 論文相同設定，查詢 m=50，重訓 N=400 | LDS | MAGIC 0.931；MAGIC 跨種子 0.829；Unrolled Gauss-Newton 0.819；EKFAC+CG 0.685；EKFAC+ASTRA 0.605；LiSSA 0.572；SOURCE 0.473；EKFAC 0.454；TrackStar 0.276；TRAK（8 模型集成）0.138 | 公開，排行榜附設定檔 |
| STRIDE 2606.05165 | nanochat 286M–1.38B 預訓練；Qwen2.5-0.5B 指令微調 | LDS | 預訓練約 0.16–0.18；SFT 約 0.16–0.40 | 沒找到自己的程式連結 |

### 2609.15044 主表逐格核對

數字都從 HTML 數學式的 alttext 原文抽出，平均 ± 標準誤。

**表 1：ResNet-9／CIFAR-10，k=500，10 個種子**

| 預算 | 方法 | 相對 Frobenius 誤差 | LDS（移除 1%） | LDS（移除 5%） |
|---|---|---|---|---|
| B=50 | Mage | 0.4872±0.0064 | 0.2358±0.0017 | 0.1698±0.0032 |
| B=50 | Spell | 0.8419±0.0061 | **0.3353±0.0023** | 0.2696±0.0045 |
| B=50 | Random probes | 0.6231±0.0049 | 0.2461±0.0013 | 0.1803±0.0032 |
| B=50 | PCA | 0.5316±0.0061 | 0.1541±0.0010 | 0.1327±0.0017 |
| B=50 | First-B queries | 0.9177±0.0040 | 0.2226±0.0024 | 0.1568±0.0032 |
| B=50 | Rank-B SVD（oracle） | 0.4483±0.0058 | 0.2328±0.0022 | 0.1599±0.0030 |
| B=50 | Spherical SVD（oracle） | 0.8611±0.0049 | 0.3948±0.0023 | 0.2852±0.0056 |
| B=100 | Mage | 0.1614±0.0034 | 0.3425±0.0017 | 0.2570±0.0040 |
| B=100 | Spell | 0.7845±0.0064 | **0.4538±0.0023** | 0.3708±0.0052 |
| B=100 | Random probes | 0.3177±0.0045 | 0.3572±0.0013 | 0.2721±0.0041 |
| B=100 | PCA | 0.1869±0.0036 | 0.2337±0.0013 | 0.2006±0.0021 |
| B=100 | First-B queries | 0.8542±0.0048 | 0.3405±0.0020 | 0.2538±0.0034 |
| B=100 | Rank-B SVD（oracle） | 0.1545±0.0031 | 0.3420±0.0016 | 0.2545±0.0040 |
| B=100 | Spherical SVD（oracle） | 0.7745±0.0047 | 0.5189±0.0026 | 0.4007±0.0057 |
| B=k | MAGIC | – | **0.9087±0.0038** | 0.7778±0.0058 |

**表 2：GPT-2／TinyStories，k=500，4 個種子**

| 預算 | 方法 | 相對 Frobenius 誤差 | LDS（移除 1%） | LDS（移除 5%） |
|---|---|---|---|---|
| B=50 | Mage | 0.6986±0.0012 | 0.2980±0.0018 | 0.2970±0.0017 |
| B=50 | Spell | 0.8206±0.0013 | 0.3931±0.0004 | 0.3947±0.0017 |
| B=50 | Random probes | 0.8165±0.0010 | 0.3191±0.0014 | 0.3221±0.0018 |
| B=50 | PCA | 0.7104±0.0012 | 0.2275±0.0022 | 0.2258±0.0012 |
| B=50 | First-B queries | 0.9478±0.0003 | 0.2683±0.0015 | 0.2704±0.0014 |
| B=50 | Rank-B SVD（oracle） | 0.6847±0.0009 | 0.3164±0.0027 | 0.3168±0.0023 |
| B=50 | Spherical SVD（oracle） | 0.8777±0.0008 | 0.4618±0.0016 | 0.4654±0.0018 |
| B=100 | Mage | 0.5257±0.0013 | 0.4341±0.0009 | 0.4350±0.0017 |
| B=100 | Spell | 0.7310±0.0012 | 0.5451±0.0006 | 0.5488±0.0009 |
| B=100 | Random probes | 0.6665±0.0009 | 0.4543±0.0005 | 0.4566±0.0021 |
| B=100 | PCA | 0.5429±0.0013 | 0.3428±0.0009 | 0.3416±0.0012 |
| B=100 | First-B queries | 0.8777±0.0002 | 0.4120±0.0010 | 0.4136±0.0011 |
| B=100 | Rank-B SVD（oracle） | 0.5075±0.0012 | 0.4622±0.0006 | 0.4635±0.0015 |
| B=100 | Spherical SVD（oracle） | 0.7711±0.0004 | 0.6130±0.0007 | 0.6217±0.0018 |
| B=k | MAGIC | – | 0.9952±0.0004 | 0.9964±0.0002 |

**和前一輪轉述的比對：** 0.454、0.335、0.909 三個數字都正確。但前一輪的描述有三處偏差：

1. **Spell 並不「低成本」。** B=100 就是 100 次完整的 replay。每次 replay 的成本是一次訓練的 3–5 倍（MAGIC 論文）或 2–3 倍（ICLR 2026 投稿摘要）。
2. **Spell 的表裡完全沒有 TRAK 或 EK-FAC。** TRAK 只在引言出現一次，所以「低成本方法對 MAGIC」並沒有同一張表裡的直接對照。
3. **投影類方法本身有上限。** 連 oracle 的 Spherical SVD 在 B=100 時也只有 0.519（CIFAR）和 0.613（TinyStories）。差距大半來自「用 B 個量測張成的子空間去投影」這種估計方式本身的結構限制。

**附錄 C 的補充：** CIFAR 上 k=1000 時，影響矩陣各列的大小橫跨 8.6 個數量級，最大的四分之一查詢佔了 99.2% 的能量。查詢梯度的大小和這些列的大小，排序相關達 0.999（CIFAR）和 0.996（TinyStories）。

**判斷：** 在 CIFAR、單模型（固定種子）、移除 1% 的設定下，差距屬實。「TRAK、EK-FAC 在 LLM 微調上接近隨機」這個說法只對 TRAK 成立。同樣是 GPT-2／WikiText，調好的 EKFAC+CG 有 0.685，Unrolled GN 有 0.819。

---

## 第 2 題：是否已被做掉

2026 年直接處理「以較低成本逼近 metagradient 歸因」的論文：

1. **2609.15044 Spell／Mage**（2026-09-14，目前是 ICLR 2027 投稿）。核心正是用 B 次 replay 量測多個查詢的線性組合，從中重建影響矩陣。這就是本題的主幹。
2. **2604.07328 Gunn「How to sketch a learning algorithm」**（v3，2026-06）。用前向模式自動微分，在隨機的資料權重方向上量測，得到「不依賴查詢」的無偏草圖，並有理論保證。程式在 github.com/SamSpo1/microgpt-sketch，但只在 microgpt 上做實驗。Spell 的 forward-mode 步驟就是借用這篇。
3. **Bergson 的 `gnh-unroll`（Unrolled Gauss-Newton）。** 在 MAGIC 的反向傳播中只看線性層，並用 Gauss-Newton 取代 Hessian，支援一次處理多個查詢。在 GPT-2／WikiText 上 LDS 0.819，MAGIC 是 0.931。這是 MAGIC 的便宜近似，表現已經不錯。我沒找到它的獨立論文，只看到文件和排行榜。
4. **2605.18814**（AdamW-influence，軌跡型歸因的誤差分析）。屬於讓便宜的展開近似更準的方向，作者自己說沒有涵蓋 MAGIC。

**判斷：**
- 「只用少量 replay 同時歸因很多查詢」這個方向已經被 Spell 佔掉。
- 在 LLM 微調上，便宜方法和精確方法的差距已被 EKFAC+CG（0.685）和 Unrolled GN（0.819）補掉大半。
- 在 CIFAR 上還有大差距，而且沒有人公開過 Unrolled GN 在 CIFAR 的數字。
- 要注意 Spell 和 MAGIC 改寫版（「MAGIC: Data Attribution via Exact Influence」）都在 ICLR 2027 審查中，同一組作者很可能在 1 月前把明顯的延伸做掉。

---

## 第 3 題：最強的反駁

**有。最強的是：在同一個 GPT-2／WikiText 設定下，比 MAGIC 便宜、而且能一次處理多個查詢的 Unrolled Gauss-Newton 已經有 LDS 0.819，MAGIC 也只有 0.931。** 換個評測方式（跨種子），MAGIC 本身也掉到 0.829，和便宜方法幾乎一樣。所以 LLM 端「便宜方法接近隨機、精確值高達 0.9 以上」的落差，大半來自評測設定（固定種子、為了 metasmooth 調過的訓練流程）和沒調好的基線，不是方法上的根本限制。

其他幾點支撐的反駁：
- **上限的可信度。** MAGIC 的 0.9 以上只在幾個條件同時成立時才有：單模型且重訓時固定所有隨機性、移除比例小、訓練流程是 metasmooth。CIFAR 上移除 5% 就降到 0.778，到 20% 下降更多（只看到圖，沒有數值）。MAGIC 自己說常需改動訓練（例如 Adam 的 ε_root、先預訓練幾百步來取得平滑性），Bergson 也警告早期預訓練的 metasmoothness 很低。
- **下游實用性差距小。** 移除前 1% 資料後查詢損失的增幅（Bergson 叫 QLD）：MAGIC 0.100、EKFAC+CG 0.077、BM25 0.062。LDS 差 0.25，實際影響只差兩成多。
- **「低成本」這個前提有問題。** Spell 在 B=100 時約等於 300–500 次訓練的計算量，比 TRAK、EK-FAC 貴好幾個數量級。題目要「低成本」又要「追上精確值」，要先講清楚是和誰在等成本下比較。Spell 沒做這件事，所以這也是一個缺口（見第 5 題）。

---

## 第 4 題：預想切入點是否已有人做

### (a) 用 TRAK／EK-FAC 當 side information 或 control variate，只用少量 metagradient 校正殘差

- **沒找到直接做這件事的論文。** 我用 arXiv API 查了 control variate、multi-fidelity、budget 搭配 data attribution 等關鍵字，也看了 MAGIC、Spell、Gunn、STRIDE 的引用者，都沒有。
- **最接近的是 Spell。** 它用最終查詢梯度的 Gram 矩陣當代理資訊來挑量測方向，但估計值只取已量測子空間上的投影，沒量到的部分直接當成零，完全沒用便宜的歸因分數。重疊程度：低到中。
- **Gunn 的前向模式草圖**是可以拿來做 control variate 的量測工具（無偏、可在查詢之間共用），但 Gunn 沒有結合便宜的歸因方法。重疊程度：低。
- **Bergson 的 Unrolled GN** 是最強的「便宜側資訊」候選，但它本身就是單獨的方法，沒和精確值做校正。
- **評估：** 新穎性還在，但缺口窄、是很明顯的下一步，被同組（Ilyas 也是 TRAK 的作者之一）或 Gunn 搶先的風險中到高。

### (b) 把多個查詢隨機線性組合成一次 metagradient，並依查詢梯度大小分配量測預算

- **已被做掉。**
  - Spell 第 3.1 節「One Replay Can Measure Combinations of Queries」就是這個做法。它的「Random probes」基線就是隨機高斯組合，B=100 時在 CIFAR 上 LDS 0.357。
  - 依大小分配預算：Spell 用估計出的列大小加權（並設第 10 百分位作為下限）。附錄 C 明確指出查詢梯度大小和列大小的排序相關達 0.999／0.996，並建議用它事先決定要用 Mage 還是 Spell。
  - MAGIC 原論文也把「一次算多個查詢」列為未來工作。
- **重疊程度：高（約九成）。不建議當主要貢獻。**

---

## 第 5 題：還沒人回答的問題、可能成立的主張、不該做的事

### 可檢驗但沒人回答的問題
1. **等計算量的比較。** 把 TRAK、EK-FAC、EKFAC+CG、Unrolled GN、Spell、MAGIC 畫在同一張「LDS 對 GPU 小時」的帕雷托圖上。Spell 和 Bergson 都沒做。
2. **Unrolled GN 在 CIFAR ResNet-9（單模型設定）的 LDS 是多少。** 如果它已經 ≥0.8，CIFAR 端的動機也就不成立了。
3. **混合估計能不能突破投影類方法的上限。** 也就是在已量測子空間上用精確投影、在正交補空間上用便宜方法的分數去補，能否超過 Spherical SVD 的 0.519。Spell 的估計方式在結構上做不到。
4. **跨種子 LDS 下，Spell 和 MAGIC 的排序是否不變。**
5. **移除比例 5–20% 時，「精確值是上限」還成不成立。**

### 10 週內最可能成立的論文主張
「多保真度資料歸因：把便宜的歸因估計當 control variate 或先驗，配合少量 replay 或前向模式量測。在等計算量下優於 Spell，並超過投影類方法的 oracle 上限。」實驗做 CIFAR-10 ResNet-9 和 GPT-2／WikiText（Bergson 的設定），附等計算量帕雷托圖。

**提醒：** 這只是一個中等的增量貢獻。審稿人一定會問 Unrolled GN，以及單模型設定還有沒有意義。

### 不該做的事
- 把 (b) 當主要貢獻。
- 在 ≥1B 模型上產生 MAGIC 的精確真值。
- 以「TRAK／EK-FAC 接近隨機」作為動機。
- 宣稱 MAGIC 是普遍適用的上限。

---

## 第 6 題：一週內、≤50 GPU 小時的停損實驗

### 需要的公開資產（除了 Spell 程式外都拿得到）
- **simple-magic-jax（JAX，CIFAR）。** 它的 `ex_target_fn` 能接受任意 tangent，所以把查詢線性組合送進去很容易。它的超參數和 Spell 附錄 B 幾乎一樣（BS 500、20 epoch、momentum 0.85），差在 LR 和 peak time。授權是「Other」，使用前要確認條款。
- **Bergson。** 有 MAGIC、`gnh-unroll`、EKFAC(+CG)、TRAK、`bergson validate` 重訓驗證，也附 GPT-2／WikiText 排行榜的設定檔。
- **其他：** MadryLab/trak、kronfluence（EK-FAC）。CIFAR-10、WikiText、TinyStories 都是公開資料。
- **Spell 程式沒有公開。** 需要依演算法 1 自己重寫，包括前向模式那一步。演算法寫得算清楚。

### 一次評測要多久（估計值，未實測）
- MAGIC 論文說 CIFAR 上一次 metagradient 在一張 A100 上約 15 分鐘，是訓練的 3–5 倍。換到 H200 推估約 7–10 分鐘（未確認）。
- 建議設定：k=100 個查詢、M=200 個重訓模型、移除 1%。
  - 重訓約 200 × 3–5 分鐘，約 10–15 H200 小時。
  - 精確 Y（100 次 replay）約 12–17 小時。
  - 重現 Spell 並跑混合法（B≤20，每次 replay 外加一次前向模式）約 3–5 小時。
  - TRAK、EK-FAC、Unrolled GN 約 2–4 小時。
  - **合計約 30–40 GPU 小時。**
- 真值（重訓結果和精確 Y）建好之後，每個方法的 LDS 評測只是內積加 Spearman 相關，幾秒到幾分鐘。
- simple-magic-jax 每一步都把狀態存到 CPU，以 2000 步計可能要上百 GB 主記憶體。這是我的推測，需要實測。

### 停損門檻建議（供 Fable 決定）
- 在 B=0.1k–0.2k 時，混合法要比 Spell 高 ≥0.08 LDS，而且超過 Spherical SVD 的 oracle。
- Unrolled GN 在 CIFAR 上要明顯低於 0.8。
- 任一條不成立就停。

---

## 額外問題

1. **2609.15044 主表逐格核對。** 上面兩張表全部核對完。前一輪的 0.454、0.335、0.909 正確。另外要注意兩點：表 1 的 MAGIC 在移除 5% 時是 0.778；文中說 B=50 時 Spell 達到 MAGIC 的 37%（CIFAR）和 39%（TinyStories）。

2. **MAGIC 程式是否公開、單次成本。**
   - 論文沒附程式。官方極簡版 simple-magic-jax 於 2026-10-20 才公開，只有 CIFAR，作者說省略了專用的 fused kernel。完整版本在 Bergson（PyTorch、FSDP、per-token）。
   - 成本：論文寫「3–5 倍訓練」，CIFAR 上一張 A100 約 15 分鐘；ICLR 2026 投稿摘要改寫為「2–3 倍」。計算量 O(T log T)、記憶體 O(log T)。Bergson 另提供 O(T) 計算、O(√T) 記憶體的版本。

3. **MAGIC 能不能當上限。**
   - 移除比例大時不行：CIFAR 上移除 1% 是 0.909，5% 就降到 0.778。MAGIC 自己說線性近似越遠越差，到 20% 明顯下降（只看到圖）。TinyStories 在 5% 時仍有 0.996。
   - 非 metasmooth 的訓練下不行：MAGIC 和 Bergson 都明說這種情況無效或不可靠，需要改訓練流程。
   - 跨種子評測：Bergson 的數字是 0.829，已經和 Unrolled GN（0.819）差不多。
   - **只能當「固定種子、移除少量、平滑訓練」條件下的上限。**

4. **LLM 規模做不做得動。**
   - 單一查詢可行：MAGIC 做過 Gemma-2B LoRA（32 題 MMLU），Bergson 支援數十億參數。
   - 但要建真值（每個查詢一次 replay，加上數百次重訓）就很難。目前做到「k=500 加完整 LDS」的最大設定是 Spell 的 92M、n=2048、256 步。
   - 用 4 張 H200 在 10 週內，實際上限大約是 GPT-2／Pythia-160M 到 410M、幾千筆微調資料。這是推估，未實測。

5. **近兩年 ICML 錄取的資料歸因論文**（依 OpenReview venue 欄位查證，未逐篇讀全文）。
   - ICML 2025 poster：「A Versatile Influence Function for Data Attribution with Non-Decomposable Loss」、「Towards Robust Influence Functions with Flat Validation Minima」。
   - ICML 2026 regular：「On the Fragility of Data Attribution When Learning Is Distributed」、「Step-Resolved Data Attribution for Looped Transformers」、「Nonparametric Data Attribution for Diffusion Models」、「GUDA: Counterfactual Group-wise Training Data Attribution for Diffusion Models via Unlearning」、「Detecting and Filtering Unsafe Training Data via Data Attribution with Denoised Representation」。
   - ICML 2026 spotlight：「Motion Attribution for Video Generation」。
   - 參考：MAGIC 在 OpenReview 標示為「Submitted to ICLR 2026」，通常代表沒被接受，但我沒看到決定頁。改寫版目前是 ICLR 2027 投稿。

---

## 總評

| 風險 | 等級 | 理由 |
|---|---|---|
| 新穎性 | 中偏高 | (b) 已被 Spell 做掉。(a) 沒找到撞題，但這是 Spell、Gunn 和 Bergson 的明顯交集，同組作者正在 ICLR 2027 審查中，很可能延伸 |
| 可行性 | 中 | CIFAR 規模可行（有官方極簡 MAGIC 和 Bergson），但 Spell 要自己重寫。LLM 規模的真值做不起，只能停在約 100M 級 |
| 動機被反駁 | 高 | Unrolled GN 0.819 和 EKFAC+CG 0.685 對 MAGIC 0.931；跨種子下 MAGIC 只剩 0.829；QLD 的下游差距小；MAGIC 的上限只在特定條件下成立 |

**一句話建議：** 有條件做，而且偏向不做。只在一週停損實驗證明兩件事時才繼續：混合估計在 CIFAR 上等計算量明顯勝過 Spell 和 oracle 投影上限，而且 Unrolled GN 在 CIFAR 上沒有接近 MAGIC。否則放棄。

---

## 沒能確認的事項
1. NeurIPS 2026「Efficient and Accurate Evaluation of TDA」：在 arXiv、Semantic Scholar、OpenReview 都找不到，內容未知。
2. MAGIC 的 ICLR 2026 審查意見和決定：OpenReview 的 API 要求人機驗證，讀不到。
3. MAGIC 圖 4 在移除 10–20% 時的確切 LDS，以及 TRAK、EK-FAC 在 CIFAR 上的平均 LDS：只有圖，無法讀出數值。
4. Unrolled Gauss-Newton 在 CIFAR 上的表現、實際成本，以及它是否有獨立論文。
5. Spell 程式是否會釋出。
6. 所有 H200 上的耗時都是從 A100 數字推估的，沒實測。simple-magic-jax 的 CPU 記憶體需求也沒實測。
7. simple-magic-jax 的授權是「Other」，條款沒讀。
8. DATE-LM、2507.14740、2509.26544 的完整數字沒讀全文；「TRAK／EK-FAC 在 LLM 微調上接近隨機」只用 MAGIC 和 Bergson 的數字核對過。
9. Semantic Scholar 對 2609.15044 的引用數是 0，可能是收錄延遲，所以 2026 年 10–11 月的追蹤主要靠 arXiv API 關鍵字搜尋，可能有遺漏。我沒有翻 arxiv.org/list 的完整列表。
10. 2610.24721（Tensor Network Datamodels）、2609.33667（BGU）只讀摘要，判斷和本題不直接重疊。
