# 深查：多保真度資料歸因（便宜分數當 control variate，加少量精確 metagradient）

日期：2026-11-14。沒有用 GPU，也沒有寫實驗程式。網路操作約 30 次：arXiv HTML 4 篇、Semantic Scholar 4 次、OpenReview API 5 次、GitHub API／raw 約 10 次、Bergson 文件 1 頁、網路搜尋 4 次、arXiv API 2 輪。

## 結論

**信心評級：中。** 目前還到不了中偏高。

- **新穎性已確認。** 讀完全文後確定，Spell 沒有以任何形式用到便宜的歸因分數：不當 control variate，不當先驗，也不當 warm start。它的估計值從 0 開始，沒量到的子空間直接當成 0。截至今天，引用者和 ICLR 2027 投稿裡都沒找到多保真度、control variate 或「精確加近似」混合的資料歸因。
- **最大的問題是可提升的空間，不是撞題。** 在 GPT-2／WikiText 上，Unrolled Gauss-Newton（以下簡稱 UGN）已有 LDS 0.819，MAGIC 是 0.931。換成跨種子評測，MAGIC 只剩 0.829，和 UGN 一樣。用下面第 3 節的粗估，若便宜方法本身已經很好，混合法比它高不了多少（約 +0.02～0.05）。只有在便宜方法中等（LDS 約 0.4～0.6）的情境，混合法才有明顯優勢。
- **建議換說法。** 不要把題目寫成「在等算力下贏過 Spell」。Spell 在 GPT-2 上很可能被 UGN 在等算力下壓過，打贏它沒有說服力。改成「**用少量 metagradient 校正並檢核便宜歸因**」，主要賣點有兩個：(1) 用已量測的子空間逐查詢校準便宜分數，並給出無偏的殘差修正；(2) **不需重訓就能逐查詢估計歸因可靠度**。實驗的主戰場要放在便宜方法表現中等的情境。

---

## 閱讀程度

| 文件 | 程度 |
|---|---|
| 2609.15044 Spell／Mage（v2，2026-09-17） | **全文**（第 1–5 節，加附錄 C 的關鍵字抽讀） |
| 2504.16430 MAGIC（arXiv v1） | 全文中的成本、Limitations 和基線段落（前一輪回報 34 讀過全文） |
| 2604.07328 Gunn | 摘要、技術概要、相關研究、變異數界的段落；第 3、4 節證明細節沒讀 |
| 2507.14740 ASTRA | 摘要、方法概要（含 Unrolled 段）、結論 |
| Bergson 排行榜文件頁、`examples/gpt2_wikitext_leaderboard/README.md`、兩個 yaml 設定檔 | 全文 |
| OpenReview：Spell ICLR 2027 投稿、MAGIC ICLR 2027 改寫版、MAGIC ICLR 2026 兩則審查、「Which Influence Are We Estimating?」ICLR 2027 | 搜尋 API 回傳的內容（摘要和審查的 weakness 欄）。討論串和附件頁要求人機驗證，打不開 |

---

## 1. 讀全文後的事實確認

### 1.1 Spell 有沒有用到便宜分數？（全文）沒有。

- **估計式。** 演算法 1 從 Ŷ←0 開始，回傳的 Ŷ = YΠ_U，也就是只取量測子空間 U 上的投影，正交補空間全部當成 0。
- **用到的近似資訊只有兩種：**
  - 最終查詢梯度的 Gram 矩陣 G_V = VᵀV，用來代替未知的殘差 Gram R 以**挑選量測方向**。
  - 由已量測部分估計的列範數，用來加權。
- **warm-up 不是 warm start。** Spell 的「warm-up」指前 a₀ = max(⌈0.1B⌉, 10) 輪改用 Mage 的方向，和便宜分數無關。
- **TRAK、EK-FAC 只出現在引言和參考文獻**，實驗裡沒有。
- **結論的未來工作只提「更好的選擇規則」**，也就是有沒有對 attribution 目標最優的量測方式，沒有提到結合便宜方法。

這對我們有兩個意義：
1. 「補上正交補空間」這個缺口，在 Spell 的結構裡確實存在。
2. G_V 在 Bergson 上對應的「Gradient dot product」只有 LDS 0.156（Bergson 文件，全文）。用 UGN 這類較好的便宜方法來代替 G_V 選方向，是另一個 Spell 沒做的改進點。

### 1.2 UGN 的 LDS 0.819 出處（全文，Bergson 文件）

- **出處：**
  - bergson.readthedocs.io 的 Leaderboard 頁：「Unrolled Gauss-Newton」QLD 0.084 [0.072, 0.096]，LDS 0.819 [0.806, 0.829]。
  - `examples/gpt2_wikitext_leaderboard/README.md` 有同樣的數字，另列逐查詢的分布：中位數 0.829、最小 0.651、最大 0.912，50 個查詢中 50 個 p<.05。
- **評測設定：** GPT-2（124M）在 WikiText 512-token 切塊上微調 4 epoch，共 4,608 個訓練切塊，50 個查詢。100 次「移除 1%」重訓；跨種子 MAGIC 的評測用 seed 43。
- **其他對照：** MAGIC 0.931；MAGIC（跨種子）0.829；EKFAC+CG 0.685。
- **方法說明：** 只對線性層做反向傳播穿過 AdamW 訓練，每一步的 Hessian 換成 Gauss-Newton，**支援一次處理多個查詢**。
- **沒找到的東西：** 獨立論文（README 和文件都沒有連結）；成本或耗時數字。
- **只有 MAGIC 和 LiSSA 有成本數字。** 文件說 MAGIC 在 BF16 或小批次下會明顯退化。LiSSA 在 8 張 A40 上跑了 25.4 小時（全文）。

### 1.3 Spell 程式是否公開

- 論文裡沒有連結，GitHub 也沒找到（同回報 34）。
- **新發現：** OpenReview 上的 ICLR 2027 投稿（forum id jvSDtCTRmv）附有 `supplementary_material` 壓縮檔，很可能就是程式。但下載時要求人機驗證，**內容未確認**。可以在瀏覽器手動下載看看。
- 演算法 1 寫得很完整：Replay、Forward、Extend-Projection、Residual-Gram 都有明確公式，自己重寫的風險低。

### 1.4 其他全文層級的確認

- **MAGIC 的成本（全文）。** CIFAR 上 n=1 時，「essentially 3–5x as much as training a single model (15 minutes on a single A100 GPU)」。Limitations 明列兩項未來工作：(a) 更有效率的 metagradient；(b) 一次處理多個查詢。另外需要 metasmooth 的訓練流程，必要時先預訓練幾百步來取得平滑性。
- **MAGIC 在 ICLR 2026 的審查（OpenReview 搜尋 API）。** Submission16561 有兩則審查，分數 2 和 4，主要批評如下：
  - 與 Bae et al. 2024（SOURCE）、Wang et al. 2025 的軌跡型影響估計重複，未充分比較；
  - 每個查詢要 2–3 倍訓練成本，不實用；
  - 只用 LDS 評測，沒有下游任務；
  - 「MAGIC 解決了資料歸因」是過度宣稱。
  - 改寫版「MAGIC: Data Attribution via Exact Influence」目前是 ICLR 2027 投稿（bpqgWK4w00），摘要和 v1 幾乎相同。
- **Gunn 的關鍵式（全文，技術概要）。** 他的前向模式草圖給出無偏估計，**變異數界約為 |D|^r/k · ‖T^(r)‖_F²**。也就是說，誤差由被估張量的 Frobenius 範數決定。若先減掉便宜估計 C，變異數會變成和 ‖T − C‖_F² 成正比。這就是 control variate 的標準形式，**可以直接寫成我們的理論貢獻**。Gunn 本人沒有這麼做，他的實驗只在 microgpt 上。程式在 github.com/samuel-gunn/microgpt-sketch（原 SamSpo1，**沒有授權檔**）。
- **ASTRA（摘要和方法層級）。** 用 EKFAC 當 Neumann 迭代的預條件子，屬於「便宜近似加速較準的迭代」，是概念上的鄰居，但對象是 iHVP，不是 metagradient 的殘差。寫相關研究時要引用並區隔。

---

## 2. 撞題查核（引用者、ICLR 2027、關鍵字）

| 來源 | 結果 |
|---|---|
| S2 引用 2504.16430（MAGIC） | 24 篇。2026 年 9–11 月有：2609.15044 Spell、2609.33667 BGU、2609.38767 dattri-LLM、2610.15236（無關）、2610.24721 Tensor Network Datamodels、2610.30421「High-Value Training Data from Noisy, Diverse Sources」（用 metagradient 學資料權重，不是歸因估計）、2610.40655 CATT。讀了摘要，**沒有一篇做多保真度、control variate 或混合估計** |
| S2 引用 2609.15044（Spell） | **0 篇**（API 回傳空清單，可能是收錄延遲） |
| S2 引用 2604.07328（Gunn） | 2 篇：Spell、2609.33915「Training Witnesses」（不相關） |
| arXiv API 關鍵字 | 「data attribution」搭配 control variate／multi-fidelity、metagradient 搭配 attribution、「influence matrix」：除了 Spell 和 2610.30421 之外沒有相關結果 |
| 網路搜尋 ×3 | 沒找到「便宜估計加少量精確量測校正」的資料歸因論文。最接近的有三篇（皆搜尋摘要層級）：2604.03858 用影響函數加變異數修正，但目的是檢索；2605.18814 做軌跡型歸因的誤差分析；AirRep 2505.18513 用實測子集效果訓練表徵，屬於「把昂貴的蒸餾成便宜的」 |
| OpenReview ICLR 2027 | 找到 Spell（jvSDtCTRmv）、MAGIC 改寫版（bpqgWK4w00）、「Which Influence Are We Estimating?」（1GmrN1BNkD）、「Optimizing Training Data via Metagradient Descent」（GLXMDkCqeP）。**沒有混合或多保真度的投稿**。注意搜尋 API 只回前 25 筆，不保證涵蓋全部投稿 |
| Bergson 文件 | TrackStar 的說明寫「presumably designed to be used as the first stage of a data search pipeline that eventually uses a more powerful influence function to re-rank the top-k」。這是**兩段式串接**（先粗篩再重排），不是 control variate，但審稿人可能拿來比 |

**判斷：** 撞題風險是低到中。主要風險不是現在有人做了，而是同組作者（Ilyas、Spell 團隊）或 EleutherAI（Bergson 同時有 UGN 和 MAGIC）在 1 月前順手做掉。Bergson 在 2026-11-13 還有推送，很活躍。

---

## 3. 最強的反駁與回應

### 反駁 A（最強）：UGN 已經夠好，混合法頂多加一點，Spell 是稻草人

**論點：**
- 在 GPT-2／WikiText，UGN 0.819、MAGIC 0.931；跨種子評測下 MAGIC 掉到 0.829，和 UGN 一樣。
- UGN 能一次處理多個查詢，成本很可能接近一次 replay（**推估**，Bergson 沒公布耗時）。
- Spell 在 TinyStories 上 B=0.2k 時也只有 0.545（全文，表 2，k=500，B=100）。

所以等算力下，主要對手是 UGN，不是 Spell。

**粗估（推估，我自己的推導，沒有實測）：**
- Spell 的式 (2) 用 Pearson 相關近似 LDS，因此 LDS 約等於 cos(Ŷ_q, Y_q)（全文推導）。
- 混合估計 Ŷ_q = Y_qΠ_U + α_q C_q(I − Π_U)。設 U 捕捉的正規化能量比例為 κ，便宜分數在正交補上與 Y 的相關為 ρ，則 cos² ≈ κ + (1 − κ)ρ²。
- Spell 在 B=0.2k 時 LDS 約 0.45，對應 κ ≈ 0.2。

代入不同的 ρ：

| 便宜方法的品質 | ρ | 混合法 | 比 Spell | 比便宜方法 |
|---|---|---|---|---|
| 中等（EK-FAC 類） | 0.5 | 約 0.63 | +0.18 | +0.13 |
| 很好（UGN 在 GPT-2，對 MAGIC 正規化後） | 0.88 | 約 0.90 | — | **約 +0.02** |

**回應：**
1. **主戰場放在便宜方法中等的情境。**
   - CIFAR ResNet-9：目前**沒有任何公開的 UGN 數字**。Bergson 的 gnh-unroll「只考慮線性層」，能不能處理 conv 層**未確認**，處理不了就只剩 EK-FAC 和 TRAK。
   - BF16、小批次、長訓練：Bergson 文件說 MAGIC 在 BF16 和小批次下都會退化，UGN 在這些條件下的表現沒人量過。
2. **逐查詢的最差情況。** UGN 最低的查詢只有 0.651（全文，README），混合法可以補救這種壞查詢。
3. **把賣點從平均 LDS 移到可靠度估計。** 已量測子空間能逐查詢估出 ρ_q，也就是不用重訓就知道每個查詢的便宜分數準不準。UGN 和 Spell 都給不出這個資訊。
4. **用 UGN 取代 G_V 挑選量測方向。** G_V 對應的 gradient dot product 只有 0.156，UGN 的 Gram 理應是更好的代理。這是混合法的第二個組件。

### 反駁 B：高維下 control variate 的相關性太低

**論點：** 即使 LDS 不差，殘差在不同查詢間可能沒有共享的低秩結構，B 個量測捕捉的殘差能量 κ_res 可能比 κ 更小。

**回應：**
- 估計式在結構上不會比便宜方法差：在量測子空間上換成精確值，正交補維持 α_q·C。只有 α_q 估錯時才會變差，而 α_q 由量測子空間迴歸得到，且可以收縮。
- 同理也不會比 Spell 差，因為 α=0 就退化成 Spell。
- 這是可驗證的實證問題：停損實驗直接量 κ_res 對 κ。附錄 C 的事實可當參考：CIFAR 上最大的四分之一查詢佔 99.2% 能量。殘差是否也這麼集中**未知**。

### 反駁 C：LDS 指標和精確上限本身受質疑

**論點：**
- MAGIC 在 ICLR 2026 的審查人批評「只用 LDS」「過度宣稱」，給了 2 分和 4 分。
- ICLR 2027 投稿「Which Influence Are We Estimating?」主張，不同估計量的排序差異主要來自估計對象的設定不同（specification mismatch），而不只是近似誤差。
- 這對本題有直接衝擊：EK-FAC 估的是收斂點的影響函數，MAGIC 估的是固定軌跡的 metagradient，兩者不是同一個量，ρ 可能先天偏低。

**回應：**
- 選 UGN、SOURCE 這類同樣是軌跡型的估計當 control variate，估計對象一致。
- 「便宜方法估的對象不同，所以 ρ 低」本身就是混合法可以量化的東西，可寫成一節分析。
- 下游指標加上 Bergson 的 QLD（移除前 1% 資料後查詢損失的增幅），回應「只用 LDS」的批評。
- 跨種子 LDS 必須報告，不能迴避。

### 反駁 D：同組作者很快會自己做

**回應：** 這無法完全排除，只能靠速度和角度區隔：Gunn 式無偏草圖的變異數界、逐查詢可靠度、BF16 和小批次等 UGN 弱的情境。前兩週的停損實驗做完，就決定是否繼續。

---

## 4. 可行性

### 程式與授權

| 資產 | 狀態 | 授權 |
|---|---|---|
| EleutherAI/bergson（MAGIC、gnh-unroll、EKFAC±CG、TRAK、ASTRA、LDS 重訓庫、排行榜 yaml） | 公開，2026-11-13 仍有更新 | **MIT** |
| andrewilyas/simple-magic-jax（官方極簡 MAGIC，只有 CIFAR，README 說不調環境變數也有 LDS >0.9） | 公開，2026-10-20 建立 | **CC BY-NC-SA 4.0**（學術使用可以；改寫版要同授權並標註，不可商用） |
| MadryLab/trak | 公開 | MIT |
| pomonam/kronfluence（EK-FAC） | 公開 | Apache-2.0 |
| samuel-gunn/microgpt-sketch（Gunn 草圖） | 公開 | **沒有授權檔**：可以參考，但不要直接拷貝 |
| Spell | 論文裡沒有；ICLR 2027 投稿附有壓縮檔，但內容未確認 | — |

### GPU 小時（括號內標來源層級）

**CIFAR-10 ResNet-9：**
- 一次 MAGIC replay 在 A100 上約 15 分鐘，是訓練的 3–5 倍（全文）。
- 換到 H200 約 7–10 分鐘，一次訓練約 2–4 分鐘（推估）。
- 建議規模：k=200 個查詢的精確 Y，約 200 × 8 分鐘 ≈ 27 H200 小時；M=200 次重訓約 10 小時；便宜方法加混合法（B ≤ 40，每次 replay 外加一次前向模式）約 5–8 小時。
- **合計約 45 H200 小時**（推估）。用 k=100 可以減半。

**GPT-2／WikiText（Bergson 排行榜設定）：**
- 規模：4,608 個切塊 × 512 token × 4 epoch，批次 512，約 144 步（由 yaml 推算）；50 個查詢、100 次重訓（全文）。
- 原設定用 8 張 GPU 跑（yaml 的 nproc_per_node: 8），沒有耗時數字。
- 以 124M 模型、約 940 萬訓練 token 推估：一次訓練幾分鐘，一次 replay 約十幾到 30 分鐘。50 次 replay 加 100 次重訓，**約 15–30 H200 小時**（推估）。
- 若要做 k=200，replay 部分約 ×4。
- LiSSA 是唯一有實測數字的：25.4 小時 × 8 張 A40（全文），但我們不需要重跑它。

**合計：** 兩個設定的停損實驗約 60–80 H200 小時（推估），4 張卡約 1 週，在預算內。

---

## 5. 信心評級與理由

**中。**

| 面向 | 評估 |
|---|---|
| 新穎性 | 中偏高。全文確認 Spell 沒有用便宜分數；引用者和 ICLR 2027 投稿都沒有撞題。扣分在於這是 Spell、Gunn、Bergson 三者的明顯交集，同組可能隨時跟進 |
| 動機 | 中偏低。在 GPT-2 的單模型評測下，UGN 讓可提升的空間只剩約 0.11（0.819 對 0.931），跨種子評測下是 0 |
| 可行性 | 中偏高。程式公開、授權可用、成本在預算內；Spell 要自己重寫，但演算法很清楚 |
| 正面貢獻 | 有。方法、無偏草圖的變異數界、逐查詢可靠度三者都屬於正面貢獻，不是純負面分析 |

**為什麼沒到中偏高：** 在最常用的 LLM 基準上，關鍵對手 UGN 的空間太小。CIFAR 上便宜方法的實際水準，以及殘差有沒有低秩結構，都還沒有數字。

---

## 6. 要把信心推到中偏高，切入角度怎麼調

1. **重新定位。** 題目改為「Calibrating Approximate Data Attribution with a Few Metagradients」之類。對手是「最好的便宜方法」（UGN、EKFAC+CG），Spell 只當 B 很小時的參照。等算力圖（LDS 對 GPU 小時）仍是第一張圖，但要把 UGN 放進去。
2. **三個組件，各自都要能獨立成立：**
   - (a) 混合估計：Y_qΠ_U 加上 α_q C_q(I − Π_U)，α_q 由量測子空間做迴歸並收縮。
   - (b) 用便宜分數的 Gram 取代 G_V 來挑選量測方向，延伸 Spell 的 Spell-Select。
   - (c) 逐查詢可靠度估計：從量測子空間估出的 ρ_q 預測真實的逐查詢 LDS，並用它把剩下的預算分給最不可靠的查詢。
3. **理論。** 套用 Gunn 的無偏草圖：變異數界從 ‖Y‖_F² 改成 ‖Y − αC‖_F²，並推出所需量測數隨 (1 − ρ²) 縮小。這一步可以在 1–2 週內寫成命題。
4. **主戰場選便宜方法中等的情境：** CIFAR ResNet-9（若 UGN 不支援 conv，就和 EK-FAC、TRAK 比）、BF16 或小批次訓練的 GPT-2、較長的微調。Bergson 標準設定當「便宜方法已經很好時混合法也不會變差」的對照。
5. **報告要涵蓋：** 跨種子 LDS、移除 5% 的 LDS、QLD，直接回應 MAGIC 審查人的批評。

---

## 7. 停損判準（一週內、≤80 H200 小時）

**繼續的條件（三條都要成立）：**
1. CIFAR k=100～200、B ≤ 0.2k 時，混合法的 LDS 比 Spell 和最佳便宜方法都高至少 0.08，且高過 Spell 論文的 Spherical SVD oracle（B=0.2k 時 0.519）。
2. GPT-2／WikiText 上，混合法比 UGN 高至少 0.03，95% 信賴區間不重疊；**或者**逐查詢可靠度估計和真實逐查詢 LDS 的 Spearman 相關至少 0.6（兩者擇一成立即可）。
3. 殘差 Y − αC 被 B 個量測捕捉的正規化能量比例 κ_res，不小於 Y 本身的 κ 的一半。

**直接停的條件（任一成立）：**
- CIFAR 上最佳便宜方法已有 0.8 以上。
- 混合法在 GPT-2 上和 UGN 的差距落在信賴區間內，**而且**可靠度估計的相關低於 0.4。
- 1 月前有人公開了同樣的混合法（建議每週查一次 Spell 和 Bergson 的引用與更新）。

---

## 沒能確認的事項

1. Spell 的 ICLR 2027 附件內容（OpenReview 要求人機驗證）。
2. UGN 的耗時、在 CIFAR 或 conv 網路上能不能用、有沒有獨立論文。
3. Spell 在 S2 上的引用者（API 回 0 筆，可能是收錄延遲）。
4. ICLR 2027 投稿的完整清單（搜尋 API 只回前 25 筆）。
5. 所有 H200 耗時都是從 A100 數字推估的。
6. 第 3 節的混合法增益是我自己用 Spell 式 (2) 的近似推出來的，沒有實測。
7. 一次網路搜尋顯示 2609.15044 的 arXiv 摘要頁標題是「Data Attribution at Scale via Influence Matrix Estimation」，和 v2 HTML 的標題不同，可能是 v1 的舊標題，沒有逐版本核對。
