協調者回報：7 組 Opus 子代理（general-purpose）已全部完成，以下依序是各組回報原文，未做摘要或改寫。

## 第 1 組：ICML 錄取分布與題型

## 範圍 1 回報：ICML 錄取分布、題型，以及 10 個小算力可做的未解問題

### 摘要（3–5 句）
ICML 2026 收到 23,918 篇投稿，是 2025 年 12,107 篇的兩倍。錄取 6,352 篇（26.6%），其中 oral 168 篇（0.7%）。這三個數字來自 ai2.work 與 WASP 的新聞稿，不是官方統計頁，我沒有對到官方來源。投稿翻倍主要來自 LLM：在官方會場資料裡，「Deep Learning→Large Language Models」這個子領域單獨就有 1,114 篇（16.8%），是最擁擠的一塊，但它拿到 oral 或 spotlight 的比例（8.0%）只在平均附近。比例最高的是可解釋性（14.3%）、Foundation Models（15.3%）、Alignment（14.7%）、深度學習理論（14.3%）、學習理論（13.2%）和因果（13.3%）這幾個子領域。電腦視覺應用只有 5.8%，偏低。

---

### 一、錄取分布（自己從官方會場資料計算）
**資料來源**：ICML 官方虛擬會場的公開資料 `icml.cc/static/virtual/data/icml-2026-orals-posters.json` 與 2025 年同名檔案。已下載到 `/tmp/claude-1047/-work-ChihChungHsu-cactus8603-icml27-s2-gated-sink-kv/bd65f9a7-961a-4b8d-adce-42df28fbb74b/scratchpad/dl/`，計算腳本是同目錄上一層的 `stats.py` 和 `kw.py`。

**計數方式與注意事項**：
- 以 poster 條目為準，依論文標題去重，2026 年得到 6,628 筆。這比官方的 6,352 篇多，推測多出來的是 position track 等其他條目。
- 2026 年有 568 筆（8.6%）沒有填主領域。
- 2026 年 oral 的 decision 欄位寫的是「Accept (spotlight)」，所以我用 Oral 場次的條目來辨識 oral。
- 照這個算法，非 oral 的 spotlight 有 421 篇，和新聞稿說的「536 spotlight」對不起來，原因未確認。

**2026 年各大領域**（篇數、占比、oral 率、spotlight 率，以及兩者合計）：
| 領域 | 篇數 | 占比 | oral | spotlight | 合計 |
|---|---|---|---|---|---|
| Deep Learning | 2241 | 33.8% | 2.2% | 6.4% | 8.6% |
| Applications | 1374 | 20.7% | 2.0% | 5.2% | 7.1% |
| General ML | 674 | 10.2% | 1.8% | 7.1% | 8.9% |
| Social Aspects | 643 | 9.7% | 3.1% | 8.1% | 11.2% |
| （未標註） | 568 | 8.6% | 3.0% | 6.0% | 9.0% |
| Reinforcement Learning | 376 | 5.7% | 1.3% | 7.7% | 9.0% |
| Theory | 373 | 5.6% | 3.5% | 6.7% | 10.2% |
| Optimization | 212 | 3.2% | 1.9% | 4.2% | 6.1% |
| Probabilistic Methods | 167 | 2.5% | 3.0% | 6.0% | 9.0% |

**2025 年對照**（3,339 筆，合計 = oral + spotlight 比例）：
- Deep Learning 33.9%，合計 10.0%
- Applications 17.8%，合計 8.9%
- General ML 14.1%，合計 10.2%
- Theory 9.9%，其中 oral 6.4%，合計 12.1%
- Social 8.3%，合計 8.7%
- RL 6.1%，合計 8.3%
- Optimization 4.7%，合計 5.8%
- Probabilistic 3.0%，合計 17.2%
- Position 2.2%，合計 31.9%

**從 2025 到 2026 的變化**：
- Theory 的占比從 9.9% 降到 5.6%，oral 率從 6.4% 降到 3.5%。
- Social Aspects（安全、可解釋性、對齊）從 8.3% 升到 9.7%。
- 可解釋性子領域的 oral＋spotlight 比例從 2.6% 升到 14.3%。

**2026 年主要子領域**（篇數、占比、oral＋spotlight 比例）：
- LLM：1114 篇，16.8%，8.0%
- 電腦視覺：467 篇，7.0%，5.8%
- 生成模型：323 篇，4.9%，7.4%
- 化學／物理／地球科學：192 篇，2.9%，8.9%
- 醫療：170 篇，5.9%
- 可解釋性：168 篇，14.3%
- 機器人：150 篇，12.0%
- 安全：149 篇，10.7%
- Foundation Models：131 篇，15.3%
- GNN：123 篇，11.4%
- 深度學習理論：98 篇，14.3%
- 時間序列：95 篇，4.2%
- 注意力機制：80 篇，3.8%
- 穩健性：72 篇，2.8%

**標題關鍵字**（2025 → 2026，占全部論文的比例）：
- LLM：14.8% → 17.0%
- reasoning：3.1% → 7.4%
- agent：3.3% → 6.9%
- RL／RLHF／GRPO：5.1% → 7.5%
- diffusion／flow：6.6% → 7.1%
- 理論字眼（provable、bound 等）：5.2% → 4.0%
- graph：6.2% → 4.2%

**2026 年獲獎論文**（來源：blog.icml.cc/?p=1347）：
- Outstanding Paper：The Flexibility Trap（arXiv 2601.15165）、High-Accuracy Sampling for Diffusion Models and Log-Concave Distributions。
- Honorable Mention：Obfuscation Atlas（2602.15515）、Motion Attribution for Video Generation、How much can language models memorize?、Random Matrix Theory Perspective on Diffusion Consistency（2602.02908）、To Grok Grokking（2601.19791）。

---

### 二、哪些領域的重點論文是小算力做出來的
**抽查方式**：我抽查了 12 篇 2026 年 oral 或獲獎論文，讀全文中寫到的硬體。這不是全面統計，只是抽樣。

**單機或不需要 GPU 叢集**：
- 理論加小型實驗：Focus and Dilution（2605.01199，單層 attention，TinyStories／WikiText）、Non-Euclidean EoS（2603.05002，MLP／CNN／小型 Transformer）、Hybrid 模型表達力（2603.08859，合成任務）、To Grok Grokking（2601.19791，線性模型）、Random Matrix 看 diffusion（2602.02908，UNet／DiT 的線性預測驗證）。
- 現成模型分析：Compositional Generalization（2602.24264，只對 CLIP／SigLIP／DINO 做推論分析）、Context 等同參數更新（2511.17864，Gemma 3 1B／Falcon 7B 推論）、How Many Outputs（2605.22223，Pythia 160M–1B，論文寫用 2 張 H100）。

**中等算力（4–8 張 GPU）**：
- Do We Need Adam（2602.07729）：4 張 GH200，Qwen3-1.7B 做 RLVR。
- On-policy distillation 機制類（2608.31046）：8 張 H100 或 H200，模型 ≤9B。
- Direct-OPD（2607.05394）：8 張 A100 跑 4 小時。

**算力較大**：
- Flexibility Trap（獲獎）：16 張 H100。
- Midtraining（2510.14865）：從頭預訓練 128B tokens，用 L40S 叢集，卡數我沒有看到。
- Procedural Pretraining（2601.21725）：需要預訓練小模型，最大到幾 B 參數。確切數字在 HTML 裡是公式，讀不到。

**結論**：可以在單機做的 oral 集中在四類：深度學習理論與訓練動態、最佳化理論、表徵幾何與可解釋性，以及「用現成模型推論做分析」的論文。LLM 後訓練（RL 或蒸餾）的機制研究，用 4–8 張卡、≤9B 模型也能做到 oral 等級，但這一塊在 2026 下半年非常擁擠。

---

### 三、10 個作者明確寫為未解或 future work、而且小算力可做的問題
每一題我都打開 arXiv HTML 全文讀了原文。這些是作者明寫的方向，不是我推論出來的。我**沒有**對它們做撞題搜尋。

1. **2605.01199 Focus and Dilution**（ICML'26 oral）
   - 原文位置：「Limitation」節。
   - 未解問題：focus–dilution 循環只在單層、單頭的 attention 上證明過。作者原文寫：「Extending the analysis to multi-layer and multi-head architectures, and studying how common components such as LayerNorm affect the dynamics, are important directions for future work.」
   - 小算力理由：原論文實驗只用合成 Markov 資料和 TinyStories。

2. **2603.05002 Non-Euclidean GD at Edge of Stability**（oral）
   - 原文位置：§6「Conclusion and Future Work」。
   - 作者列出的開放問題：(a) 尚未證明在一般範數下，任意初始化都會發散。(b) 有一段「pre-EoS oscillatory regime」，也就是方向平滑度先上升、廣義 sharpness 才到門檻；作者稱理解它是「important direction」。(c) 要推廣到隨機、帶動量和自適應的最佳化器，例如真正的 Muon。
   - 小算力理由：小型網路、full-batch 訓練。

3. **2601.21725 Procedural Pretraining**（oral）
   - 原文位置：§7 Discussion 的「Why is procedural data helpful?」段。
   - 未解問題：作者的實驗排除了「只是最佳化變好」或「只是記憶」這兩種簡單解釋。原文說「Investigating a first-principles explanation or studying the mechanisms … using mechanistic interpretability techniques is a promising avenue」。同段也提到，或許能把程序資料化簡成「closed-form smart initialisation」。
   - 小算力理由：小模型預訓練，0.1% 的程序資料就有效果。

4. **2602.24264 Compositional Generalization Requires Linear, Orthogonal Representations**（oral）
   - 原文位置：§6「Limitations and future work」。
   - 未解問題：(a) 把最壞情況的穩定性放寬成平均或近似的穩定性。(b) 訓練資料的覆蓋範圍怎麼改變表徵的必要幾何結構。(c) 理論假設編碼器固定、可以換資料重訓，但實務上只訓練一次；這個假設對結論的影響未知。
   - 小算力理由：只需要對 CLIP、SigLIP、DINO 做推論。

5. **2603.08859 Expressivity-Efficiency Tradeoffs for Hybrid Sequence Models**（oral）
   - 原文位置：§6 Conclusion。
   - 未解問題：目前只用合成任務和受限的 attention。作者把以下列為「important direction for future work」：「extending the theory to broader attention patterns, external memory, identification of real function-composition datasets, and naturalistic long-context workloads」。
   - 小算力理由：合成任務加小型 hybrid 模型。

6. **2605.22223 How Many Different Outputs Can a Transformer Generate?**（oral）
   - 原文位置：§7 Future Work。
   - 未解問題：作者寫「we leave an empirical evaluation of Theorem 4.10, as well as the derivation of theoretical lower bounds, to future work」，也希望推廣到 Mamba 等 SSM，以及 test-time training（Titans 的 memory-as-context）需要多少記憶向量。
   - 小算力理由：原論文用 2 張 H100、Pythia 160M–1B。

7. **2511.17864 Equivalence of Context and Parameter Updates**（oral）
   - 原文位置：§6.1 Limitations 與 §6.2 Future Work。
   - 未解問題：目前導出的 rank-1 權重修補只對單一 token 成立，延伸到自由生成時，新生成 token 的 attention 會破壞等價。作者原文是「extending our single-token equivalence to unconstrained free generation requires aggregating these transient updates into reusable thought patches」，也提到自動化的「context compiler」。
   - 注意：已經有同期工作 Mazzawi et al. 在做「thought patch」，撞題風險高。
   - 小算力理由：Gemma 3 1B 推論。

8. **2608.31046 Does On-Policy Distillation Really Distill?**（2026 年 8 月，HF 熱門，84 讚）
   - 原文位置：Appendix A「Limitations and Future Directions」。
   - 未解問題：不需要老師、只給低機率 token 固定的負 advantage，就能重現 OPD 大部分的增益。作者原文是「calls for a reexamination of the mechanism underlying OPD with the K1 estimator and motivates further theoretical analysis to identify the true source of OPD's performance gains」。
   - 小算力理由：8 張 H100 或 H200，模型 ≤9B。

9. **2609.04172 Rethinking On-Policy Distillation II: One Training Example**（2026 年 9 月，81 讚）
   - 原文位置：§9「Limitations」與「Future work」。
   - 未解問題：作者原文是「What sets the absorption rate is still open」。absorption rate 是每次更新能吸收的師生差距比例，用 1 題和用 1.7 萬題訓練時下降得一樣快。另一個方向是只從題目本身估計「state coverage」，把它當選資料的準則。
   - 小算力理由：one-shot 設定，資料量極小。

10. **2610.11765 Thinking Inertia: LLMs Keep Thinking When Told Not To**（2026 年 10 月，126 讚）
    - 原文位置：§5.1 與 §5.2。
    - 未解問題：論文只看可見的輸出。作者寫：「internal-state analyses can examine whether computation disappears, moves to a hidden channel, or is only omitted from view」，也提出要訓練「calibrated stopping」，而不是一律壓抑推理。
    - 小算力理由：本地的 Qwen3 模型推論加 probing。

**備選（未列入前 10）**：
- 2602.07729 Do We Need Adam：SGD 只更新不到 0.02% 的參數，作者在 Impact Statement 說這「could … help explain why RL-trained models exhibit reduced catastrophic forgetting」。這只是附帶的一句，不是正式的 future work。
- 2609.29845 Linear Superposition：§8 寫多模態「remain completely unexplored」。但它的賣點跟 KV-cache 有關，接近已排除的方向，所以不建議。

**擁擠度提醒**：第 8、9 題所在的 on-policy distillation 是 2026 下半年最熱的方向。光是同類論文我就看到 2607.05394、2608.06296、2608.31046、2609.04172、2609.36484 五篇，追的人多，撞題風險高。第 1、2、4、5、6 題屬於理論或表徵幾何，擁擠度較低。在會場資料裡，這幾題所屬的子領域（深度學習理論、學習理論、可解釋性）拿到 oral 或 spotlight 的比例也較高，在 13–14% 左右。

---

### 沒能確認的事項
- 官方正式的 2026 統計頁（投稿 23,918、錄取 6,352、oral 168、spotlight 536）：只來自二手新聞稿。
- 我算出的非 oral spotlight 421 篇，和新聞稿的 536 篇對不上，原因不明。
- 2025 年的官方錄取數：我只用會場資料計數（3,339 筆，可能含 position track），沒有核對官方數字。
- 會場資料中 2026 年有 568 筆沒有標主領域，可能讓各領域的比例偏低。
- 「單機做出重點論文」只抽查了 12 篇，不是全體統計。Procedural Pretraining 和 Midtraining 的確切 GPU 數沒讀到。
- 「Information Flow Reveals When to Trust LMs」（oral）找不到 arXiv 版本。
- 搜尋引擎對 2026 年 9–11 月的 arXiv 收錄很少。「熱門」改用 Hugging Face daily papers 的讚數（2026/7/1–11/13，共 3,808 篇）當代理指標，它偏向 LLM 和 agent，不代表 ICML 的取向。
- 上面 10 個問題都沒有做撞題搜尋，進入候選前需要另外查核。

**資料來源**：
- icml.cc 虛擬會場 JSON
- blog.icml.cc/?p=1347
- ai2.work/blog/icml-2026-hits-record-23-918-submissions-as-agents-take-over
- wasp-sweden.org/strong-wasp-presence-at-icml-2026/
- 各篇論文的 arxiv.org/html/<編號>
- huggingface.co/api/daily_papers

---

## 第 2 組：LLM 內部機制與可解釋性

## 範圍 2：LLM 內部機制與可解釋性，回報

## 領域份量與擁擠程度
- ICML 2026 總投稿 23,918 篇，錄取 6,352 篇，錄取率 26.6%。投稿數出自大學新聞頁；錄取數與錄取率只有二手來源 ai2.work 部落格。ICML 2026 有正式的 Mechanistic Interpretability workshop（ICML 官方 blog，2026/04/06）。
- 可解釋性與機制分析在主會的占比，我沒找到官方數字。估計約占錄取論文的 3–6%，這只是推測。
- 擁擠程度：SAE／transcoder 方法與評測非常擠，光 2026 就有多篇「SAE 打不贏隨機基線」的 sanity check（2501.17727v2，ICLR 2026；2602.14111）。算術機制（helix 與 heuristics 兩派）也已有多篇跟進。相對空的是「訓練動態 × 機制」的交界：公開的中間 checkpoint 很多，但真正做因果介入的很少。

## 候選 1（唯一合格候選）

**一句話主張**：預訓練中的「mode-hopping」（相鄰 checkpoint 之間，泛化行為忽然消失又恢復）來自兩條在局部、可定位的電路互相競爭，勝負由最近一段訓練資料的組成決定，而不是優化雜訊。這可以否證：在 hop 發生前的 checkpoint 載入優化器狀態後重播訓練，只把該資料窗打亂或過濾，若 hop 照樣出現，主張就被推翻。

**對立說法與出處**
- 優化雜訊說：2510.04848（Instability in Downstream Task Performance During LLM Pretraining，EMNLP 2025 Findings）認為下游分數的劇烈起伏可用平均或集成相鄰 checkpoint 壓下去，沒有指出成因。
- 資料窗與電路競爭假說：2609.33150（Generalization Dynamics of LM Pre-training，2026/09/27）發現 OLMo3-32B 的 answer+1 正確率是 81% → 0% → 81.7%（2.17T / 2.19T / 2.21T tokens），平均 5 個 checkpoint 只能減輕、不能消除。作者把電路競爭與資料窗決定勝負列為「假說」，明說實驗只看到行為，沒找出電路。
- 優化動力學說：NeurIPS 2025 的 Li et al.（Tracing the Representation Geometry of LMs）也看到 OLMo／Pythia 表徵幾何的非單調變化，但歸因於交叉熵優化本身。
- 玩具模型的競爭說：2503.05631（Strategy Coopetition，ICML 2025）在玩具設定中以容量競爭解釋 ICL 的出現與消退，沒有在真實 LLM 上驗證。

**最接近的 5 篇與差別**
1. 2609.33150（2026/09）：發現 mode-hopping，只做行為測量。有載入 Adam 狀態的單步穩定性測試，以及初步的資料子集續訓，但沒有資料窗重播或打亂的因果對照，也沒有電路層級分析。本題正是把它的假說拿去做因果與機制檢驗。
2. 2510.04848（2025/10）：只描述不穩定，緩解方法是 checkpoint 平均或集成，沒有成因分析。
3. 2611.16882（2026/11/12，Pretraining Data Statistics Shape the Phases of Learning Entity Comparison）：最接近的方法論。它用 163M 模型從零訓練，證明資料統計因果決定「頻率 → 位置 → 正確」三個學習階段，並在 OLMo-2 32B 觀察到同樣的階段。但它談的是單調的階段演進，不是可逆的跳動，任務只有實體比較，也沒有在大模型 checkpoint 上做重播介入。
4. 2606.02378（2026/06，When Do Attention Circuits Form?）：追蹤 Pythia／OLMo 1B 的 induction 與 previous-token 電路形成時點，指出「能力穩定，但電路實作不穩定」。它不談能力消失後再恢復，也不做資料介入。
5. 2608.13515（2026/08，COLM 2026，Task-Agnostic Training Data Influence Across Pretraining）：在 Pythia／PolyPythia 上算不綁任務的資料影響力，看到的是漸進的資料影響力交替，沒有連到 checkpoint 之間的突變。本題可以借用它的工具。
- 補充：2503.05631（Strategy Coopetition，ICML 2025）只做玩具模型。

**停損實驗**（約 1 週，約 30–45 GPU 小時，全部是推論）
- 第一步，在可以重播的模型上重現 hop：用公開的 GDsuite（github Jiaxin-Wen/GDsuite）掃 OLMo3-7B 連續約 60 個 checkpoint，和 Pythia-6.9B／12B 的 143 個 checkpoint。Pythia 的每個 batch 內容都可以重建。約 20–25 GPU 小時。
- 第二步，挑 1–2 個 hop，對 hop 前後兩個 checkpoint 做 attribution patching 或 activation patching，看翻轉能不能定位到少數元件。
- 判準：以下兩條都達到才繼續，否則停。
  - (a) 7B 級模型至少一個任務有可逆 hop，且 soft margin 的振幅大於種子間標準差的 3 倍。
  - (b) 不超過 5% 的 attention head 或 MLP 元件，就能還原 hop 前後 logit 差的 70% 以上。
- 公開資產：OLMo3 的 checkpoint 與訓練資料、Pythia 的 checkpoint 與 batch_viewer、PolyPythias（9 個種子，各自決定初始化與 batch 組成；種子 2 的資料索引可能缺失）、TransformerLens 或 nnsight。

**全套實驗估計**（約 1,500–2,500 H200 小時，4 張卡 10 週的上限是 6,720 小時，可行）
1. 因果重播：在 hop 前的 checkpoint 載入優化器狀態，對同一資料窗做四種處理：原序、窗內打亂、換成別的窗、濾掉支持淺層捷徑的樣本，再加上只換種子不換資料的對照。重播時每 50 步評估一次，先把 hop 定位到子窗，只重播那一段。7B 每 40 億 tokens 約 120 GPU 小時，總計約 1,000–1,500 GPU 小時。
2. 在 PolyPythias（410M 以下）上，用「同資料、不同種子」對「不同資料」區分初始化與資料的作用。前提是小模型也有 hop，這點未確認。
3. 用資料影響力，或窗內 n-gram／捷徑頻率，預測 hop 的方向，並在保留的 hop 上測試。
4. 電路層級描述：說清楚哪兩條電路在競爭，以及兩者的讀出權重範數消長能不能提前預告 hop。

**為何至今沒人做**：這個現象 2026 年 5 月才以部落格形式出現，9 月底才上 arXiv。要做因果重播，必須同時有優化器狀態、確切的資料順序、7B 級中間 checkpoint，這三樣湊齊是最近的事（OLMo3）。

**主要風險**
- 原作者說小模型（低於 7B）不太跳，那就無法用 Pythia-410M 或 PolyPythias 便宜重播，只能做 7B，算力吃緊。
- OLMo3 的 checkpoint 間距與優化器狀態是否全面公開，未確認。原作者有載入 Adam 狀態，但公開範圍不明。
- 原作者團隊可能很快自己做跟進，搶題的時間窗短。
- 結果可能是「混合成因」（資料窗與雜訊各占一部分），這樣故事會比較弱，但仍可發表。
- 2610.02098 指出介入式 faithfulness 分數可能把電路排序排錯，電路比較方法要謹慎。

**信心：中**
- 有利：對立說法明確，成本可控，資產公開，5 篇近作都沒有做因果重播。
- 不利：能不能在可負擔的模型尺度重現 hop，要等停損實驗才知道；搶題風險高。

## 沒有列入的方向
- SAE 和隨機模型基線：2026 年已有 ICLR 2026 版的 2501.17727v2 與 2602.14111，太擠。
- 算術機制（helix 對 bag of heuristics）：2506.07824、2508.02513、2607.16693 都在嘗試調和兩派；而且只關於加法，推廣性不足。
- ICL 的 FV head 對 induction head、跨 checkpoint 的 SAE 特徵演化：記得已有多篇，這輪沒有逐篇開啟查證，所以沒有提。

## 沒能確認的事項
- ICML 2026 中可解釋性主題的實際篇數與占比。錄取數與錄取率只來自二手來源 ai2.work。
- OLMo3-7B 的 checkpoint 間距、每個資料窗的 token 數、優化器狀態是否對所有 checkpoint 公開。上面的 GPU 小時估計假設每窗約 40 億 tokens、MFU 約 40%。
- Pythia 或 OLMo-2 7B／1B 有沒有 mode-hopping，原論文沒測。
- Pythia 帶優化器狀態的 NeoX checkpoint 是否完整公開，我印象中有，但沒查證。
- PolyPythias 種子 2 的資料索引是否已補上（2608.13515 寫說缺失）。
- 2610.30884、2605.24059 及其系列論文，只看到搜尋摘要，沒有開全文。
- 2606.02378、2608.13515、2611.16882 只讀了摘要頁，沒讀全文。2609.33150 讀了 arXiv 摘要頁與作者部落格，沒讀 PDF 全文。這些論文的相關工作清單都沒看，可能還有更接近的論文。

來源：
- arxiv.org/abs/2609.33150
- jiaxin-wen.github.io/blog/generalization-dynamics
- arxiv.org/abs/2510.04848
- arxiv.org/abs/2611.16882
- arxiv.org/pdf/2606.02378
- arxiv.org/abs/2608.13515
- arxiv.org/abs/2503.05631
- arxiv.org/abs/2503.09543
- arxiv.org/abs/2509.23383
- arxiv.org/html/2602.14111
- arxiv.org/abs/2501.17727
- arxiv.org/abs/2502.00873
- arxiv.org/abs/2410.21272
- arxiv.org/abs/2607.16693
- blog.icml.cc/2026/04/06/announcing-the-icml-2026-workshops-and-affinity-workshops/
- ai2.work/blog/icml-2026-hits-record-23-918-submissions-as-agents-take-over

---

## 第 3 組：生成模型的理論與方法

## 範圍 3：生成模型的理論與方法（含離散擴散語言模型）

**領域份量與擁擠程度：** 我沒有找到 ICML 2026 依主題分類的官方統計。以下是估計：diffusion 與 flow 的理論與方法一直是 ICML 錄取論文中最大的主題群之一，約佔一成上下（估計值，沒有出處）。2025 下半年到 2026 年，熱點明顯轉向離散 diffusion／masked diffusion 語言模型（LLaDA、Dream 一系）。能找到的子題幾乎都有 2026 年的論文：
- reversal curse：2602.02133，ICML 2026
- masked diffusion 裡的 CFG：Rojas 等，ICLR 2026
- 資料受限下 diffusion 為何贏 AR：至少 5 篇
- diffusion 語言模型的記憶與外洩：2026 年已有 3 篇

這個領域非常擁擠，空位都在「不同社群的結論互相矛盾、但沒人用對照實驗收拾」的縫隙裡。

結論：**1 個候選，信心中低。** 第二個沒有合格候選，原因寫在後面。

---

### 候選 1：masked diffusion 語言模型比 AR 少記憶，是機制還是量測方式造成的假象？能否與它「資料效率高」用同一個量解釋？

**一句話主張：** 在同一份資料、同樣重複次數下，masked diffusion LM 對單一序列的逐字記憶，取決於它對「(序列, 遮罩樣式)」這個條件任務被訓練到的次數。因此：
- 即使改用對它最有利的抽取方式（任意遮罩），它要被抽出所需的重複次數仍比 AR 高。
- 這個倍數可以由訓練時遮罩比例的分布預測。
- 同一個量也決定它在資料受限時多晚開始過擬合（記憶起點隨資料量的增長斜率）。
- 加了輸入遮罩的 AR 會落在兩者中間。

**主張可否證：** 若改用最強的抽取方式後，兩者差距消失，主張即被推翻。

**文獻裡的對立說法：**
- 記憶與外洩的結論相反：
  - arXiv 2603.02333（2026-03）在以前綴為條件的對齊設定下，報告 diffusion LM 外洩個資「大幅少於」AR，並證明 AR 解碼是「最高取樣解析度」的極限情形。
  - arXiv 2605.24173（2026-05）認為只用前綴探測會低估風險：邊緣遮罩可多抽出約 3 倍逐字序列。在已移除個資的資料上，它對 email 的召回率還「高於」同規模的 AR。
  - 兩篇用的都是預訓練好的模型，無法控制重複次數，所以無法判斷誰對。
- 資料效率的原因說法相反：
  - 2507.15857 說主因是多種 token 順序。它的 AR 加 token masking 實驗「沒有改善」，但用 16 種順序的 AR 就接近 diffusion。
  - 2510.04071 說「隨機遮罩輸入 token 是主因」，而且 AR 加 dropout 與 weight decay 也能達到。
  - 2606.06888 說 AR 加上遮罩輸入的輔助損失，約等於多 1.3 倍不重複資料。
  - 2511.03276 說在 AR 加雜訊有幫助，但「補不上差距」。

**最接近的 5 篇（都已開啟摘要頁）：**
1. **2603.02333**：diffusion LM 記憶特性分析與取樣解析度定理。用預訓練模型，只用前綴協定比較 AR。沒有控制重複次數，沒有連結到資料效率。
2. **2605.24173**：任意遮罩填空式抽取，用 LLaDA-8B、Dream-7B。量測的是風險，沒有自己從零訓練對照模型，也沒有機制解釋。
3. **2507.15857**：資料受限下 diffusion 勝過 AR，主張是順序多樣性。量的是驗證損失，不是逐字記憶。
4. **2510.04071**：認為 token dropout 是主因。技術報告，仍是第一版。沒有量記憶或抽取率。
5. **2606.16246**（2026-06，v3 在 2026-08）：拆解 AR 的訓練時資料增強（遮罩、隨機替換、右到左、FIM、多步預測），最接近「因子拆解」。但摘要沒提到 diffusion 模型、資料量的變化，也沒有逐字記憶。

其他相關：
- 2505.16959：用 MD4 在 text8 上已證實記憶起點與資料量成正比，所以「MDLM 記憶起點線性增長」本身已不新。
- 2609.00873：EMNLP 2026 Findings，針對微調後的 diffusion LM 提出 token 層級記憶不對稱，遮罩比例越低記憶越多。方向相近，但做的是成員推論攻擊，沒有和 AR 對照。
- 2601.22450：ICML 2026，把遮罩比例分布當作隱性正則來調。
- 2606.06888、2511.03276：見上一段。

**停損實驗（約 1 週，約 25–35 GPU 小時，以下是我的估算）：**
- **設定：**
  - 同一個 Transformer 骨幹（約 1.25 億參數）從零訓練三個模型：AR、MDLM、AR 加 15% 輸入遮罩。
  - 用 FineWeb-Edu 或 OpenWebText 約 1B token 的子集，訓練約 2B token。
  - 植入約 1000 條 canary（隨機字串與自然句各半），重複次數依 1、2、4 一路倍增到 256 分組。
- **量測：**
  - 每組 50% 可被逐字抽出所需的重複次數（以下稱 k50），分別在三種協定下測：(a) 前綴；(b) 揭露 token 數相同時，取最佳遮罩樣式；(c) Carlini 的 exposure 排名。
  - MDLM 的似然只是 ELBO 上界，所以主指標用抽取率，不用似然。
- **判準：**
  - 在協定 (b) 下，MDLM 與 AR 的 k50 比值 < 1.5：「內在差異」不成立，停損。
  - 比值 ≥ 3，且 AR 加遮罩的 k50 落在兩者之間：繼續做。
- **公開資產：**
  - 程式：kuleshov-group/mdlm 或 MD4、nanoGPT
  - 資料：FineWeb-Edu／OpenWebText
- **算力估計：** 每個模型約 1.5e18 FLOPs，單張 H200 約 3–5 小時；三個模型加上抽取評估。

**全套實驗估計（約 600–1200 GPU 小時；10 週、4 張 H200 約有 6700 GPU 小時，做得完）：**
- 模型規模：1.25 億、3.5 億、10 億。
- 遮罩比例分布：均勻、截斷在 [0.25, 0.75]、偏低、偏高。
- 對照組：AR 加 N 種排列順序（複製 2507.15857 的設定）、AR 加 token dropout、AR 加 dropout 與 weight decay。
- 對每種設定量 k50 與記憶起點隨資料量的增長斜率，驗證兩者是否由同一個「各遮罩比例的有效曝光量」預測。
- 補一個簡單理論：每次重複隨機抽一個遮罩比例，記住某個條件分布需要約 c 次相近遮罩的曝光，由此推出 k50 對遮罩分布的公式。
- 最後在大模型上做外部驗證：LLaDA-8B 或 SMDM-1.1B，對照同資料訓練的 AR，用資料中自然出現的重複。是否有同資料訓練的 AR／diffusion 公開配對，見文末未確認事項。

**為何至今沒人做：**
- 研究記憶與隱私的一方，用的是現成預訓練模型和攻擊協定，無法控制重複次數。
- 研究資料效率的一方，量的是驗證損失，不量逐字記憶。
- 兩邊互相沒有引用到對方的矛盾點。2605.24173 已點出量測協定的問題，但沒有做從零開始的對照訓練。

**主要風險：**
1. 擁擠：2026 年已有 3 篇 diffusion LM 記憶論文，第 3 篇是 9 月，隨時可能再出現從零對照的研究。
2. 停損結果若是「量測方式造成的假象」，論文會變成量測釐清型，ICML 的吸引力較弱。
3. 審稿人可能把它看成實證研究。理論部分必須夠扎實，目前只有直觀模型。
4. 小模型的結論未必能推到 8B。
5. 與「資料效率原因」那條線重疊：2606.16246 與 2606.06888 已部分拆解增強方法，必須強調本題新在「記憶量」與「單一預測量」這兩點。

**信心：中低。**
- 加分：矛盾是實際存在、有出處的，停損很便宜、資產全公開，而且兩個社群的結論能接起來，貢獻定位清楚。
- 扣分：周邊非常擁擠；而且我只讀了摘要，沒讀全文。2603.02333 與 2606.16246 的全文可能已有部分對照實驗。

---

### 第二個：沒有合格候選

以下方向檢查過，都不合格：
- **單純拆解「遮罩 vs 順序 vs 密集計算」誰造成 diffusion 的資料效率：** 已有 2507.15857、2510.04071、2511.03276、2606.06888、2606.16246 五篇，空間太小，只能併入候選 1。
- **reversal curse 為何被 masked diffusion 緩解：** 2602.02133 已做，ICML 2026。
- **masked diffusion 裡的 CFG：** Rojas 等（ICLR 2026，arXiv 編號未確認）、A-CFG（2505.20199，NeurIPS 2025）已做。
- **MDLM 記憶起點與資料量成正比：** 2505.16959 已在 text8 上證實。
- **flow matching 的泛化是否來自目標的隨機性：** 有明確的對立文獻，但太接近已排除的「閉式去噪器解釋 DiT」，所以沒有提。
- **少步蒸餾與 guidance 區間：** 主要證據和實驗都以影像為中心，不符合這一輪的要求，沒有深入搜尋。

---

### 沒能確認的事項

- ICML 2026 生成模型相關論文的實際篇數與比例（上面是估計）。
- 以下各篇都只讀了 arXiv 摘要頁，沒讀全文：
  - 2603.02333：AR 與 DLM 是否同資料訓練；是否用了 SMDM-1.1B 對 TinyLlama 之類的配對。
  - 2606.16246：模型規模；有沒有改變資料量；有沒有量記憶。
  - 2510.04071：實驗設定。
- Rojas 等 masked diffusion CFG 論文的 arXiv 編號。
- 是否有「同資料訓練、規模相同的 AR 與 diffusion 公開權重配對」可供大模型驗證。
- 2026 年 10–11 月是否有從零訓練、植入 canary 比較 AR 與 diffusion 的新論文：搜尋沒找到，但沒有逐日翻 arXiv 列表，視為未確認。
- 停損與全套實驗的 GPU 小時是我的估算，假設 H200 的有效算力約 150 TFLOPs，未經實測。

**主要出處：**
[2507.15857](https://arxiv.org/abs/2507.15857)、[2510.04071](https://arxiv.org/abs/2510.04071)、[2511.03276](https://arxiv.org/abs/2511.03276)、[2606.06888](https://arxiv.org/abs/2606.06888)、[2606.16246](https://arxiv.org/abs/2606.16246)、[2603.02333](https://arxiv.org/abs/2603.02333)、[2605.24173](https://arxiv.org/abs/2605.24173)、[2609.00873](https://arxiv.org/abs/2609.00873)、[2601.22450](https://arxiv.org/abs/2601.22450)、[2505.16959](https://arxiv.org/abs/2505.16959)、[2505.17638](https://arxiv.org/abs/2505.17638)、[2602.02133](https://arxiv.org/abs/2602.02133)、[2603.22075](https://arxiv.org/abs/2603.22075)

---

## 第 4 組：最佳化與訓練動態

【範圍 4：最佳化與訓練動態】回報

一、領域份量與擁擠程度
最佳化（optimizer、學習率排程、訓練動態理論）一向是 ICML 的核心領域之一。我沒有查到 ICML 2026 依主題分類的正式錄取數。我估計它加上深度學習理論，約占錄取論文的一成上下，這是估計值，未確認。
這個領域在 2026 年非常擁擠，特別是 Muon 相關題目：光是我這次搜到的 2026 年 arXiv 就有幾十篇，每個月都有新文。題目包括 Muon 與 edge of stability、Muon 與 plasticity、Muon 預訓練與微調的優化器不一致、Muon 的 grokking、Muon 加權重衰減的平衡態等。
大部分「Muon × 某現象」的組合在 2026 年都已經有人做了。比較有空間的是用 Muon 當「介入變因」去解決一個老爭議。

二、候選（1 個）

候選 1：學習率 warmup 的機制。用 Muon 當自然實驗，判定 warmup 到底是在「壓低曲率」還是在「限制更新幅度」。

● 一句話主張
可以把 warmup 的效果拆成兩個預測不同的機制。第一個是更新幅度：早期的角度更新太大。第二個是曲率：一開始的（廣義）sharpness 太高。
Muon 的每步更新在譜範數下是固定的。如果「更新幅度說」是對的，Muon 加上角度更新控制之後應該幾乎不需要 warmup，而且與初始化尺度無關。如果「曲率說」是對的，那麼不用 warmup 時能用的最大學習率會由初始的譜範數廣義 sharpness 決定，也會隨初始化尺度改變。
可以否證的預測：在 Muon 下，「不用 warmup 時的最大穩定學習率 ÷ 用 warmup 時的最大穩定學習率」這個比值，是否會隨初始廣義 sharpness 的倍數變化。

● 對立說法與出處
- 曲率說：Kalra & Barkeshli，arXiv 2406.09405（NeurIPS 2024）。主張 warmup 的主要好處是逐步壓低（preconditioned）sharpness，讓模型撐得住較大的學習率。
- 更新幅度說：Kosson et al.，arXiv 2410.23922（NeurIPS 2024）。主張 warmup 是在抵銷早期過大的角度更新。把 optimizer 的更新正規化之後，可以大幅降低甚至不需要 warmup。
- 未解的矛盾：TrasMuon（arXiv 2602.13498）說標準 Muon「仍然依賴 warmup 長度與排程調整」。可是 Muon 本身就是正規化更新，照 Kosson 的說法不應該需要 warmup。另一篇 Muon 變體的技術報告則說 Muon 不用 warmup 也可以（只看到搜尋片段，未開全文確認）。兩邊說法對不上。
- 第三種說法：arXiv 2602.18849（2026）主張問題來自 attention sensitivity，並報告在 774M 模型上「attention 不會為了降低 sensitivity 而變尖」，與曲率說的直覺相反。

● 最接近的 5 篇與差別
1. arXiv 2406.09405（Kalra & Barkeshli）：曲率說，只研究 SGD/Adam，沒有用譜範數類 optimizer 當介入變因，也沒有和更新幅度說做對照實驗。
2. arXiv 2410.23922（Kosson et al.）：更新幅度說，只研究 AdamW/Lion 和他們自己的正規化變體，沒有操弄初始 sharpness 來區分兩種解釋。
3. arXiv 2602.05813「Where Does Warm-Up Come From?」（2026 年 2 月）：針對 Muon/Lion 這類範數約束 optimizer。假設「曲率隨次優差距下降」，從收斂證明推出 warmup 加 decay，並提出自動決定 warmup 長度的排程器。它是理論加排程器，沒有做機制判別實驗（只看了摘要頁）。
4. arXiv 2602.13498 TrasMuon（2026 年 2 月）：提出修正法（trust-region 縮放），把「Muon 依賴 warmup」當作動機，但沒有解釋原因。
5. arXiv 2603.05002「Non-Euclidean GD Operates at the Edge of Stability」（2026 年 3 月，ICML 2026）：定義了任意範數下的廣義 sharpness，涵蓋不含 momentum 的 Muon。實驗全程用固定學習率，完全沒有討論 warmup 或初始 sharpness（已開全文前 2/3 確認）。這篇提供了本題要用的量測工具。
其他相關但不重疊的：
- arXiv 2606.16768（Taming Curvature，2026 年 6 月）：用逐步加深架構來控制 preconditioned 曲率，只用 AdamW。
- arXiv 2609.34915（Muon Sublates EoS，2026 年 9 月）：研究 Muon 在 130M/1B 的 EoS，摘要沒有提到 warmup。
- PMLR v306（Alimisis et al., ICML 2026）「Why Do We Need Warm-up? A Theoretical Perspective」：收斂理論，沒有開全文，未確認有沒有做實驗判別。

● 停損實驗（約 1 週，約 30 GPU 小時）
- 設定：124M GPT 跑 FineWeb，用公開的 modded-nanogpt 或 nanoGPT，每次約 0.5B tokens。
- 因子設計：
  - optimizer 三種：AdamW；Muon（hidden 矩陣，其餘參數用 AdamW）；Muon 加上固定權重範數或角度更新控制。
  - 初始化尺度三種：0.5×、1×、2×，用來改變初始 sharpness，但不改變更新的範數。
  - warmup 兩種：0 與 2%。
  - 峰值學習率四種。
  - 合計 72 個短跑，每跑約 0.25 H200 小時，約 18 GPU 小時。
- 另外在初始點和 warmup 結束點，用 power iteration 量 Hessian 最大特徵值，以及 2603.05002 定義的譜範數廣義 sharpness，約 10 GPU 小時。
- 量測：
  - W = 不用 warmup 與用 warmup 的最佳 loss 差。
  - R = 不用 warmup 時的最大穩定學習率 ÷ 用 warmup 時的最大穩定學習率。
- 判準：
  - 值得做：Muon 在至少一種初始化尺度下 W ≥ 0.02 nats，而且 R 在三種初始化尺度之間的變化超過 1.5 倍。這表示 Muon 下仍有和曲率相關的 warmup 需求，現象真實存在且可以拿來判別。接著檢查 log R 和 log（初始廣義 sharpness 的比值）的相關，Spearman ≥ 0.8 就支持曲率說。
  - 停損：Muon 加角度控制在所有初始化尺度下 W < 0.01 nats 且 R ≈ 1。這等於只是確認了 Kosson 的說法，新意不夠。另一種停損情況是種子之間的差異大於效果本身。種子變異要對部分設定補做第二個種子來估。
- 公開資產：modded-nanogpt 或 nanoGPT 程式、FineWeb 資料、Muon 官方實作。廣義 sharpness 的估計可能要自己照 2603.05002 的定義實作，因為我沒看到它釋出程式碼（只讀了前 2/3）。

● 全套實驗估計（約 600–900 H200 小時，10 週內 4 張卡做得完）
- 模型規模擴到 350M、760M，跑 3 個種子。
- optimizer 擴到 Adam、Lion/SignSGD（ℓ∞ 幾何）、Muon、Muon 加角度控制。
- 再分別操弄初始化尺度、深度、embedding 與 head 是否進 warmup。
- 理論部分：在譜範數幾何下，推出「初始廣義 sharpness → 不用 warmup 時的最大學習率 → 所需 warmup 長度」的預測式，最後在 1B 模型加 WSD 排程上驗證預測的 warmup 長度。不需要從零訓練大模型。

● 為何至今沒人做
- 量 Muon 廣義 sharpness 的工具 2026 年 3 月才出現（2603.05002）。
- 討論 warmup 機制的論文幾乎都只用 AdamW。
- Muon 相關論文多半追求速度，或在提新變體，把「Muon 要不要 warmup」當作調參細節，沒有拿它來判別機制。

● 主要風險
1. Muon 實務上只作用在 hidden 矩陣，embedding 和 head 仍用 AdamW，會混淆結果。所以 warmup 要分參數群組操弄，否則判別會失效。
2. 審稿人可能認為題目太窄，只是「warmup 小題目」。要靠理論預測式和跨幾何（ℓ2、ℓ∞、譜範數）的一致性撐起份量。
3. 結果可能是兩種機制都有一點、無法乾淨區分。
4. 2026 年 warmup 理論論文正在增加（至少 4 篇），10 週內可能被搶先。

● 信心：中低
理由：對立說法明確、有出處，停損實驗便宜，資產也都公開。但這個領域 2026 年更新極快，題目本身偏窄，貢獻要夠大取決於理論部分做不做得出來。

三、檢查過、判定不合格的方向
- Muon 預訓練與微調的優化器不一致：2026 年已有至少 3 篇且互相矛盾，分別是 arXiv 2605.10468（ICML 2026）、arXiv 2605.06654，以及一篇 NeurIPS 2026 論文「The Ephemeral Cost of Switching Optimizers」（只看到會議頁，沒查到 arXiv 編號）。已經被做掉。
- 過度訓練（overtraining）與學習率衰減造成的 sharpness、遺忘：arXiv 2604.13627、arXiv 2605.02105 已經做了。
- Muon 與 loss of plasticity、持續學習：arXiv 2608.18319、2609.24678、2608.27518、2605.08949 已經做了。
- Muon 與 edge of stability：arXiv 2603.05002、2608.25990、2609.34915、2605.26929 已經做了。
- 權重衰減的平衡態結合 Muon：arXiv 2606.16899（Hyperball，只從部落格得知，未開原文）已經涵蓋。

四、沒能確認的事項
- ICML 2026 最佳化領域的錄取數與比例：未確認，只是估計。
- TrasMuon（2602.13498）「Muon 依賴 warmup」的具體實驗數字：只看到搜尋摘要，未開全文。
- 「Muon 不用 warmup 也可以」出自哪篇（搜尋結果指向 Muon 變體或技術報告，可能是 2610.37914）：未確認。
- 2602.05813 有沒有比較 Muon 與 AdamW 對 warmup 長度的敏感度：未確認。搜尋片段只顯示它在 124M Muon 上做了 warmup 範圍掃描。
- PMLR v306 那篇 warmup 理論（Alimisis et al.）、2602.18849 的全文內容：未確認。
- 2603.05002 有沒有釋出程式碼：前 2/3 沒看到，後段未讀。
- 2026 年 10–11 月是否有直接用 Muon 判別 warmup 機制的新 arXiv：我只做了 4 次相關搜尋，沒有找到，但不能排除。

---

## 第 5 組：微調、模型編輯與可信 ML

## 範圍 5：微調、模型編輯與可信 ML

**領域份量與擁擠程度**
- ICML 2026 收到 23,918 篇投稿，錄取 6,352 篇，錄取率 26.6%。這組數字出自二手來源（wasp-sweden.org、ai2.work 等），各來源一致，但我沒有核對 icml.cc 官方頁面。
- 我查不到 PEFT、model merging、unlearning、MIA 各占多少篇。我的估計是這幾類合計約占錄取論文的 4–7%。
- 這一區非常擁擠，以 2026 年 arXiv 的情況為例：
  - subliminal learning 的機制解釋，2026 年內至少有 5 篇互相競爭（2606.00995、2605.23645、2608.05734、ICLR 2026 divergence tokens、NeurIPS 2026 的 trait-direction drift）。
  - 「unlearning 為何被重新微調救回」至少 4 種說法：2505.09500、2605.11685、2505.22310，以及一篇 EMNLP 2026 論文。
  - 「on-policy 資料減少遺忘」也有 2510.18874、2601.19897、2603.01683 三篇。
- 以上三個方向我都判定已經做爛，不列為候選。

只有 1 個候選勉強合格。

---

### 候選 1：LoRA 合併失敗的共同原因是 intruder dimensions 嗎？

**一句話主張**
- LoRA 會在權重裡長出「intruder dimensions」：排名很前面、但和預訓練權重的奇異向量幾乎正交的新方向。主張是：LoRA 合併時的干擾主要來自這些方向。
- 文獻裡已有三種解釋：B 矩陣共用方向過度擁擠、各任務大小差異大、各任務方向不對齊。主張認為這三種都只是 intruder dimensions 的表徵或結果，不是獨立原因。
- 推論：合併前只壓低每個任務的 intruder 奇異值，應能挽回大部分合併損失，而且效果明顯勝過「拿掉同等能量的非 intruder 方向」。

**對立說法與出處**（各家對 LoRA 合併為何失敗說法不一）
- 2410.19735（KnOTS）：LoRA 各任務的更新方向比全參數微調更不對齊。
- 2505.15875（DO-Merging）：LoRA 各任務的參數大小差異大，差異越大合併越差。
- 2604.16826（Pico，"Crowded in B-Space"）：干擾主要來自 B 矩陣裡少數跨任務共用的方向；全文沒提 intruder。
- 2506.14126：主因是專家模型過度訓練，記住少數難例。
- 2608.11797：輸出端的交叉項只是旁觀者，崩壞是由各任務本身的位移帶來。
- 2608.09490：餘弦相似度、符號衝突、子空間重疊這些參數空間指標都解釋不了實際干擾。
- 只有 2510.19155（VeFA）一句推測「intruder dimensions 可能造成合併衝突」，沒有做實驗。

**最接近的 5 篇與差別**（摘要都已開啟）
1. **2410.21228**（LoRA vs Full FT: An Illusion of Equivalence，NeurIPS 2025）：提出 intruder dimensions，並用干預實驗證明它造成遺忘。只談單一模型的遺忘，沒談合併。
2. **2607.23711**（The Intruder Threshold，2026 年 7 月）：從預訓練奇異值預測哪一層會出現 intruder，也做了遺忘的干預。摘要沒提合併。
3. **2604.16826**（Pico，2026 年 4 月）：合併前壓低 B 的共用方向。沒有檢驗這些共用方向是不是 intruder，這正是本題要否證的一點。
4. **2410.19735**（KnOTS）：用 SVD 把各任務對齊後再合併，屬於方法論文。沒有找干擾的來源方向，也沒做干預。
5. **2606.03723**（Compress then Merge，ICML 2026）：先壓到共用的 rank-r 子空間再合併。摘要頁沒提 intruder，全文有沒有提未確認（二手摘要頁說有提）。

其他相關的：2505.15875、2509.17786（Core Space，NeurIPS 2025，沒說明失敗原因）、2608.09490、2608.11797。這些都沒有拿 intruder 做合併的干預。

**停損實驗**（約 1 週，約 30–45 GPU 小時）
- **資產**：
  - Qwen2.5-7B 或 Llama-3.1-8B，用公開資料各訓 6 個 LoRA（rank 16），任務如 NLI、GSM8K、程式、QA 等。每個約 2–3 GPU 小時。
  - 或者直接用 KnOTS 公開的 LoRA checkpoint 與程式（github.com/gstoica27/KnOTS），可省下訓練時間。
  - 合併方法用 Task Arithmetic 與 TIES，基準用 Pico、KnOTS。
- **做法**：
  1. 依 2410.21228 的定義，找出每個任務、每一層的 intruder 方向。
  2. 量這些 intruder 方向和 Pico 那些「B 共用方向」的子空間重疊度，以隨機子空間當基準。
  3. 合併前三種處理對照：壓低 intruder 奇異值；拿掉等能量、排名相近的非 intruder 方向；不處理。
- **判準**（以「各任務單獨準確率」正規化後的合併準確率計算）：
  - 繼續做：壓低 intruder 能挽回 ≥40% 的合併差距，且比對照組至少多 3 個百分點。
  - 停損：挽回 <20%，或和對照組差 ≤1 點，或 intruder 與 B 共用方向的重疊度和隨機差不多。

**全套實驗估計**（約 8–10 週，約 600–1,000 GPU 小時，4 張 H200 足夠）
- 模型與合併規模：3 種模型家族（含一個 MoE 或 SSM）、2 種規模、合併 2–8 個任務、4 種合併方法。
- 學習率與 α 掃描，用來操控 intruder 數量，做劑量反應曲線。
- 全參數微調當對照：全參數微調幾乎不長 intruder，預期它的合併差距主要不來自這裡。
- 最後可以延伸出一個只用預訓練奇異值、不需資料的合併前處理方法。

**為何至今沒人做**
- intruder 概念到 NeurIPS 2025 才正式發表。2026 年的後續工作（2607.23711）集中在遺忘。
- 合併社群近一年在 B 矩陣、大小差異、對齊三條線上各自出方法，沒人把兩條文獻接起來。
- 但我只查了摘要層級，不能排除某篇全文裡已做過類似的消融實驗。

**主要風險**
- 可能 intruder 和 B 共用方向本來就高度重疊，結論會變成「Pico 換個名字」。審稿人可能認為貢獻偏小。
- 也可能 intruder 只解釋一小部分干擾，結果是負面或混合的。2608.09490 的提醒在這裡適用：干擾隨提示詞而變，單一指標容易被反駁。
- merging 是方法論文的紅海，要靠「否證既有三種說法」的機制故事撐，不能只靠刷分。
- 2607.23711 的作者很可能下一步就做合併，搶先風險在這 10 週內真實存在。

**信心：中低**
- 加分：停損便宜，資產全公開，對立說法明確且多。
- 扣分：題目偏增量，搶先風險高；全文層級的撞題檢查還沒做完。

---

### 考慮過但排除的方向
- **unlearning 的重新學習機制**：已有 4 種以上互相競爭的 2025–2026 解釋（2505.09500、2605.11685、2505.22310，以及 EMNLP 2026 的 Forget-Retain 那篇）。要做也只能寫成仲裁性論文，撞題風險太高。
- **subliminal learning**：2026 年內已有至少 5 篇機制論文。
- **on-policy 蒸餾與遺忘**：2510.18874、2601.19897、2603.01683 已經涵蓋。
- **蒸餾的 capacity gap**：要做蒸餾預訓練，停損實驗做不到 50 GPU 小時以內。
- **LLM 的 membership inference**：Duan 2402.07841 認為 MIA 在 LLM 上接近隨機，Hayes 2505.18773 用強 MIA 反駁。這場爭論已有人用強 MIA 回應，加上時間有限，我沒找到能便宜否證的新機制角度，沒有深入搜尋。

### 未能確認的事項
- ICML 2026 錄取數字只來自二手部落格；各子領域篇數查不到，文中是估計。
- 2606.03723（Compress then Merge）全文是否已拿 intruder 做合併實驗：摘要頁沒提，二手摘要說有提，未確認。
- KnOTS 公開了哪些 checkpoint（我印象中有 ViT-B/32、ViT-L/14 各 8 個視覺任務，以及 Llama-3-8B 的 NLI 任務）：未確認。
- 2607.23711、2604.16826、2608.09490、2608.11797 我只讀了摘要或頁面摘要，沒讀全文。
- 2026 年 9–11 月是否有「intruder + merging」的新 arXiv：我做了兩次針對性搜尋都沒找到，但搜尋引擎的覆蓋不完整。
- 一些段落引用了 Pico 與 KnOTS 的二手描述（例如「B 的跨任務重疊遠大於 A」），沒有逐一核對原文。

---

## 第 6 組：表格與時間序列基礎模型、不確定性

## 範圍 6 回報：表格與時間序列基礎模型、不確定性

### 領域份量與擁擠程度
我沒有找到 ICML 2026 按子領域統計的投稿或錄取數字，以下都是估計。表格 in-context 模型確實擠進了 ICML 主會，例如 TabICLv2 收錄於 ICML 2026（PMLR 306，arXiv 2602.11139）。不過這個子領域被幾個大團隊的模型迭代主導：TabPFN-2.5 → TabPFN-3（2605.13986）、TabICLv2、TabDPT、Mitra。小團隊能切入的大多是「分析或診斷」型論文。
時間序列基礎模型（TSFM）的論文量大，但會議論文偏重模型與基準，機制分析幾乎集中在少數實驗室。conformal 與分布偏移下的校準是 stat.ML 的傳統主題，穩定但擁擠。整體估計中等偏擁擠。

### 結論
有 1 個勉強合格的候選，信心低到中。另外兩個方向我原本看好，但查證後發現已被做掉，列在最後給你參考。

---

### 候選 1：表格 in-context 模型在「遠離 context 的輸入」上的行為與不確定性

**一句話主張**
這是一個可否證的機制假說。表格 in-context 模型（TabPFN-2.5/3、TabICLv2、TabDPT）遇到超出 context 資料範圍的輸入時，預測會退回 context 的標籤邊際分布（prior reversion），而不是像貝氏後驗那樣依先驗外推。若假說成立，在共變量偏移下會出現「機率仍校準，但資訊量歸零」的現象：
- 原生區間與未加權的 split conformal 覆蓋率大致守得住；
- 但準確率崩壞；
- 依距離分組後，覆蓋失效集中在中等距離帶，而不在最遠處。

**對立說法與出處（都已開原文頁確認摘要）**
- 2505.20003（TabPFN: One Model to Rule Them All?，v2 2025-11）：把 TabPFN 當近似貝氏推論，聲稱它在共變量偏移預測上「有時」勝過專門方法。這隱含它的外推是合理的。
- 2602.04596 v2（2026-07）：在 two moons / spirals 上寫「Far from the data TabPFN reverts to a maximum-entropy predictive distribution」。但同一篇在 logistic 實驗又說 epistemic 不確定性在 OOD 最高，文內沒有調和兩者，v1 和 v2 的說法也不同。這是現成的「未解釋現象」。
- 2605.13986（TabPFN-3 報告）：作者聲稱加入 OOD prior 後模型「能外推」，並說多數其他表格基礎模型沒有這個能力。這一點我沒有獨立驗證。
- 2505.16226（Realistic Evaluation of TabPFN v2 in Open Environments）：結論是 TabPFN v2 在開放環境限制明顯，但適合共變量偏移的小任務。說法和上一條互相拉扯。

**最接近的 5 篇與差別**
1. 2602.04596 v2，Uncertainty Decomposition for Bayes-Filtered Transformers（2026-07）
   - 做了什麼：在合成資料上分解 TabPFN 的不確定性，觀察到遠處退回最大熵；附錄 K 有共變量缺口實驗。
   - 差別：沒有跨模型比較，沒有在真實偏移資料集上量覆蓋率或 conformal，也沒有對照精確貝氏後驗、去量化「退回程度隨距離的變化」。
2. 2505.20003，TabPFN: One Model to Rule Them All?
   - 做了什麼：只看 TabPFN 一個模型，比較共變量偏移下的點預測表現。
   - 差別：不分析機制，也不看不確定性與距離的關係。
3. 2609.02766，Do Tabular Foundation Models Know Physics?（2026-09，v3 2026-10）
   - 做了什麼：用 316 條物理方程測 TabPFN-3、TabICLv2、TabDPT、Real-TabPFN-2.5；指出外推會「在半個週期內鬆弛成常數」，且模型的不確定性表達不了無雜訊的確定性機制。
   - 差別：限於物理方程的點外推，沒有做分布偏移下的覆蓋或 conformal，也沒有提出退回機制。這是最大的撞題風險。
4. 2606.01427，On the UQ Ability of Tabular Foundation Models（2026-05）
   - 做了什麼：比較 TabPFN-2.5 與 GP 的迴歸不確定性。
   - 差別：摘要沒有提到偏移或 OOD；全文我沒有確認。
5. 2505.16226，Realistic Evaluation of TabPFN v2 in Open Environments（2025-05）
   - 做了什麼：開放環境下的表現評測。
   - 差別：摘要沒有校準或不確定性分析，只測 v2。

**停損實驗**
- 第一部分，合成資料：用 GP、線性、加法模型產生資料，context 限制在一個盒子內，測試點放在 Mahalanobis 距離 d = 0–6。量兩件事：
  - 退回指數 R(d) = |pred − ȳ_ctx| / |f(x) − ȳ_ctx|；
  - 預測熵或變異隨 d 的變化，並對照同先驗下的精確 GP 後驗。
- 第二部分，真實資料：取 TableShift 的偏移切分，加上在 TabArena 資料集上用特徵分位數做的人工偏移。比較三種區間的覆蓋率，並依距離分組：原生區間、未加權 split conformal、加權 conformal。
- 判準：
  - 支持主張：至少 3 個模型都滿足「d > 3 時 R(d) 的中位數 < 0.3」（明顯退回）；而且原生或未加權 conformal 在至少 60% 的偏移資料集上，覆蓋率偏離名目值不超過 3 個百分點，但準確率下降至少 10%。
  - 停損：各模型的行為不一致（沒有共同機制），或 TabPFN-3 的 OOD prior 已讓 R(d) 接近真實外推。
- 計算量：只有推論，估計 20–40 張 H200 小時，一週內做得完。
- 公開資產：
  - 權重：TabICLv2（GitHub soda-inria/tabicl）、TabDPT、TabPFN-2.5 與 TabPFN-3（HF 上有權重，但授權條件未確認）；
  - 資料：TableShift、TabArena；
  - 工具：MAPIE 等 conformal 套件。

**全套實驗估計**
- 機制部分：檢查最後一層 test→train 注意力，看遠點是否變成均勻注意力，也就是退回邊際分布；並做溫度或 logit 介入的因果檢驗。
- 方法部分：基於距離的 conformal 修正，或用不確定性做偏移偵測。
- 規模：5 個模型 × 約 50 個偏移切分。
- 估計 4 張 H200 用 3–4 週推論，10 週內足夠，不需要訓練。

**為何至今沒人做**
現有工作分成兩群，各自在原分布內評估：一群在模型層面追求準確率與外推能力（TabPFN-3 加入 OOD prior），另一群做不確定性理論（martingale 後驗、不確定性分解）。「遠處行為到底是退回還是外推」只在 2602.04596 的合成圖上順帶出現，而且前後矛盾。TabPFN-3 與 TabICLv2 剛在 2026 上半年發布，「新 OOD prior 是否改變這個行為」是新問題。

**主要風險**
1. 條件 1（可推廣的機制）只是勉強過關，容易被審稿人看成評測論文。
2. 2609.02766 的作者群或 2602.04596 的作者群很可能自然延伸到這裡。
3. 如果 TabPFN-3 已經會外推，主張只對部分模型成立，故事會變成「不同 prior 決定遠處行為」，力道會變弱。
4. TabPFN 權重的授權。

**信心：低到中**
有明確矛盾的出處，停損也很便宜。但貢獻天花板偏低，撞題風險中等。

---

### 查過但已被做掉、不建議的方向
- **表格 ICL 的長 context 注意力稀釋**：ChunkTabPFN（2509.00326，ICLR 2026）說 TabPFN 的表現會隨 context 變大而提升，和 LoCalPFN（NeurIPS 2024）的「欠擬合、需要檢索」相反。但 TabICLv2（ICML 2026）已經用 log-n 縮放（QASSMax）處理，TabPFN-3 也採用；Localized TabICLv2（2608.16429）顯示 kNN 局部 context 只保留全 context 98.64% 的準確率，已經不再有幫助。另外這個方向和本專案已排除的 attention 題材太接近。
- **TSFM 的 context parroting 機制**：2505.11349（v3 2026-03）提出 parroting 基準；2602.01605（v2 2026-08）已用 head ablation 把 motif parroting 與季節性偏誤歸因到特定 head。已被做掉。
- **時間序列的共變量與 lead-lag**：Chronos-2 論文自己已報告約 30 步的 lead-lag 上限。
- **full conformal 結合 ICL**：2504.15722、2509.01840 已做。
- **TabPFN 的 martingale 性質**：2505.11325、2510.25154、2603.12037 已做。

### 沒能確認的事項
- ICML 2026 各子領域的投稿或錄取數量：沒有找到官方統計，上面全是估計。
- 2609.02766、2606.01427、2505.16226、2505.20003 都只讀了摘要，全文是否已含「偏移下覆蓋率對距離」的分析，未確認。
- TabPFN-3 的權重是否公開、授權是否允許研究使用，未確認。
- TabPFN-3 的 OOD prior 實際的外推行為：只有作者自述。
- 2602.04596 的 v1 與 v2 說法矛盾，未確認是更正還是不同實驗。
- 2026 年 9–11 月是否有直接撞題的新稿，只做了有限次搜尋，無法排除。

---

## 第 7 組：推理、test-time compute 與強化學習

【範圍 7：推理、test-time compute 與強化學習】結論：沒有合格候選

一、這個領域在 ICML 2026 的份量與擁擠程度
- ICML 2026 收到 23,918 篇投稿，錄取 6,352 篇，錄取率 26.6%；投稿數是 2025 年（12,107 篇）的兩倍左右。出處是二手報導 ai2.work（https://ai2.work/blog/icml-2026-hits-record-23-918-submissions-as-agents-take-over），我沒有核對官方數字。
- 「LLM 推理加上 RLVR（用可自動驗證的獎勵做強化學習）」與 test-time scaling 各佔多少篇，我找不到官方統計。我估計這是 2026 年最擁擠的子領域之一。依據是：同一個現象常在幾個月內就出現 3–6 篇互相回應的論文。例如「隨機獎勵也能提升 Qwen 的數學分數」這個現象，至少有 Shao 等人（ICML 2026）和 Chen 等人（ICLR 2026）兩篇在爭論成因；「RL 之後思維鏈到底有沒有被模型真正用到」這個問題，2026 年內就有 4 篇以上。
- 對只有 10 週的新進者，這個領域的特點是：停損實驗很便宜（多數只需推論，或用小模型跑 GRPO），但被別人搶先的風險很高。我檢查的三個方向，在 2026 年 2–9 月都已有論文直接做掉核心主張。

二、為何沒有合格候選（三個方向都檢查過，最後都卡在條件 5「2026 年內沒被做掉」）

1. 隨機獎勵為何有效：GRPO 的 clipping 偏差，還是 entropy 動態？
- 對立說法確實存在：
  - Shao 等人 arXiv 2506.10947（ICML 2026）認為原因是 clipping 偏差放大了模型原本就偏好的寫程式解題行為。
  - Chen 等人 arXiv 2512.16912（ICLR 2026）認為只靠 clipping 偏差不夠解釋，並指出 Llama、QwQ 也有類似增益。
- 但這兩篇已經互相交手，2026 年 3 月還有一篇 "On the Direction of RLVR Updates for LLM Reasoning"（只看到標題，內容未確認）。
- 剩下能做的空間只是再補一個解釋，不夠撐一篇主會論文，而且被搶先的機率高。不提。

2. RLVR 讓 pass@k 在 k 很大時反而下降（pass@k 倒轉）：用基礎模型每題原本的答對率來預測
- 對立說法：
  - Yue 等人 2504.13837 認為 RL 不會擴大模型能解的題目範圍。
  - ProRL 2505.24864 認為訓練夠久就能擴大。
- 已被做掉：arXiv 2607.20543 "When RLVR Shrinks the Reasoning Boundary: Diagnosing Pass@k Inversion"（2026 年 7 月）已提出機制：基礎模型只偶爾答對的「邊界題」，正確解法太稀疏，RL 取樣時抽不到。它還依每題的基礎答對率提出修正方法 PBA。
- 相關的「RL 等於把分布變尖」觀點，也已有 Karan & Du 2510.14901（ICLR 2026）和 2601.15609。不提。

3. RLVR 是否讓思維鏈真正左右答案，取決於任務難度或「不寫思維鏈能否答對」
- 對立說法：
  - Fu 等人 2509.17380 認為 RLVR 會強化推理過程中的因果結構。
  - RFEval 2602.17053（ICLR 2026）的結果是 RLVR 會降低忠實度。60.05→46.32、61.38→50.93 這兩組數字來自二手摘要，我沒有核對原文。
- 已被做掉：
  - Yu 等人 2604.22074（ICML 2026，Qwen2.5 搭配 ReasoningGym）：RLVR 之後，40 個任務中有 19 個的因果重要性指標（CIR）下降，17 個的充分性指標（SR）變差；而且原本 CIR/SR 較高的任務，RL 之後較可能再提升。
  - 2609.25366 "From Decorative to Load-Bearing: Task Difficulty Shapes the Causal Role of CoT"（2026 年 9 月）已證明思維鏈是否真正被依賴主要由任務難度決定，也提到推理型 RL 會壓低錯誤傳播。
  - 我原本想主張的「難度或免思維鏈準確率決定 RL 的效果」，核心已被這兩篇涵蓋。不提。

另外快速看過、判斷太擁擠所以沒有深挖的方向：
- 多數決與 RL 造成的多樣性塌縮：2504.10478、2505.21444、2509.15194、2603.08660。
- verifier 與不用 verifier 方法的比較：2502.12118（ICML 2025），之後的進展未確認。
- 熵最小化或自信度獎勵：2505.15134、2505.22660。
- 離線 RL 的 horizon 瓶頸：Park 等人 2406.09329、2506.04168。後者已把價值函數與策略的 horizon 分開處理，2026 年的後續發展未確認。

建議：如果主對話仍想在這個領域找題目，比較可能有空隙的是「離線 RL 或世界模型」這種非 LLM 的小規模 RL 機制題，但這一輪我沒有做到開 5 篇論文的深度查核。

三、沒能確認的事項
- ICML 2026 中推理與 RL 主題的論文篇數；上面的總投稿與錄取數也只來自二手報導。
- 2604.22074 有沒有分析訓練過程中的中間檢查點：摘要沒寫，全文沒讀。
- RFEval 的具體數字只來自二手摘要。
- 2609.25366 用的 RL 設定是不是 RLVR：摘要只寫 "reasoning-specific RL"。
- "On the Direction of RLVR Updates for LLM Reasoning"（2026 年 3 月）：只看到標題，內容與 arXiv 編號都未確認。
- 離線 RL 與世界模型方向在 2026 年 6–11 月的最新論文：未搜尋，未確認。
- 第一個方向的資訊大多來自摘要與二手摘要，我沒有逐篇讀全文。
