## 文獻新穎性查核：S2（gated attention × sink × KV 淘汰／repair）

**結論：** 題目的三個部件都已有很近的前作。沒有一篇同時做完三件事，但每一塊單獨看，剩下的差異都很窄。目前 repair 的效果（+0.67pp，區間跨 0）撐不起方法論文。

只讀了 repo，沒有改任何檔案。下文標「已開」的，是我實際開過 arXiv abs 或 html 頁核對的內容。標「未確認」的，只看過搜尋摘要。

### A. 方法複述

依 README、paper/draft.md、cards/S2-repair-exploration.md、cards/S2-Astra-method-design.md、baselines/README.md。

1. **背景與診斷。** 研究對象是 Qiu et al. 2025 釋出的配對 1.7B checkpoint，分成 non-gated、headwise-gated、elementwise-gated 三種。gated attention 在 SDPA 之後乘上 sigmoid output gate。在這組模型上，gate 幾乎消除了 attention sink：首 token 注意力質量從 35.3% 降到 1.9%／0.4%。固定 KV 預算下多保留前 4 個 token 的收益因此縮減 93–97%。另一個假設是「逐 head 的 sink 質量能預測該 head 的保護收益」，結果 ρ=0.265，判 NO-GO。
2. **gval 打分。** 第一代方法是 `s = 平均_q a_{q,k} · ‖g_q ⊙ v_k‖²`，在 KV group 內平均後做 top-k。在 PG-19 上，相對 Observed，壓縮造成的額外 NLL 降 12.6%。但在 RULER 上明顯輸給 CAOTE。
3. **現行方向：一次性 prompt KV 淘汰後的集合修補（repair）。**
   - 起點是 SnapKV 的保留集合（之後打算換成 Afterglow 或 KeyDiff）。
   - 對每層、每個 KV group，用觀測窗內的 query 計算重正規化後的輸出殘差 `r = o − u/m`。
   - 誤差經過 native gate 與 W_O 投影，在 head 內加總後再取平方範數；聚合方式有平均、tail（最差 25% query）、entropic 與 mass 約束幾種。
   - 每步最多評估 4×4 個 keep/drop 交換，最多交換兩次，保持預算與 prefix／recent 保護不變，只改保留集合，不改寫 K/V。
4. **目前主張。** gate 與 W_O 感知的多 query 集合修補，可以改善強 initializer 的下游品質。現有證據：RULER13 DEV208、W1024 上，Projected 37.98 對 SnapKV 37.31，區間跨 0。平均每個 group 只淨換約 1.1 個 token。Afterglow+SnapKV 則有 89.3。

### B. 最接近的工作（依重疊程度由高到低）

| # | 論文（已開） | 方法一句話 | 與本題重疊 | 本題仍不同之處 |
|---|---|---|---|---|
| 1 | **Trust the Mass: Forced Weights in KV-Cache Eviction**，Shi & Gu，arXiv 2608.25230（2026-08，v3 11-05），NeurIPS 2026 FEDL workshop。https://arxiv.org/abs/2608.25230 | 推導重正規化子集合誤差 `es(A)=‖g_A‖/p(A)`。「balancing selector」從 top-mass 集合出發，每步做一次 keep/drop 交換以降低對 dense 輸出的誤差，中位數交換 1–2 次。結論是 top-mass 已接近最佳，交換只補上 2–5% 的差距。也量測 GQA 共用集合的代價。正式方法 ContourKV 是在實體預算下執行 KVzip 打分。 | 幾乎就是 repair 的核心：精確重正規化誤差加上從強起點做貪婪交換，連「交換次數很少、收益很小」的發現都一致。 | 該文以單一 query row 為主、沒有 gate 與 W_O 度量、沒有 tail 聚合，交換只用作分析，不是部署方法。本題換成多 query、group 共用集合、gate⊙W_O 投影、tail 目標，並當成 runtime 方法。 |
| 2 | **GateSight**，Li, Lou, Li，arXiv 2610.14754（2026-10），preprint。https://arxiv.org/abs/2610.14754 | 在 hybrid（GDN 加 full attention）模型上，用 GDN 的 write gate、decay 與寫入量預測 KV 效用。用 native attention output gate 的層平均值分配各層預算。 | 直接測過 gate × value 作 head 內 token 排序：在 Qwen3.8-27B 上，gated 與 ungated 分數的 Spearman 為 0.995，端到端與不用 gate 無法區分。論文結論是 gate 是很弱的逐 token 訊號，但在 head／層之間是很強的訊號。 | 這是 gval 與 gate 敘事的直接反證（不同模型）。本題用的是 Qiu 配對 checkpoint、純 attention 模型、有 elementwise gate。該文沒引用 Qiu 2025，也沒討論 sink。 |
| 3 | **ValueDiff: Value-Geometric KV Cache Eviction for Sink-Suppressed LLMs**，Park, Choi, Lee，arXiv 2609.23314（2026-09），preprint。https://arxiv.org/abs/2609.23314 | 新架構（QK-norm、gated attention、learned sink、softcap）的 sink 變弱，現有淘汰法依賴 sink 而失效。改以 `‖v_j − v̄‖` 排序淘汰。 | 已占住「sink-free／gated 模型需要不同淘汰法」的敘事。量了 11 個模型的 sink rate 與 BOS 質量，包括 gated 的 Qwen3.5-4B／9B。RULER13 8K、LongBench 21 任務；固定保留首 token 加最近 128 個，與本題同類設定。 | 沒用 Qiu 的配對 checkpoint，沒量「保護前 4 個 token 的收益」，打分不用 gate。本題 W1 的配對設計比較乾淨，但只剩這一點差異。 |
| 4 | **Reformulating KV Cache Eviction Problem（LaProx）**，Mai & Kim，arXiv 2605.07234，NeurIPS 2026。https://arxiv.org/abs/2605.07234 | 把淘汰改寫成「注意力 × 投影後 value（V·W_O）」的層級矩陣乘法近似。打分 `‖A[:,i]‖·‖VW_O[i,:]‖`，考慮 head 間依存，全模型做一次 top-K。 | 「Projected 用 W_O 投影誤差」這個點已被做過。模型為 Llama-3.1-8B、Mistral-7B、Qwen3-8B；LongBench 加 NIAH。 | 沒有 gate，不做集合交換，也不在 head 間先相加再取範數。 |
| 5 | **CriticalKV**，Feng et al.，arXiv 2502.03805，ICML 2026。https://arxiv.org/abs/2502.03805 | 從輸出擾動角度選 KV：先依注意力選 α·b 個，其餘用 `(A+ε)·‖V W_O‖₁` 補滿，目標是最小化最壞情況的擾動。 | 已涵蓋 value 加 W_O 投影，以及「最壞情況（類 tail）」目標。 | 沒有 gate、沒有集合交換、沒有重正規化殘差。 |
| 6 | **CAOTE**，Goel et al.，arXiv 2504.14051（v6）。https://arxiv.org/abs/2504.14051 | 閉式的淘汰誤差 `c_j = α_j/(1−α_j)·‖VAᵀ−v_j‖`；Appendix B 推廣到多 token 聯合淘汰。 | 本題的 `r=o−u/m` 集合公式就是它 Appendix B 的形式。gval 也是「注意力 × value」家族。 | 沒有 gate／W_O、沒有交換搜尋。 |
| 7 | **DropKV**，Zhang et al.，arXiv 2610.22130，NeurIPS 2026。https://arxiv.org/abs/2610.22130 | 指出輸出擾動最小化的集合問題是 NP-hard。改為逐 token 獨立決策，在 cone 條件下有常數倍近似保證，另附 Triton kernel。 | 同樣以輸出擾動為目標，並且已討論集合與逐 token 的取捨，還有理論保證。 | 本題做集合層級的交換，但沒有任何保證；DropKV 有保證也有效率。 |
| 8 | **ReST-KV**，An et al.，arXiv 2605.08840，ICLR 2026。https://arxiv.org/abs/2605.08840 | 以層級輸出重建誤差建模每個 token 被移除的影響，加上 EMA 時間平滑與自適應窗。 | 輸出誤差導向的淘汰。 | 是否用 W_O 未確認（摘要沒寫）；沒有 gate。 |
| 9 | **Monte Carlo Estimation for KV Cache Eviction（LORE-KV）**，Bilal et al.，arXiv 2610.07643。https://arxiv.org/abs/2610.07643 | 先抽樣生成短續寫，取回應端的 query 作為未來 query，用投影後 leave-one-out 輸出刪除成本排序。 | 投影後的輸出成本，並處理 prefill 與 decode 的 query 錯配（這正是本題 Astra 卡列出的失敗假設之一）。 | 本題不做額外生成；LORE-KV 不做集合交換。 |
| 10 | **VATP（Attention Score is not All You Need… Value Also Matters）**，Guo, Kamigaito, Watanabe，EMNLP 2024。https://arxiv.org/abs/2406.12335 | 分數 = 注意力 × ‖v‖₁；並指出 sink token 的 value norm 很小，會被誤刪。 | gval 去掉 gate 後就是它；本題的 value-only 對照等同 VATP（L2 平方而非 L1）。 | 只多了 gate 這一個因子。 |

其他相關（已開）：

- **Sigmoid Attention as a Better Substrate for Learned KV Eviction**（2608.23296，ICML 2026 AdaptFM workshop，GPT-2 規模）。
- **Protection Is (Nearly) All You Need**（2605.18053）：結構保護主導打分；position-0 sink 約占 prefix 質量 75%。沒有 sink-free 模型。
- **AFTERGLOW**（alphaXiv 2026-10，NeurIPS 2026 標記）：把分數往後傳播，以保住答案 span 的結尾。
- **MomentKV**（2606.01563）：保存被淘汰集合的 moment，做一階輸出修正。
- **OBCache**（2510.07651，ICML 2026）：OBD 式的輸出感知閉式分數。
- **The Gating Mechanism of KV Cache Eviction**（2611.07605）：gate 放在 softmax 內，用 dense 模型的預測訓練。repo 把這個 arXiv 號記為 IndexMem++；abs 頁標題不同，但 comments 指向 IndexMem repo，請核對。
- **VaSE**（2606.03928）：保護大 norm 的 value。
- **When Does Value-Aware KV Eviction Help?**（2605.08234）：診斷 value-aware 選擇何時有用。
- **LAVa**（2509.09754）。
- Sink 機制面：**Gu et al., When Attention Sink Emerges**（ICLR 2025）；**A Unifying View of Attention Sinks**（2606.08105，指出 gating 消除 nop 型 sink，但會增加 broadcast 型 sink）。
- **Gated Attention**（2505.06708）。

### C. 有沒有工作已經做了實質相同的事

**沒有一篇同時做完三件事**：在配對 gated／non-gated checkpoint 上量 sink 保護收益、gate⊙W_O 投影的多 query 集合交換修補、從強 initializer 出發。但三個部件各自都有很近的前作。

- **repair 部件與 Trust the Mass 重疊最深。** 重正規化子集合誤差、從 top-mass 起點做貪婪 keep/drop 交換、收益很小，這三點都一致。CAOTE Appendix B 已有集合誤差公式，LaProx 與 CriticalKV 已有 W_O 投影，CriticalKV 已有最壞情況目標，DropKV 已有「集合問題 NP-hard、改逐 token」的論述。本題仍獨有的只剩：多 query 共用集合、native gate 進入度量、tail／entropic 聚合，以及作為 runtime 方法。但現有數據是 +0.67pp（區間跨 0）、平均每 group 淨換約 1.1 個 token，等於重現了 Trust the Mass「交換幾乎沒用」的結論。
- **gval 與 gate 訊號部件已被 GateSight 直接打臉。** gate×value 在 head 內不改排序（ρ=0.995）。本題的差異只剩 elementwise gate 與 Qiu checkpoint，而且本題自己的 RULER 結果也是負的。
- **sink 診斷部件被 ValueDiff 先占了敘事。** ValueDiff 已在 gated 的 Qwen3.5 上量 sink 消失，並主張要用 value 幾何淘汰。Gated Attention 原論文本身也主打 sink-free。本題剩下的新意是：用配對 checkpoint 做受控比較、量出「保護前 4 個 token 的收益」縮減 93–97%。這是一篇診斷性 short paper／workshop 等級的發現。它不是因果 gate ablation（elementwise 版的 FFN 寬度不同），逐 head 機制 P3 也已 NO-GO。

**剩餘新穎性很窄。** 最站得住的只有「配對 checkpoint 下 sink 保護收益消失」這個受控實證，以及「gate⊙W_O、多 query、tail 目標的集合修補」這個未被做過的組合。後者目前沒有品質增益，所以撐不起 ICML 的方法論文。

### D. 審稿人最可能的「這不就是 X 嗎」

1. **「這就是 Trust the Mass 的 balancing selector 加上 W_O 投影，而且它已經證明交換沒什麼用。」** 回應需要三樣：
   - 在同一個 evaluator 下，top-mass、單 query 交換、多 query 交換三者做配對比較。
   - 顯示多 query 與 gate 度量能帶來顯著的下游增益，並在多預算、多模型上複現。
   - 固定 support，量 joint 誤差與 decode query 誤差，說明增益的機制。
2. **「gate×value 就是 VATP／CAOTE 乘一個 gate；GateSight 已顯示 gate 不改 head 內排序。」** 回應需要：
   - 在自己的 checkpoint 上報 gated 與 ungated 分數的逐 head 排序相關。
   - identity-gate、shuffled-gate、mean-energy gate 三種對照的下游結果（不只 NLL）。
   - 若沒有差異，就放棄 gate 敘事。
3. **「sink-free 所以不用保護 sink，這很顯然，Gated Attention 與 ValueDiff 都已指出。」** 回應需要：
   - 量化的保護收益曲線，加上逐 head 或逐層的機制。P3 已失敗，必須換一個可預測的量。
   - 擴到公開 gated 模型（Qwen3.5-4B／9B 或 Qwen3-Next），以及超過訓練長度。
4. **「為什麼不直接用 Afterglow？它 89.3，你 38。」** 回應需要 Afterglow+repair 相對 Afterglow 的配對增益，也就是 repo 計畫中的 A1−A0。
5. **「LaProx、CriticalKV 已用 W_O，DropKV 已處理集合與逐 token 的權衡，並有保證與 kernel。」** 回應需要：
   - 實際的成本量測：prefill、打分、交換的時間與峰值記憶體。目前完全沒有。
   - 與 DropKV、LaProx 在相同預算下的直接比較。
6. **規模質疑：只有 1.7B、原生 4K。** 近期接受的工作都在 7–14B，以及 RULER 8K–128K 與 LongBench。

### E. 審稿人預期的 baseline 與 benchmark

以下依已開論文的實際設定歸納：ValueDiff、LaProx、CriticalKV、Trust the Mass、LORE-KV、ReST-KV、MomentKV。

- **模型。**
  - 主流 dense：Llama-3.1-8B-Instruct、Mistral-7B-Instruct-v0.3、Qwen2.5-7B／14B／32B、Qwen3-8B／14B。
  - gated／sink-suppressed：Qwen3.5-4B／9B（ValueDiff 已用）、GPT-OSS-20B。
  - 1.7B 只能當受控診斷。
- **Benchmark。**
  - RULER 完整 13 任務，至少 4K–32K，最好到 64K／128K。
  - LongBench 16–21 任務。
  - NIAH／Needle；InfiniteBench；SCBench（多輪 reuse，GateSight 有用）。
  - 推理類可加 MATH-500。
- **預算。** 每 head 128–2048，或壓縮比 25–90%。並清楚區分 query-aware 與 query-agnostic（Trust the Mass 指出問題是否在 context 內可差 87.6 分）。
- **Baseline。**
  - 必備：StreamingLLM、H2O、SnapKV、PyramidKV、AdaKV、CriticalKV、KeyDiff、Expected Attention、KVzip、CAOTE／FastCAOTE、DropKV、ValueDiff、Afterglow。
  - 輸出誤差類必比：LaProx、ReST-KV、OBCache、Compactor。
  - learned 類另列：FastKVzip、KVzap。
- **指標。**
  - 任務分數與相對 Full 的保留率。
  - 實體 KV 記憶體：Trust the Mass 指出 mask 實作會讓分數虛高 14–62 分。
  - prefill／打分額外延遲、decode 延遲、峰值記憶體。

### F. repo related work／baseline 清單漏掉的重要論文

`paper/references.bib` 只有 8 筆，連 StreamingLLM、H2O、KeyDiff、ValueDiff、DropKV、Afterglow、Trust the Mass、GateSight、CriticalKV 都沒收。

`baselines/recent-method-survey.json` 共 21 項，以下論文不在其中：

- **LaProx（2605.07234，NeurIPS 2026）**：最該補，W_O 投影與 head 間依存。
- **ReST-KV（2605.08840，ICLR 2026）**：輸出重建。
- **VATP（2406.12335，EMNLP 2024）**：gval 的無 gate 原型。
- **LORE-KV（2610.07643）**：只在 Astra 卡出現，沒進清單。
- **LAVa（2509.09754）**、**VaSE（2606.03928）**、**When Does Value-Aware KV Eviction Help?（2605.08234）**。
- 機制面：**Gu et al. 2410.10781**、**A Unifying View of Attention Sinks（2606.08105）**、**Sigmoid Attention substrate（2608.23296）**、**Protection Is (Nearly) All You Need（2605.18053）**。後兩篇只在 go-topics 提到，沒進清單。

只看到搜尋摘要、未開啟，因此**未確認**：Active-Dormant Attention Heads（2410.13835）、AdaKV（2407.11550）、LookaheadKV（2603.10899）、Rethinking KV Eviction via Information-Theoretic Objective（2604.25975）、PyramidKV、KVzip、TOVA。

另外兩點：
- repo 的 IndexMem++ 對應 arXiv 2611.07605，abs 頁標題是「The Gating Mechanism of KV Cache Eviction」，請核對對應關係。
- GateSight 的模型名「Qwen3.8」是網頁原文寫法，未另行查證。
