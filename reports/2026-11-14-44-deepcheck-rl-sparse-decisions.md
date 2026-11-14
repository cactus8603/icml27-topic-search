# 深查：把「RL 只改少數決策」變成單一模型（回報 39 的候選 1）

日期：2026-11-14。沒有用 GPU，沒有寫程式。

## 結論先講

| 項目 | 結果 |
|---|---|
| 照原本寫法（蒸餾版加 gated GRPO 訓練版）的信心 | **中偏低**（原本評為「中」） |
| 改成下面「建議切入角度」後的信心 | **中**；第 1 週的先導實驗若過關，可以升到**中偏高** |
| 撞題程度 | 現象本身至少已有第 5 篇；「只在決策位置訓練、其餘位置貼住 base」至少已有 4 篇（其中 2 篇是 ICLR 2027 投稿）；「把兩個模型的混合策略蒸餾成單一模型，並以 pass@k 為目標」我沒找到 |
| 最大的實證風險 | rl-replace 在 pass@k 上的優勢只在 k≥64 左右才明顯，而且相對 base 的差距在統計上不顯著 |
| 可行性 | 程式碼公開，RL checkpoint **沒有公開**。不過 SimpleRL-Zoo、Oat-Zero、ORZ、DeepScaleR 都有公開的 base／RL 配對可以直接用；改在 1.5B 上做，前兩週約 150–250 GPU 小時【推估】 |

降級的原因有三：

1. 原本寫的「訓練版（gated GRPO）」已經被做掉。
2. 原本寫的「蒸餾版」有部分前例。
3. 題目賴以成立的 pass@k 優勢，統計上比回報 39 寫得弱。

剩下的空間是「混合目標蒸餾」，再加上嚴格的 pass@1 對 pass@k 前緣比較。

## 來源層級標記

- 【全文】：用 curl 抓 arXiv HTML 全文，轉成純文字後自己讀（不是小模型轉述）。
- 【摘要】：arXiv API 或 OpenReview API 回傳的原始摘要（逐字）。
- 【搜尋摘要】：只看到搜尋引擎整理的摘要。
- 【推估】：我自己的估計。

網路操作共約 32 次：WebSearch 4 次；WebFetch 2 次，都因 ACL Anthology 連不上而失敗；其餘是 curl 對 arXiv、arXiv API、Semantic Scholar、OpenReview、Hugging Face 與 GitHub 的請求。

---

## 1. 三篇核心論文全文的確認

### 1.1 arXiv 2610.24663「What RL Brings to Reasoning: A Few Decisions the Base Model Would Not Make」【全文】

作者來自 Meituan LongCat、UCSD、CUHK-Shenzhen，2026-10-17 上線。

**回報 39 的數字，逐一核對：**

| 回報 39 的說法 | 全文 | 判定 |
|---|---|---|
| 約 4% 位置 | 依 token 計 4.0%，依回應平均 4.5%；各年度 4.3–4.7% | 正確 |
| 拿回 90.5% pass@1 增益 | 120 題 AIME：base 10.00 → rl-replace 18.44，RL 19.32；回收率 90.5%，95% CI [77.0, 108.6] | 正確 |
| k=256 時 67.5% 高於 RL 的 61.4% | 表 6：base 61.64、RL 61.40、rl-replace 67.50 | 正確，但有兩個重要的但書（見下） |
| rl-replace 每個 AIME 年度 7.3–7.9 GPU 小時 | 正確。GPU 是 **H800**；MATH-500 約 55 GPU 小時；k=256 的大 k 實驗約 **1,012 GPU 小時** | 補上型號與總量 |

**回報 39 沒寫、但很關鍵的細節：**

- **pass@k 優勢在統計上偏弱**：
  - 相對 RL 模型，k=256 的差距是 6.10 點，95% CI [0.82, 11.02]。
  - 相對 base 是 5.86 點，CI **[-0.33, 10.71]，包含 0**。作者自己寫「the comparison with the base model remains less certain」。
  - 小 k 時幾乎沒差：相對 RL，k=4 只差 0.13 點，k=16 差 2.28 點。
  - Qwen3-8B 的 pass@256：rl-replace 72.50、RL 66.82、base 68.33。全文沒給這組的 CI。
- 這組 Qwen3-4B 配對的 RL 模型在 k=256 時**並沒有輸給 base**（61.40 對 61.64）。所以這篇的故事是「混合策略同時贏過兩者」，不是「RL 失去 pass@k、混合策略把它保住」。
- 和整段回應層級的混合（每次嘗試以 0.92 機率選 RL 模型）比，在 pass@1 對齊（19.00 對 18.98）的條件下，k=256 時 rl-replace 領先 6.05 點，CI [0.93, 10.98]。
- 作者自己標註，比較是「等嘗試次數」而非「等算力」。
- 跑與跑之間變異大：base 在 120 題上的 pass@1 介於 8.2% 到 10.26%；RL 在 AIME 2025 上六次評估介於 20.4% 到 23.3%。
- **附錄 E**：只用 base 自己的信心就能定位高 KL 位置。
  - 熵的 AUROC 0.945／0.960；gradient boosting gate 0.969／0.980。
  - 在 90% recall 時 precision 只有 0.54／0.62。
  - 作者結論：「位置好找，難的是在那裡選哪個 token。」
- RL 設定：
  - verl GRPO，DAPO-Math-17k，lr 1e-6，每步 128 個 prompt × 8 個 rollout，8 張 GPU。
  - 選用的 checkpoint：4B 是 step 460，8B 是 step 320，Llama-3.2-3B 是 step 500。

**有沒有單一模型的方法**：沒有。作者明說把 rl-replace 當診斷工具，不是實用的解碼方法。結論只延伸到「學習式或熵式的選擇方案我們沒有評估」，並點名 Akgül et al. (2026)。

**程式與 checkpoint**：
- 程式公開：github.com/CyrusZhang04/WhatRLBrings，2026-10-14 建立，4 顆星。
- README 寫「Model checkpoints and datasets must be supplied separately」，所以 **RL checkpoint 沒公開**。
- `configs/rl_training/` 有 4B、8B 與 Llama-3.2-3B 的 GRPO 設定，可以照著重訓。

**相關工作段落自己列出的先行者**（這是回報 39 漏掉的部分）：
- Meng et al. 2026（2603.22446，ICLR 2026）
- Huang et al. 2026（2603.22117，ICLR 2026）
- Shen et al. 2026（2605.16874）
- Akgül et al. 2026（2605.06241）

前三篇的引用文字是「substituting the RL model's tokens at such positions recovers most of its accuracy」；Akgül 那篇在 §3.5 另外點名（見上一段）。也就是說，這篇在「稀疏替換」這個現象上已經不是第一篇。它新增的是 pass@k 那一段（§3.6）。

### 1.2 arXiv 2610.39295「There Is No Free Lunch in RLVR: Every pass@1 Gain Has a pass@k Price」【全文前段約 1.6 萬字元，加上關鍵字檢索】

- 單一作者，純理論，附驗證程式（github.com/yliusci/no-free-lunch-rlvr）。
- **模型**：把 RLVR 視為在「跨 prompt 共用、**集合固定**的解題策略」上的 exponential tilt。Q 矩陣固定，policy 是策略單形上的一點。
- 主要定理（以下都是全文原句的意思）：
  - 在 exclusive／dominant specialization（假設 1）下，任何提高 pass@1 的更新都會讓某些 prompt 變差。
  - 加上 fine coverage（假設 2）時，pass@k 交叉是被迫的。
  - 「no update both raises pass@1 and dominates the base pass@k curve at every k」。
  - 帶有每題下限的受限更新會描出最佳前緣（定理 4）。
  - 作者也指出例外：若有一個策略支配其他所有策略，就有 free lunch。
- 作者明說「We do not claim anything about a specific language model beyond what published curves show」。全文沒有任何 LLM 實驗，只轉錄了 2607.20543 的一張表。
- **和本題的關係**：
  - rl-replace 是 token 層級的重組，會產生 base 策略集合以外的新軌跡，等於改變 Q 本身，不屬於「在固定策略集合上 tilt」。
  - 所以它不是這個定理的反例，而是落在定理的假設之外。
  - 這可以當論文的一個論點：「token 層級重組可以離開策略層級的前緣」。但要有 rl-replace 在 pass@1 與 pass@k 兩端都**顯著**優於 base 的證據才站得住，而 2610.24663 目前相對 base 的 CI 包含 0。
- 評價：單一作者、沒有實證，審稿人未必重視。只能當對照，不能當主軸。

### 1.3 SCOPE（ACL 2026 Findings，2026.findings-acl.641）【搜尋摘要】

- 標題：「A Few Bad Apples Spoil the Bunch: Preventing Global Entropy Collapse Driven by a Small Set of Tokens in LLM Reasoning」。作者 Jaeeun Jang、Hansle Lee、Sangmin Kim。
- **全文沒讀到**：
  - ACL Anthology 的 PDF 用 curl 連線逾時，WebFetch 回傳 ECONNREFUSED（主站與 preview 站都一樣）。
  - 也找不到 arXiv 版：arXiv API 以標題片語搜尋只回傳無關結果，兩次 WebSearch 也沒找到。
- 搜尋摘要給的內容：
  - 對每個 token 算 redistribution score，只對分數最高的約 5% token 加 KL 正則。
  - 主張熵崩塌是由少數結構性決策點的過早自信造成的。
  - 在 RLVR 與 RLIF 下，pass@1 與 pass@k 都進步，贏過先前的熵控制基線。
- 區隔：SCOPE 是「在少數 token 上加約束」，屬於 RL 訓練時的正則。它不需要 RL 模型、也沒有混合策略當目標分布。這個區隔仍然成立，但我無法確認它的 k 值範圍與基線。

---

## 2. 「只在決策位置訓練」是否已被做掉：是

以下是回報 39 沒列、但直接打到「訓練版（gated GRPO）」的工作：

| 工作 | 狀態 | 做法 | 有沒有評 pass@k | 等級 |
|---|---|---|---|---|
| **ReasonMaxxer**（Akgül et al., 2605.06241） | arXiv v3；**已投 ICLR 2027**（OpenReview 上看得到） | 只在 base 高熵的決策點用 advantage 加權的對比損失，**其餘位置對凍結的 base 加 KL**；LoRA rank 32；完全不跑 RL；50 題、1,000 條序列 | **沒有**，只報 pass@1／avg@8 | 【全文】 |
| 同篇 §4 的 KL-LoRA 蒸餾 | 同上 | 用 100 題，把 RL 模型的 token 層級 KL 蒸餾進 LoRA，在 MATH-500 與 GSM8K 上追平 RL | 沒有 | 【全文】 |
| **DPR**（Decision-point reinforcement） | ICLR 2027 投稿 | 由凍結的 reference 依熵排序位置；在最高熵位置做 clipped PG，其餘位置用低變異 KL 貼住 reference；不需要 verifier | 摘要沒提 | 【摘要】 |
| **Archer**（2507.15778） | **NeurIPS 2026** | 高熵 token 放寬 clip、減弱 KL；低熵 token 加強 KL | 有，「improving both pass@1 and pass@K」 | 【摘要】 |
| SENT（2512.04359） | arXiv | 只在低熵 token 上加 KL | 摘要宣稱勝過其他熵方法 | 【摘要】 |
| 2506.01939（80/20 forking tokens） | NeurIPS 2025 | 只對最高熵的 20% token 算梯度 | — | 【全文】，從 2605.06241 的轉述得知 |
| ICT（2606.19771） | arXiv | 用 JS 散度挑出 top 10% token，只更新這些 | pass@4 平均 +4.58% | 【摘要】 |
| SRA（2610.34199） | arXiv，2026-10-25 | 蒸餾後學生加上 Discrepancy Gate 與 top-K 殘差 adapter，把稀疏修正內化；Qwen3 0.6B–4B | Pass@16 最多 +4.37 點 | 【摘要】 |
| Shen et al.（2605.16874） | **ICML 2026** | 只在高分歧位置讓推理模型接手一個 token；可以超過同尺寸的推理模型 | 附錄有 Pass@8 | 【摘要】，加上全文關鍵字檢索 |
| Policy-Merge | ICLR 2027 投稿 | 在解碼時組合多個專家，並證明「學生對多個教師做 on-policy 蒸餾、最小化加權 per-token KL」等價於這個組合 | — | 【摘要】 |

**判定**：
- 「訓練時只放決策位置自由、其餘貼住 base」至少有 ReasonMaxxer、DPR、Archer、SENT 四個版本。回報 39 的「訓練版」不能當貢獻。
- 「把 RL 模型的修正蒸餾成小 adapter」也已有 KL-LoRA（ReasonMaxxer §4）與 SRA。
- **還空著的地方**：
  1. 以 rl-replace 這種**混合策略本身**當蒸餾目標（不是 RL 模型），而且目標是 pass@k。
  2. 上面這些方法彼此之間、以及和 rl-replace 之間，在同一組配對上的 pass@1 對 pass@k 前緣比較，搭配嚴格的配對統計。

  ReasonMaxxer 沒評 pass@k；DPR 的摘要也沒提。

## 3. 引用者與擁擠程度

- **2610.24663 的引用者**：Semantic Scholar 回傳「Paper with id arXiv:2610.24663 not found」，還沒收錄。改用標題搜尋時被限流（429）。OpenReview 以標題搜尋 ICLR 2027，沒找到直接的後續。**這篇沒有已知的後續工作，但無法完全排除。**
- **2603.22446（ICLR 2026，現象的先行者）的引用者**：27 篇。和本題接近的有：
  - token 選擇類 RLVR：2608.30632 GMTS（EMNLP）、2606.31575、2606.19771 ICT、2605.21467 DelTA、2605.25381。
  - 解碼融合：2607.27248 Divergence Decoding。
  - 2611.02574「RLVR Sharpens the Pencil, Not the Mind」。

  沒有一篇是「把混合策略蒸餾成單一模型」。
- **2605.16874（ICML 2026）的引用者**：3 篇，即 SRA、2606.22830 DEAR（決策加證據 token 的 on-policy 蒸餾）、2606.17735。
- **2605.06241 的引用者**：4 篇，都不相關。
- **OpenReview ICLR 2027**：查得到（api2.openreview.net）。相關投稿有 ReasonMaxxer、DPR、Policy-Merge、IDA-OPD（解決 on-policy 蒸餾的 pass@k 瓶頸）、「How RL Reshapes LLM Reasoning: Transferability, Coverage, and Scaling Laws」、「Learning from the Gap Between Pass@K and Pass@1」、「Theoretical Analysis of On-Policy Distillation: Generalization and Pass@k」、Cliff Tokens。
- **9–11 月保留多樣性／pass@k 的方法**（皆為【摘要】）：
  - PBA（2607.20543，每題 base 錨定）
  - ReCo（2607.26862）
  - IDA-OPD（2608.29846）
  - 2610.07405（指出 pass@k 量不到多樣性，NeurIPS 2026 workshop）
  - 2609.22547（見第 4 節）

  再加上回報 39 已列的 TAR、DPH-RL。

**擁擠度評估**：「保住 pass@k 的 RLVR」與「決策 token 的 RL」兩條線都很擠，每月有數篇。本題能站的位置很窄：「混合目標蒸餾 + 前緣比較 + 嚴格統計」。

## 4. 最強的反駁與能否回應

**反駁 1（最強）：pass@k 的差距不可靠，可能根本沒有可以保住的東西。**
- 2609.22547【摘要與全文檢索】在 3k–16k token 上限下審了 9 個公開 RLVR checkpoint，包含被 Yue et al. 當主要證據的 4 個 SimpleRL-Zoo。結果：
  - 只有 1 個在多重比較校正前勉強達到「先贏後輸」的交叉標準。
  - 校正後沒有一個存活。
  - token 上限本身就能製造、掩蓋或反轉交叉。
- 在 32k 上限、1,060 題新題目上，只有 DeepScaleR-1.5B 與 ProRL-1.5B（base 是 R1-Distill-Qwen-1.5B）有被認證的交叉，兩個 7B 模型都沒有。
- 2610.24663 的優勢也只在大 k 才明顯，而且相對 base 的 CI 包含 0。
- **回應**：
  - 把主張從「保住 base 的 pass@k」改成「在 pass@1 對齊下贏過 RL 與各種基線」。2610.24663 相對 RL 的差距 CI 不含 0。
  - 評測改用 2609.22547 的建議：**加題數比加取樣深度更有統計檢定力**。用 1,000 題以上、k≤64 的配對 bootstrap，而不是 120 題、k=256。
  - 選配對時先確認 RL 在 pass@k 上真的有輸，或 rl-replace 真的有贏。
  - 這個反駁可以回應，但必須先做第 1 週的檢查才知道有沒有題目。

**反駁 2：權重內插（WiSE-FT）或全域 KL 已經達到同樣效果。**
- 2504.10478 在 SFT 上，靠 WiSE-FT 就「almost completely recovers Pass@k while also improving Pass@1」【摘要】。
- 2610.24663 沒有比較權重內插。
- **回應**：只能用實驗回應。把 λ 掃描與 KL 係數掃描畫成前緣，和本方法的前緣比。若本方法輸，就停。這是第 2 週的停損點之一。

**反駁 3：這就是 ReasonMaxxer、DPR 或 Archer 的變體，或就是換了教師的 OPD。**
- **回應**：
  - 這些方法是在 RL 或對比損失上做 token 選擇，目標是 pass@1。
  - 本方法的目標分布有封閉形式：位置 t 的教師分布 q*(·|y<t)，在兩個模型 greedy 一致時等於 p_B，不一致時等於 δ(argmax p_R)。這正是 rl-replace 的逐步分布（2610.24663 附錄 A.1 寫明「at the positions where the two greedy tokens agree the output is distributed exactly as standard base sampling」）。
  - 所以「學生能否學會一個已知有高 pass@k 的混合策略」是可以檢驗的明確問題。
  - 這個回應站得住，但新意不算大，**必須靠前緣比較與 pass@k 結果撐起來**。

**反駁 4：蒸餾本身會壓掉多樣性。**
- 2606.26091 指出 on-policy 自我蒸餾會降低輸出多樣性【搜尋摘要】；IDA-OPD 指出取樣 token 版的 on-policy 蒸餾在 pass@k 上會停滯。
- **回應**：q* 在約 96% 的位置就是 base 的完整分布，可以用全詞表的 forward KL（只需要兩個模型各一次 prefill）。這反而是 IDA-OPD 想避開的高成本做法，在 1.5B 上負擔得起。不過蒸餾後是否真的保留多樣性，仍是本題最主要的實證不確定點。

**反駁 5：等算力比較。**
- rl-replace 每一步都要跑兩個模型。單一學生模型的推論成本和 RL 模型相同，所以這反而是本題的賣點：把需要兩倍算力的診斷工具變成一倍算力的模型。可以回應。

## 5. 可行性

**公開的 base／RL 配對**（Hugging Face API 實查）：

| 配對 | 備註 |
|---|---|
| Qwen2.5-Math-1.5B → sail/Qwen2.5-Math-1.5B-Oat-Zero；Llama-3.2-3B → sail/Llama-3.2-3B-Oat-Zero | 小，便宜 |
| Qwen2.5-0.5B／1.5B／7B／14B、Llama-3.1-8B、Mistral-7B、DeepSeek-Math-7B → hkust-nlp/*-SimpleRL-Zoo | ReasonMaxxer 用的就是這組；但 2609.22547 指出它的 pass@k 交叉多半不顯著 |
| Qwen2.5-0.5B／1.5B／7B／32B → Open-Reasoner-Zero | 1.5B 被 2609.22547 審過 |
| R1-Distill-Qwen-1.5B → DeepScaleR-1.5B-Preview、ProRL-1.5B | **唯一有被認證 pass@k 交叉的配對**，但要 32k 上限才顯著，評測貴 |
| Qwen3-4B-Base 的 GRPO 版 | 2610.24663 沒公開；HF 上有社群版（例如名稱含 DAPO17K 的），品質未經確認，不建議依賴 |

**壓低算力**（全部【推估】）：
- **不需要 lock-step 雙模型解碼來訓練**：學生用 vLLM 正常生成，之後 base 與 RL 各做一次 prefill，就能算出 q*。成本約等於一般 on-policy 蒸餾。
- rl-replace 本身只需在小規模重現一次當參考。原文 4B 大 k 實驗花了約 1,012 GPU 小時（H800），不要照做。
- 1.5B 配對的成本粗估：
  - rl-replace 重現：MATH-500 加 AIME，k≤64，約 30–60 GPU 小時。
  - 每次蒸餾：約 15–30 GPU 小時。
  - 每次評測：1,000 題 × 64 次取樣，約 3–8 GPU 小時。
- **前兩週約 150–250 GPU 小時**，比回報 39 估的 400–600 少一半以上。若再加一個自訓的 Qwen3-1.7B GRPO 配對（照作者公開的設定），另加約 60–100 GPU 小時【推估】。

## 6. 信心評級與理由

**原寫法：中偏低。**
- 訓練版已被 ReasonMaxxer、DPR、Archer 做掉。
- 蒸餾版有 KL-LoRA 與 SRA 前例。
- 現象本身至少是第 5 篇。
- pass@k 優勢的統計基礎薄弱。

**調整後：中。**
- 「以混合策略為目標的單一模型，加上 pass@1 對 pass@k 前緣」我沒找到撞題。
- 目標分布有封閉形式，問題明確。
- 公開配對多，算力可以壓到 1.5B 規模。
- 卡在兩個實證未知：
  1. 在公開配對上，rl-replace 的 pass@k 優勢能否以配對統計重現。
  2. 蒸餾後的學生能否保住這個優勢，並贏過 WiSE-FT 與全域 KL。

**要推到中偏高的條件**：第 1 週在至少一個公開 1.5B 配對上同時看到下面三件事（見停損判準）：
- rl-replace 的 pass@64 顯著高於 RL；
- 一次 LoRA 混合目標蒸餾的先導實驗，pass@k 高於 RL；
- 而且不輸給同樣 pass@1 的 WiSE-FT 點。

這三件事都做得到，但我沒有證據預測結果，所以現在不能給中偏高。

## 7. 建議的切入角度

**題目方向**：「Distilling a token-level hybrid policy: one model on the pass@1–pass@k frontier」。

1. **方法（主貢獻）**：混合目標的 on-policy 蒸餾。
   - 教師是封閉形式的 q*_λ：在一致位置用 p_B；在不一致位置用 p_B^(1-λ)·p_R^λ 正規化後的分布，λ=1 再取 greedy 就是 rl-replace。
   - 用 λ（以及「不一致」的判準：greedy 不一致、KL 閾值、或 base 熵 gate）描出一條前緣。
   - 學生從 base 起始，LoRA 與全參數都做；損失用全詞表 forward KL。
   - 另外做一個 **RL-free 的變體**：用 ReasonMaxxer 或 base 熵 gate 取代 RL 模型的位置資訊，看位置來源對前緣的影響。這樣可以直接和 ReasonMaxxer、DPR 正面比較。
2. **評測（第二貢獻，也是說服審稿人的關鍵）**：
   - 依 2609.22547 的方法做 pass@k 的配對 bootstrap 與同時信賴帶。題目數 1,000 以上，k≤64，再挑一組做 k=256。
   - 在 pass@1 對齊的條件下比 pass@k。
   - 基線：RL、WiSE-FT λ 掃描、GRPO+全域 KL、ReasonMaxxer、Archer（有公開程式）、整段回應層級混合、rl-replace（上限，標註 2 倍成本）。
3. **分析（第三貢獻，可選）**：用 2610.39295 的診斷工具（只需要每題的 base 與 RL 成功率）檢驗「token 層級重組是否落在策略層級前緣之外」。只當作討論，不當主張。

**明確不做**：gated GRPO（已被做掉）；量化與混合解碼的延伸（回報 39 的候選 2、3），留到主線成立後再說。

## 8. 停損判準（事先寫定）

1. **第 1 週，D5 前**：在 Oat-Zero-1.5B 與 SimpleRL-Zoo-1.5B（或 ORZ-1.5B）兩個配對上：
   - rl-replace 必須用 ≤10% 的位置拿回 ≥70% 的 pass@1 增益；
   - 在 ≥500 題的配對 bootstrap 下，pass@64 必須比 RL 高，且 CI 不含 0。

   兩個配對都做不到就**整題停**（沒有可以蒸餾的目標）。只有一個做到就縮成單一配對，加上自訓的 Qwen3-1.7B 配對再試一次。
2. **第 2 週，D10 前**：LoRA 混合目標蒸餾必須保住 ≥70% 的 pass@1 增益；在同樣 pass@1 下，pass@64 必須高於 RL 與最佳 WiSE-FT 點，且 CI 不含 0。做不到就停蒸餾線。剩下的前緣比較只夠 workshop，不符合「純負面不行」的門檻，應換題。
3. **任何時候**：若 arXiv 或 OpenReview 出現以 rl-replace（或任何 base／RL token 層級混合）當蒸餾目標並報告 pass@k 的工作，重新評估。每週查一次 2610.24663 的引用（Semantic Scholar 收錄後）與 ReasonMaxxer 的後續。
4. **第 3 週檢查點**：若學生只在 k≥128 才贏，k≤64 不贏，評估成本與統計檢定力都不夠，降為備案。

## 沒能確認的事項

- SCOPE 全文（ACL Anthology 連不上，也沒有 arXiv 版）；它的 k 值範圍與基線不明。
- Archer、SENT、DPR、SRA、Policy-Merge 只看了摘要。DPR 有沒有評 pass@k 不明。
- 2610.24663 在 Semantic Scholar 尚未收錄，引用者無法用 API 查。
- 2610.39295 只讀了前段約 1.6 萬字元與關鍵字段落，附錄的證明沒讀。
- 所有 GPU 小時都是推估；唯一有出處的是 2610.24663 的 H800 數字。
- 公開配對上 rl-replace 的稀疏度與 pass@k 行為，目前只有 2610.24663（Qwen3、Llama-3.2-3B）與 ReasonMaxxer（SimpleRL-Zoo，只看 pass@1）的證據。
