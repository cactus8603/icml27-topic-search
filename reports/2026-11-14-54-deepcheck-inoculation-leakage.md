# 深查：選擇性泛化防禦（inoculation 家族）的漏出預測與修正

日期：2026-11-14｜對象：回報 50 的候選 A｜方式：只讀文獻與 API，沒有動 GPU，也沒有寫實驗程式
網路操作：約 30 次。來源包括 arXiv HTML 全文 8 篇、arXiv API、Semantic Scholar citations API、OpenReview ICLR 2027 搜尋，以及 GitHub 存在性檢查。

來源層級標記：
- 〔全文〕：讀了 arXiv HTML 全文。註：HTML 轉純文字時，LaTeX 數學式裡的數字會遺失，這類情況會另外註明。
- 〔摘要〕：只讀了摘要。
- 〔API〕：API 回傳的 metadata。
- 〔推估〕：本報告自己的估計。

---

## 0. 結論先講

| 項目 | 判斷 |
|---|---|
| 照回報 50 原樣做（兩路徑理論 + step-0 預測量 + 無提示 KL 錨定修正） | **中偏低**。三個組件各自都已經有人做過相近的東西，詳見 §2。 |
| 改成本報告 §5 的切入（跨防禦類型的一階「跨情境轉移」預測量 + 預測量對決 + 解釋 on-the-line 斜率） | **中** |
| 改切入後，前兩週停損判準通過 | **中偏高** |

最大的新發現是 **2610.09384 Persona Hierarchy Model（PHM，ICLR 2027 投稿中）**〔全文〕。它幾乎同時佔走候選 A 的三個組件：
- 理論：「共享的預設 persona」對應候選 A 的無條件路徑，「局部 persona」對應受情境條件化的路徑。
- 訓練前預測量：訓練前綴的 persona 向量與預設 persona 向量的距離，和泛化窄度相關，Qwen3-4B 上 Pearson r = 0.72。
- 修正：PPR，在預設前綴下對初始模型加 KL 懲罰，這就是「無提示 KL 錨定」。

回報 50 沒有找到這篇。

---

## 1. 指定論文逐篇確認

### 1.1 原作

**2510.04340 Tan et al.**〔全文〕
- 實驗：主實驗用 GPT-4.1，透過 OpenAI 微調 API；Qwen2.5-7B-Instruct 做複製。
- 機制假說：inoculation 讓資料變得「較不意外」，降低全域更新的最佳化壓力。這等於提出了一個以 loss 為基礎的預測方向。
- 他們用中性提示下特質回應的 log-prob 隨訓練的變化，追蹤哪個特質被全域學到。這個量幾乎就是「無條件漂移」。
- 限制段自承：特質會漏到預設 assistant persona；某些情境下漏得更多（例如 "You write secure code" 仍可引出 EM）。
- 程式：論文附 anonymous.4open.science 匿名連結，沒有確認是否已有正式 GitHub。

**2510.05024 Wichers et al.**〔全文〕
- 程式公開：github.com/safety-research/inoculation-prompting（HTTP 200）。
- 四個設定：
  - MBPP reward hack：Qwen-2-7B base，以執行測試判定；
  - CEBaB 虛假相關：Llama-3-8B-Instruct，以準確率判定；
  - GCD 諂媚：可程式判定；
  - CMV 毒性：用 OpenAI Moderation API 加 Claude 評審。
- 至少 5 個 seed。
- **已有訓練前預測啟發式**：提示在初始模型上越能引出不良行為，當 inoculation 提示就越有效。5 個設定中有 4 個相關係數 > 0.57；Mixtral 上 Pearson 0.60（圖 6 說明文字）。
- 這是本題必須擊敗的基線之一。

### 1.2 2604.16423 Shifting the Gradient（Grant, Gillioz, Ward, McGrath；Goodfire + MATS）〔全文〕

- 版本與程式：
  - v2 是 2026-11-07，文中提到「camera-ready evaluations」，所以已被某個會議接受。會議名稱沒有在擷取到的文字中看到。
  - 程式公開：github.com/satchel-goodfire/shifting_the_gradient，MIT 授權。
- **step-0 預測量的實際定義**：
  - 計算 activation gradient（loss 對 residual stream 的梯度，第 20 層）與該特質 persona vector 的餘弦；
  - 在基礎模型上、用前 200 筆訓練序列，分別在防禦物件存在與不存在於前向時計算。
- **做到哪裡**：
  - 涵蓋 6 個特質 × 4 個模型的 step-0 量測；
  - 預測跨特質 PPS 的不對稱（evil 向量能防 sycophancy，反過來不行），並在訓練後確認；
  - 稱它為「有用、便宜的訓練前診斷，但有記錄在案的例外」。
- **自承的缺口（對本題重要）**：
  - 原文：「its size does not order the strength of the defenses」，也就是梯度位移大小無法排序防禦強弱。例如 Llama-3.1-8B 上 IP 位移小，防禦卻和 PPS 一樣好甚至更好；hallucination 位移大，PPS 卻幾乎無效。
  - IP 對梯度的改變是「diffuse」，在特質軸上只推到「接近 0」，所以沿 persona 軸的單一餘弦本來就不適合拿來預測 IP。
  - 機制分析只在 evil / Qwen2.5-7B / LoRA 上深入做。
  - 判讀依賴 LLM 評審。附錄 D 的五評審校準顯示排序穩健、絕對值隨評審而異；具體數字在 HTML 擷取時遺失。
- **與本題預測量的差別**：
  1. 它看「防禦情境下」的梯度沿 persona 方向的分量；本題要看「防禦情境下的參數更新」對「無提示 / 非誘發情境下特質機率」的一階影響，也就是跨情境轉移項。
  2. 它是 activation 空間、單層、單一方向；本題是參數空間，並限制在實際可訓練的參數上（LoRA 或層段）。
  3. 它主要診斷 PPS；本題要跨 IP、NIP、PPS、IA、SIP 和層限制，給出共同排序。
  - 結論：**預測量被「部分佔走」的疑慮成立**，但佔走的是「step-0 梯度診斷 PPS」這一塊；「跨防禦類型排序漏出」這一塊原作自承做不到。

### 1.3 2611.04260 NIP（Africa；Holmes，UK AISI）〔全文〕

- 設定：
  - 模型：Qwen3.6-27B、Gemma-4-31B、OLMo-3.1-32B；rank-32 LoRA，5,000 筆，3 個 seed。
  - 評審：gpt-6-luna-pro（透過 OpenRouter）。
  - 漏出量的定義沿用 SIP：六族非誘發系統提示的平均。六族是無提示、同結構良性指令、否定句、共享關鍵詞、鄰近領域、無關 persona。
- **附錄 G.7「Leakage tracks drift」**：
  - 無提示下，各 adapter 在答案開頭的狀態沿基礎模型 recklessness 方向的漂移，在 6 個「模型 × 任務」格子裡都和漏出呈 Spearman 正相關。具體數值在擷取時遺失。
  - 這是**訓練後**的事後量測，不是事前預測。
- **附錄 H（token 選擇）**：
  - MBPP 上，Qwen 的 4 個以相同方式訓練的 token，漏出量從 0.044 到 0.137；相比之下，微調 seed 之間的差距只有 0.014。
  - 作者試過一個「在凍結模型上量測、不需標籤的代理量」，盲測 sports / finance 的 token，6 格只對 4 格。
  - 也就是說，**事前預測已經有人試過，成績只算普通**。這正是本題可以切入的明確缺口：token 之間的差異比 seed 雜訊大約 7 倍〔由全文數字計算〕，是一個可以客觀判定的預測目標。
- 程式：論文中沒有程式連結；重現性聲明只指向附錄。〔全文〕

### 1.4 2609.35356 Stratified IP（SIP；Cambridge、KCL 等，作者包含 Gillioz、Tan、Riché）〔全文〕

- ICLR 2027 投稿 RBB7QhM9ZX〔API〕。
- 做法：用少量乾淨子集（最少到 1%），在多樣的非誘發提示下過採樣，其餘樣本照常 inoculate。
- 他們定義了「leakage」，量測方式就是六族非誘發提示，後來 NIP 沿用。
- 屬於啟發式方法，沒有預測理論。
- 程式：全文沒有看到公開連結。

### 1.5 2606.30252 Inoculation Adapters（IA；CLR 的 Riché、Tan 等）〔全文〕

- ICLR 2027 投稿 hz0bRtpaCi〔API〕。
- 規模：9 個設定、5 個模型家族。和 IP、PPS、CAFT、KL 正則化比較。
- **KL 正則化（在 OOD 中性資料上）整體最差**。這提醒本題：「加一個 KL 錨定項」很可能被審稿人視為已經被打敗的基線。
- 作者自承信賴區間很寬，改進幅度不確定。
- 程式：全文沒有看到公開連結。

### 1.6 2611.12864 Is Persona Generalization Predictable?（CMU：Mukherjee、Goel、Raghunathan、Springer）〔全文〕

- 程式公開：github.com/olimu/persona-generalization（HTTP 200）。同時也是 ICLR 2027 投稿 9SxsrLIsLR〔API〕。
- 評審：Qwen3.6-27B-FP8，貪婪解碼，三分類（persona 一致 / 不一致 / 退化）。
- on-the-line 現象：
  - ID 與 OOD 的 persona 率呈 probit 線性；
  - 學習率、權重衰減、訓練長度只會讓模型沿線移動；
  - **只改變訓練哪些層會改變斜率**：只訓練後段層時，unsafe 的斜率較平。
- 它沒有處理 inoculation 類防禦。對本題的用處：
  1. 評估時要在相同 ID 學習程度下比較，用來排除「只是學得少」的情況；
  2. 「層段改變斜率」是一個現成、可以拿來檢驗本題預測量的外部事實，見 §5。

---

## 2. 撞題地圖（引用者、新作與 ICLR 2027）

### 2.1 Semantic Scholar 引用〔API〕

- 引用數：2510.04340 有 46 篇，2510.05024 有 45 篇，聯集 60 篇。
- 2026-06 之後相關的引用者，其中最危險的是下表前三篇。

### 2.2 直接衝擊本題的工作

| 論文 | 狀態 | 佔走的部分 | 剩下的縫 |
|---|---|---|---|
| **2610.09384 PHM**〔全文〕 | ICLR 2027 投稿（xiuXr2UoH8）；程式公開 | 兩路徑理論（共享預設 persona 對局部 persona）；訓練前預測量（前綴 persona 與預設 persona 的距離，r = 0.72）；修正法 PPR（預設前綴下的 KL，等同無提示 KL 錨定） | 只處理「換系統提示前綴」這一類；作者自承距離只能預測「相對窄度」，無法校準；沒有涵蓋 token、steering、adapter、資料混合這幾類防禦，也沒有處理層限制 |
| **2611.10110 Homeopathic Learning**〔全文部分〕 | arXiv 2026-11-06；程式公開 | 兩特徵（任務、觸發）的 ridge regression 模型，解釋性質「歸因給觸發還是給任務」；包含 IP 加 School of Reward Hacks 的實驗 | 玩具級的定性理論，沒有對 LLM 給出定量預測量 |
| **2604.16423**〔全文〕 | 已被接受（會議未確認）；程式公開 | step-0 activation 梯度餘弦，用來診斷 PPS | 原作自承無法排序跨設定的防禦強度；IP 的訊號太分散 |
| 2611.04260 NIP G.7 / H〔全文〕 | arXiv | 事後「漂移追蹤漏出」；事前代理量 4/6 | 事前預測仍是開放問題 |
| 2510.05024 Wichers〔全文〕 | 已發表 | 誘發強度啟發式（r > 0.57，5 個設定中 4 個） | 只適用於提示類防禦 |
| AlignRadar（ICLR 2027 投稿 xmDdTDx241）〔摘要〕 | 投稿中 | 由資料集加基礎模型在 SFT 前預測 EM 風險（76 個語料、13 個家族） | 預測對象是「資料集風險」，不是「防禦漏出」 |
| 2610.14937 Propensity spillover〔摘要〕 | arXiv；程式公開 | 預測傾向的 OOD 外溢 | 沒有處理防禦 |
| Where to Anchor?（ICLR 2027，已撤稿 GO2FJjCWW7）〔TLDR〕 | 撤稿 | 「錨定用哪個提示分佈」；結論是匹配可塑性時，初始 checkpoint 最好 | 反面訊號：錨定項的設計空間已有人探索 |

### 2.3 OpenReview ICLR 2027 其他命中

用 inoculation 搜尋〔API〕，命中三篇：
- IA
- SIP
- 4Mg0HWnNSL「Technique Awareness」：模型越熟悉 IP，IP 越無效

用 emergent misalignment 搜尋，相關的有：
- What Survives Continued Training
- Narrow Neurons（已撤稿；做法是 activation anchoring）
- Intent Heads（在微調前識別 EM 路徑）
- Identifying the Effect of Training Context in EM
- See it, Say it, Sorted（參數空間投影，可減少約 80% EM）

ICLR 2027 裡**沒有**「預測 inoculation 漏出」的直接投稿。但 PHM 和 Persona Generalization 都在審，1 月底前會公開評審意見。

### 2.4 其他相關新作〔摘要〕

- 2610.21633 ContextLift：針對條件式 EM 的反事實增強，程式公開。
- 2611.08569 EMERGE：分解式 EM 基準。
- 2609.38379 Context confusion。

**撞題判斷**：「提出一個新防禦」或「無提示 KL 錨定」已經沒有新意。PHM 加上 IA 的 KL 基線結果，等於直接預告審稿人會怎麼質疑。「理論」若只停在兩路徑或兩特徵的定性模型，也已被 PHM 和 homeopathic 佔走。剩下可守的是：**一個不依賴防禦類型的定量事前預測量，加上它勝過四個既有預測量的證據**。

---

## 3. 評測可靠度

**LLM 評審的雜訊**
- 2604.16423 的五評審校準〔全文，數字遺失〕：arm 層級的排序在各評審之間一致；絕對值有評審相依的仿射偏移，而且偏移大小依 arm 而異。
- 結論：**秩相關型的主張（本題的預測量評估）比絕對漏出率穩健**。
- NIP：第二評審（Claude Opus 5.5）在 plausibility 上與主評審的一致率是 85%（240 筆）〔全文〕。
- 2611.08569：EM 率對評估問題集、評審、門檻都敏感，還會把能力下降和失準混在一起〔摘要〕。

**可以撐起主結果的客觀任務（全都有公開或可重建的資料）**

1. **MBPP hard-coding**：以執行隱藏測試判定。Wichers 有公開程式；NIP 也用了，而且 token 間漏出差 0.044–0.137 對 seed 差 0.014，訊雜比好〔全文〕。
2. **Spanish + ALL-CAPS 玩具**：用正則和語言判別即可。來自 Tan，Grant 附錄 H.3 也複製過〔全文〕。
3. **GCD 諂媚**：是否附和錯誤答案，可以直接比對〔全文，Wichers / NIP〕。
4. **CEBaB 虛假相關**：準確率〔全文，Wichers〕。

**建議**：主結果用 1–3。EM（Betley 問題集或 EMERGE）當作「延伸到評審型任務」的次要結果，並用兩個評審報告秩相關。

---

## 4. 最強反駁與回應

**反駁 1（最強）：PHM 已經給了理論、事前預測量（r = 0.72）和 KL 錨定修正；Grant 已經有 step-0 梯度診斷；Wichers 已經有誘發強度啟發式。這篇只是再換一個預測量。**

回應要靠一個「別人做不到、而且可以證偽」的主張。
- 現有預測量各自綁定一種防禦：
  - Wichers 和 PHM 需要一個「提示」；
  - Grant 需要一個 persona 向量，而且原作自承不能排序防禦。
- 本題的預測量定義在參數空間，量的是「防禦情境下的訓練梯度」對「非誘發情境下特質 log-prob 梯度」的內積，並限制在實際可訓練的參數上。
- 因此可以用**同一把尺**排序提示、token、steering、adapter、資料混合和層限制，並預測 2611.12864 的層段斜率差異。
- 主張寫成：「在跨類型的防禦集合上，Spearman 高於最佳既有預測量」。這可以證偽，失敗也有資訊量。
- **前提**：停損實驗要真的贏。如果贏不了，這個回應就不成立。

**反駁 2：一階 step-0 近似在訓練後期失效。**
- 依據：Grant 發現餘弦會隨訓練衰減；NIP 的漂移是事後才量到的。
- 回應：
  - 同時報告 k 步前瞻版本（k ≪ 總步數）以及「預測地平線」，也就是可預測性隨 k 的曲線。這本身就是結果。
  - 理論上可以用 NTK 或 lazy 近似界定什麼時候成立；LoRA 小學習率是有利的情形〔推估〕。

**反駁 3：LLM 評審雜訊。**
- 回應：主結果用 §3 的三個可程式判定任務，EM 放次要；所有主張都寫成秩相關。

**反駁 4：MATS / Anthropic / CLR 產出很快。**
- 這是事實：一年內有 IA、SIP、NIP、Grant 等約 10 篇，而且作者高度重疊（Tan、Riché、Gillioz）。
- 回應：避開「新防禦」這條擁擠的賽道，主打「評估工具 + 預測理論」。對方較少做這種分析型論文。
- 殘餘風險仍然高；1 月前出現「NIP token 事前選擇」後續工作的機率不低〔推估〕。

**失敗時能留下的正面貢獻**

1. 第一個在**同一套客觀指標、匹配 ID 學習程度**（on-the-line 軸）下，比較 6 種以上 inoculation 類防禦的統一測試平台與公開程式；目前各篇的設定互不相容。
2. 五種事前預測量的對決表：哪種防禦適用哪種預測量、何時失效。只要至少一個預測量在某類防禦上可靠，就是正面結論。
3. NIP「token 選擇」的事前篩選方法：即使跨類型失敗，只在 NIP 或 IP 提示選擇上超過 4/6 也有實用價值。

注意：如果 2 的結論是「全部都預測不了」，就屬於使用者不接受的純負面分析。所以停損判準要在第 2 週就把這種情況擋下來。

---

## 5. 建議切入角度（要升到中偏高必須這樣調）

**題目（建議）**：*One Ruler for All Inoculations: A First-Order Cross-Context Transfer Predictor of Leakage in Selective-Generalization Defenses*

1. **砍掉**：「新修正方法」不再當主貢獻，因為 PPR、KL、SIP、ContextLift 都已存在。「兩路徑定性理論」改成引用 PHM 和 homeopathic，自己只做定量化。
2. **主貢獻 A：預測量**
   - 符號：θ_T 是可訓練參數；L_def 是防禦情境下的訓練損失；τ_e 是第 e 族非誘發情境下的特質 log-prob。
   - 定義 Λ = −η·⟨∇_{θ_T} L_def, ∇_{θ_T} τ_e⟩，也就是一步 SGD 後非誘發特質 log-prob 的一階變化量。可以用 per-example 的 Gram 矩陣或隨機投影（TRAK 式）來估算。
   - 防禦只改變 L_def 的前向，例如加提示、加 token、加 steering、加 adapter、改資料混合；層限制只改變 θ_T。因此**一把尺涵蓋所有防禦**。
   - 理論貢獻：證明在線性化（NTK / LoRA lazy）條件下，漏出 ≈ Σ_t Λ_t；並推出「只訓練後段層 → 跨情境梯度內積較小 → 斜率較平」，可以和 2611.12864 的觀察對照。
3. **主貢獻 B：對決**
   - 在 Λ（step 0 與 k 步）、Wichers 誘發強度、Tan loss 下降、Grant 梯度餘弦、PHM persona 距離之間比較；
   - 目標：在「跨防禦類型」與「同類型內的 token / 提示選擇」兩個尺度上，以 Spearman 預測最終漏出。
4. **次要貢獻**：用 Λ 做事前篩選，挑 NIP token 或 IP 提示，直接回應 NIP 附錄 H 的開放問題（6 格對 4 格）。是否加入修正項（把 Λ 方向投影掉），要看停損結果再決定，而且只定位成推論。
5. **評估**：on-the-line 匹配 ID；主任務是 MBPP、玩具特質、GCD；EM 次要，用雙評審。

---

## 6. 前兩週停損實驗（GPU 小時全為〔推估〕）

### 第 1–3 天：零訓練檢驗（約 5 GPU 小時）

- 利用 2611.12864 的公開程式和模型設定，在其層段（chunk）設定上計算 Λ，不需要訓練，看 Λ 的排序是否和論文報告的層段斜率排序一致。
- 同時用 Grant 的公開程式，在其 9 個設定上計算 Λ，對照論文報告的 IP / PPS 防禦結果排序；這是原作自承無法排序的部分。
- **判準 0**：兩組外部結果中，至少一組的 Spearman ≥ 0.6。兩組都 < 0.3 就直接停，改走別題。

### 第 4–12 天：自建對決（約 60–80 GPU 小時；在 4 張 H200 上約 3 天）

- 模型：Qwen2.5-7B-Instruct（主）、Qwen3-4B（複驗），LoRA r = 32。
- 任務：MBPP hard-coding（Wichers 資料）、Spanish + CAPS、GCD 諂媚。
- 每個任務約 14 個 arm：
  - 無防禦；
  - IP 6 種提示（強、弱、無關、改寫）；
  - PPS 2 個強度；
  - SIP；
  - 只訓練後段層的 IP；
  - 簡化版 NIP（取 2 個 token；如果重建 neologism 訓練超過 2 天，就改用 random-row token 當代理）。
- 2 個 seed。合計約 3 × 14 × 2 ≈ 84 次訓練。
- 每次 7B LoRA（約 5k 筆、1 個 epoch）加 vLLM 評估約 0.6–0.9 GPU 小時〔推估〕。
- 預測量計算：每個 arm 約 0.1 GPU 小時（梯度內積、200 筆、隨機投影）〔推估〕。

### 停損判準（事先寫定）

- **判準 1（主）**：漏出量定義為六族非誘發提示平均減去無防禦基線，並在匹配 ID 學習程度下比較。
  - Λ（step 0 或 k ≤ 5% 總步數）與漏出的 Spearman ≥ 0.6，在 3 個任務中至少 2 個成立；
  - **而且**在跨類型合併的 arm 集合上，比最佳既有預測量高出 ≥ 0.15。
- **判準 2（同類型內）**：在 IP 提示選擇（6 個）上，Λ 的 top-1 / bottom-1 判對率 ≥ 5/6 格，高於 NIP 代理量的 4/6。
- **判準 3（雜訊底線）**：同一 arm 兩個 seed 的漏出差距，要小於 arm 間差距的 1/3；做不到就增加 seed 或換任務，否則秩相關沒有意義。

### 判定方式

| 結果 | 處置 |
|---|---|
| 判準 1 加判準 2 都通過 | 信心升到**中偏高**，第 3 週起擴到 EM 和 Llama / Gemma，並寫理論 |
| 只通過判準 1 的前半（Λ 可用，但沒有贏過基線） | 維持**中**；改主打「統一平台 + 對決」，要確認至少一個預測量可靠，避免變成純負面 |
| 判準 1 前半就失敗 | **停**。兩週總花費約 90 GPU 小時〔推估〕 |

---

## 7. 信心評級

| 版本 | 評級 | 主要理由 |
|---|---|---|
| 回報 50 原案 | 中偏低 | PHM（ICLR 2027 投稿）同時佔走理論、預測量和 KL 修正；IA 已證明 KL 正則化是最差基線；homeopathic 已有兩特徵模型 |
| §5 改切入 | 中 | 縫是真的：Grant 自承無法排序防禦；NIP 事前選 token 只有 4/6；Wichers 和 PHM 只適用於提示類；客觀任務和公開程式齊全（Wichers、Grant、2611.12864、PHM 都有）；計算便宜 |
| §5 加停損判準通過 | 中偏高 | 有一個可證偽的跨類型主張被實證，加上外部結果（層段斜率）佐證 |

仍然存在的風險：
- 同組作者群在 1 月前推出「NIP / IP 事前選擇」的後續工作〔推估〕；
- PHM 被 ICLR 接受後，審稿人可能認為 Λ 只是 PHM 的參數空間版本。對策：§6 的對決表必須直接列出 PHM 距離作為基線。

---

## 8. 沒查到或未確認的項目

- 2604.16423 被哪個會議接受：只看到「camera-ready」字樣。
- NIP、SIP、IA 的程式：全文中沒有看到連結，可能要向作者索取或自己重建。
- Tan et al. 正式 GitHub 是否存在：只有匿名連結。
- 多處 Spearman 和評審一致度的數值在 HTML 擷取時遺失（Grant 附錄 D、NIP 圖 20），需要看 PDF。
- AlignRadar、Where to Anchor 只讀了摘要或 TLDR。
