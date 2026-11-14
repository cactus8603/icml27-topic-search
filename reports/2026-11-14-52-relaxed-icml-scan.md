# 放寬條件後的 ICML 2025／2026 錄取論文延伸掃描（2026-11-14）

## 結論

- 從 ICML 2026 的 oral／spotlight（共 742 篇 spotlight，其中 168 篇排 oral 場次）與 ICML 2025 oral（240 篇）標題中初篩約 30 篇，對 10 篇查摘要與 Semantic Scholar 引用者，對 4 篇讀全文相關段落。
- 推薦 5 個延伸題，信心由高到低：
  1. **文字生圖模型的罕見失敗機率估計**（延伸 ICML 2026 oral「Rare Event Analysis of LLMs」與 spotlight「Estimating Tail Risks」）— 中偏高
  2. **擴散模型「偏差泛化」在條件生成與 CFG 下的行為，加上單模型偵測與停止準則**（延伸 ICML 2026 spotlight「Biased Generalization in Diffusion Models」）— 中偏高（偏下緣）
  3. **PoPE 推廣到 2D／3D 視覺（ViT、DiT 的解析度外推）**（延伸 ICML 2026 spotlight PoPE）— 中
  4. **把 VisualSwap 的診斷變成訓練方法：讓 VLM 的「再看一次」真的重看**（延伸 ICML 2026 oral）— 中
  5. **蒸餾推理模型的迴圈：針對循環狀態的訓練期修正**（延伸 ICML 2026 spotlight「Why Do Reasoning Models Loop?」）— 中偏低
- 查過但不建議：Neural Thickets（引用者 32 篇，已有至少 5 篇直接接手）、Equivalence of Context and Parameter Updates（已有「Transmuting prompts into weights」、PatchKV 等方法）、ICML 2025 oral 普遍已被大量接手（例如 Scaling Collapse 已有「Scaling with Collapse」延伸）。

## 方法與資料說明

- 本機官方會場資料 `dl/icml26.json`、`dl/icml25.json`：只用標題、主題、decision、eventtype。ICML 2026 檔案沒有摘要欄位，摘要改從 Semantic Scholar／arXiv 取得。
- ICML 2026 的標記方式：所有 oral 的 decision 都是「Accept (spotlight)」，另有 eventtype=Oral 的場次紀錄（168 篇）。下文「oral」指有 Oral 場次者，「spotlight」指只有 spotlight poster 者。
- 網路操作約 38 次（Semantic Scholar 標題比對 15、引用者 9、arXiv 查詢 9、arXiv 全文 4，含失敗重試）。
- 數字來源層級標記：〔全文〕讀過 arXiv HTML 原文；〔摘要〕來自論文摘要；〔API〕Semantic Scholar 引用數（查詢當日）；〔本機〕本機會場資料計算；〔推估〕我的估計，沒有查證。

---

## 候選 1：文字生圖模型的罕見失敗機率估計（信心：中偏高）

### 一句話題目
在文字生圖擴散模型（特別是做過概念抹除或安全微調的模型）上，用初始雜訊空間的罕見事件模擬，以遠少於暴力抽樣的樣本，估計「良性或已抹除提示產生不安全或被抹除概念」的機率，並給出可校準的誤差。

### 延伸的原論文
- Rare Event Analysis of Large Language Models，arXiv 2602.06791，ICML 2026 oral〔本機〕。摘要明說「outline extensions and applications to other models and contexts」〔摘要〕。引用者 7 篇〔API〕，全是語言模型。
- Estimating Tail Risks in Language Model Output Distributions，arXiv 2604.22167，ICML 2026 spotlight〔本機〕，程式碼公開（github.com/rangell/LMTailRisk）〔摘要〕。用 steering vector 造「不安全版本」當 importance sampling 的提議分布，約 500 個樣本估到 10^-4 量級，樣本數比暴力 Monte Carlo 少 10–20 倍〔摘要〕。限制段寫明需要白箱、依賴 LLM 評審、輸入集固定；future work 寫推理模型與多輪〔全文〕。引用者 10 篇〔API〕。

### 研究價值
- 概念抹除／unlearning 的評估目前多用少量 seed 的攻擊成功率。2609.37537（Weeding Out Bad Seeds，2026-09）指出「probabilistic forgetting」：被抹除的概念只在特定初始雜訊區域重新出現，並把「跨隨機初始化的重現率」降低 67.2%〔摘要〕。但「重現率多低才算安全」需要可信的低機率估計，目前沒人提供。
- 擴散模型用確定性 ODE 取樣時，輸出是初始高斯雜訊的確定函數，因此「失敗」等價於標準高斯空間中的一個區域。這正好可以套用結構可靠度領域成熟的工具（subset simulation、cross-entropy importance sampling、2610.27442 GeoIS 等），是 LLM 離散 token 空間沒有的乾淨結構，構成這個延伸自己的方法學洞見。
- 介於可信 ML、生成模型與機率方法之間，屬 ICML 主流。

### 切入角度（可當貢獻）
1. 問題設定：把 T2I 安全評估從「有沒有被攻破」改寫成「每個提示的失敗機率 p(prompt)」，並定義部署層級風險（提示分布上的平均與極值，沿用 Tail Risks 論文的極值理論部分）。
2. 方法：雜訊空間的 subset simulation／自適應 importance sampling，用概念分類器（如 NudeNet、CLIP 概念分數）的連續分數當 performance function 分層逼近；隨機取樣器的情況改用 SMC（參考 DireSMC）。
3. 白箱捷徑：用「抹除前的原模型」或負 CFG 當提議分布（對應 Tail Risks 的「不安全版本」），比較黑箱（只動雜訊）與白箱兩種設定。
4. 應用：對 ESD、UCE 等抹除方法與 2609.37537 的修正版，給出 10^-3 到 10^-6 的重現率曲線，揭示少量 seed 評估看不到的差異；再把估計器的高風險雜訊區域回饋到抹除訓練中（選做）。

### 最接近的工作與區隔
- 2609.37537：發現問題並修正訓練，評估應該是暴力抽樣〔摘要，推估評估方式〕；本題提供估計工具。
- 2610.08652 DireSMC（2026-10）：擴散模型罕見事件 SMC，但對象是氣候等科學模擬代理，事件是可解析放寬的集合〔摘要〕；本題的事件是分類器定義的語意概念，且看重安全評估情境。
- LLM 側的 ALE（2610.35472）、RareTrap（2609.33638）、Estimating Rare Events with Proper Evaluation（2607.18454）都只做文字〔API 標題、摘要〕。
- 兩次 arXiv 搜尋（T2I＋初始雜訊＋機率估計＋不安全）沒有找到直接做這件事的論文；但搜尋不完整，殘餘撞題風險中等。

### 最強反駁與回應
- 反駁：「只是把現成的罕見事件模擬套到擴散模型，方法沒新意。」
  回應：主張的新意是「雜訊空間把語意安全問題轉成高斯可靠度問題」這個橋接，加上概念分類器分數不平滑、多峰的實務困難（失敗區可能分散在多個小區域，正是 probabilistic forgetting 現象），以及用來比較抹除方法的新結論。若只做到套用，論文會偏弱，所以第 4 點的實證發現必須做出來。
- 反駁：「分類器不可靠，估的是分類器的錯誤率。」
  回應：估計目標明確定義為「分類器判陽性的機率」，再用小量人工或第二分類器抽查校正；這點和 Tail Risks 依賴 LLM 評審相同，屬已接受的作法。

### 前兩週停損實驗與 GPU 小時
- 第 1 週：SD 1.5 加 2–3 個公開抹除權重（ESD 等，是否都有公開權重屬〔推估〕），5 個提示，每組暴力抽樣 10^5 張當參考值。以 H200 每秒約 10 張（512 解析度、25 步、批次，〔推估〕）算，每組約 3 小時，合計約 40 GPU 小時。
- 第 2 週：實作 subset simulation 與 CE-IS，在相同組合上以 1/10 樣本量估計。約 20 GPU 小時。
- 停損條件：(a) 參考重現率全落在 10^-2 以上或為 0（問題沒有「罕見」可言）；或 (b) 估計器在 1/10 樣本下，超過 30% 的組合與參考值差距大於 2 倍。

---

## 候選 2：擴散模型「偏差泛化」在條件生成與 CFG 下的行為（信心：中偏高，偏下緣）

### 一句話題目
在 test loss 還在下降時，擴散模型已開始偏向訓練樣本（偏差泛化）。本題研究條件生成與 classifier-free guidance 是否提早或放大這個階段，並提出不需要兩個不相交資料模型的單模型偵測量與停止準則。

### 延伸的原論文
- Biased Generalization in Diffusion Models，arXiv 2603.03469，ICML 2026 spotlight〔本機〕。用兩個在不相交資料上訓練的模型（sample-split）比較輸出距離，發現偏差在 test loss 最小值之前就出現；機制歸因於「粗結構先以與資料無關的方式學到，細特徵後期依賴個別樣本」〔摘要〕。CelebA 實驗沿用 Bonnaire 等的公開程式碼，作者另釋出修改版與階層資料模型程式（github.com/davide-beltrame/biased-generalization）〔全文〕。
- 原文限制段：只研究標準 DDPM，沒處理其他生成模型類別〔全文〕。結論段：明確把「條件生成與 CFG 如何與偏差泛化交互作用、與資料萃取的關係」列為 future work，並推測 CFG 可能把模型導向過擬合的特徵〔全文〕。
- 引用者只有 5 篇〔API〕：兩篇是解析理論（2605.21402、2603.13419）、一篇推到離散圖擴散（2610.13786）、兩篇統計物理延伸。沒有人做條件生成、CFG 或方法。

### 研究價值
- 擴散模型的記憶與泛化是 ICML／NeurIPS 熱門主題。原論文的實務訊息是「用 test loss 早停不足以保護隱私」，但沒給替代方案。補上「實務上會用到的條件生成與 CFG」和「能用的偵測與停止準則」，就把分析變成方法。

### 切入角度（可當貢獻）
1. 現象：類別條件與文字條件模型中，偏差泛化的起點與幅度如何隨 CFG 強度、條件粒度（類別或標題）、每類樣本數改變。
2. 機制：沿用原文的階層資料模型（有 belief propagation 的精確分數可比），把條件變數加進生成樹，分析 guidance 如何放大後期學到的、依賴個別樣本的細特徵。
3. 方法：只用單一模型就能算的偏差代理量，例如訓練與留出樣本在中間噪聲水平的去噪誤差差距（原文 Figure 2 已顯示中間時間點差異最明顯〔全文〕），驗證它與 sample-split 指標的相關性，並據此提出「偏差感知早停」或依時間步調整的 guidance。
4. 推到 flow matching（原文限制段點名的空缺）。

### 最接近的工作與區隔
- 2411.16738「Classifier-Free Guidance inside the Attraction Basin May Cause Memorization」：處理的是逐字複製（明顯記憶）〔摘要〕；本題處理的是「還沒複製之前的偏差」，並在訓練動態上找起點。
- Bonnaire 等（Why Diffusion Models Don't Memorize，原文用其程式碼）：研究記憶出現的時間尺度，沒有條件與 CFG〔全文引用描述〕。
- Wen 等以文字條件雜訊量偵測記憶（〔推估〕，依記憶，未在本輪查證）：偵測逐字記憶的提示，不是訓練過程的偏差起點。

### 最強反駁與回應
- 反駁：「CFG 會放大記憶是已知的，條件版本只是預期中的結果。」
  回應：已知的是逐字複製。本題要回答 CFG 是否在 test loss 最小值之前就讓偏差起點提前，以及提前多少；若結果是「CFG 不影響起點、只影響幅度」，也是可用的結論，但要搭配貢獻 3 的偵測量才不會淪為純負面分析。
- 反駁：「32×32 灰階 CelebA 太小，結論推不到大模型。」
  回應：主實驗用 CIFAR-10／ImageNet-64 類別條件；再加一個小型文字條件 latent 模型的微調實驗（例如小資料 LoRA 微調）當外部效度。

### 前兩週停損實驗與 GPU 小時
- 第 1 週：用原作者程式碼重現 CelebA 的 sample-split 曲線（偏差起點早於 test loss 最小值）。小 U-Net（base width 32〔全文〕），約 20 GPU 小時〔推估〕。
- 第 2 週：CIFAR-10 類別條件，每類 N∈{100, 400}，兩份不相交資料，CFG 強度 {0, 1.5, 4} 於取樣時比較；另訓練一組 condition dropout 不同的模型。約 60 GPU 小時〔推估〕。
- 停損條件：不同 CFG 強度下，偏差起點的時間或幅度差異小於模型種子間的標準差。此時改以「單模型偵測量」為主軸；若第 3 週偵測量與 sample-split 指標的等級相關也低於 0.6，就放棄。

---

## 候選 3：PoPE 推廣到 2D／3D 視覺（信心：中）

### 一句話題目
把「位置與內容解耦」的 PoPE 推廣到影像與影片的多軸位置，檢查 RoPE 的 what-where 糾纏在視覺定位任務與解析度外推中是否是真正瓶頸。

### 延伸的原論文
- Decoupling the "What" and "Where" With Polar Coordinate Positional Embeddings（PoPE），arXiv 2509.10534，ICML 2026 spotlight〔本機〕。實驗只有音樂、基因、OpenWebText 語言模型與長度外推〔全文目錄〕；程式碼公開（github.com/agopal42/pope，含 Triton kernel）〔全文〕。
- 引用者 13 篇〔API〕，沒有人把 PoPE 用在 ViT 或 DiT。最近的是 What-Where Transformer（2605.12021，slot 架構，不是位置編碼）、HD-RoPE（2608.29715）、DaRoPE（2609.34556，非文字基準）〔摘要〕。一次 arXiv 搜尋（polar coordinate＋position embedding＋transformer）只找到 PoPE 本身。

### 研究價值
- 視覺的位置編碼很活躍（RoPE-Mixed、VideoRoPE 為 ICML 2025 oral〔本機〕、Phase-Aligned RoPE 2511.19778 等），而且高解析度外推是實際痛點。PoPE 的論點（RoPE 讓內容改變位置調諧）在偵測與分割這類「純靠位置」的任務上應該更明顯，可以直接檢驗。

### 切入角度（可當貢獻）
1. 2D／3D PoPE 的構造（各軸頻率分配、軸向混合），並證明保留相對位置性質。
2. 視覺版 what-where 診斷任務（例如依位置索引 patch、依內容索引位置），重現原文 indirect indexing 的落差。
3. ViT 分類與密集預測的解析度外推（224 訓練、384／512 測試），以及小型 DiT 的高解析度生成。
4. 頻率使用分析（原文 Figure 3 的視覺版），解釋為什麼外推改善或不改善。

### 最強反駁與回應
- 反駁：「把 X 搬到模態 Y，增量。」
  回應：貢獻 2 與 4 要提供視覺特有的新發現（例如 2D 中 what-where 糾纏集中在低頻軸、與解析度外推失敗的因果關係）。只拿到 +0.3% top-1 的話不值得投。
- 反駁：視覺已有很多 RoPE 變體，基線很強。
  回應：基線至少要含 RoPE-Mixed 與一個 2025–2026 年的變體，並在相同訓練預算下比較。

### 前兩週停損實驗與 GPU 小時
- 第 1 週：ViT-S/16 於 ImageNet-100（或 IN-1k 100 epoch 縮短版），RoPE-Axial、RoPE-Mixed、PoPE-2D 三組，測 224→384／512 外推。約 60 GPU 小時〔推估〕。
- 第 2 週：視覺 what-where 診斷任務加上 DiT-S/2 小規模生成外推。約 80 GPU 小時〔推估〕。
- 停損條件：外推與診斷任務都沒有超過 RoPE-Mixed 一個種子標準差以上的優勢。

---

## 候選 4：讓 VLM 的「再看一次」真的重看（信心：中）

### 一句話題目
以影像替換（VisualSwap）建構訓練訊號，透過 SFT 或 RL 教 thinking VLM 在自我反思時重新讀取視覺證據，同時不讓模型學成「一律懷疑影像被換掉」。

### 延伸的原論文
- Are VLMs Seeing or Just Saying? Uncovering the Illusion of Visual Re-examination，arXiv 2605.15864，ICML 2026 oral〔本機〕。VS-Bench 800 對影像；替換影像後準確率最多掉 60%，thinking 模型的脆弱度約為 instruct 的 3 倍，放大規模沒有幫助〔摘要〕。程式與資料公開（visualswap.github.io）〔摘要〕。
- 原文只做了免訓練的「放大影像 token 注意力」概念驗證；future work 明寫「把多輪視覺驗證資料放進 SFT／RL」與「以視覺注意力強度當輔助獎勵」〔全文〕。
- 引用者 19 篇〔API〕。最接近的接手是 Recompute or Reuse?（2608.01930），它提出免訓練的 FSAF 注意力隔離〔摘要〕；What Keeps VLMs Looking at the Image?（2607.12815）是分析〔摘要〕。沒看到以訓練解決的論文，但「thinking with images」與「回看」類 RL 方法很多，間接競爭擁擠。

### 研究價值
- 「推理過程中的視覺 grounding 是否為真」是多模態推理的核心可信度問題。原文是 oral，future work 明確，接手正當性高。

### 切入角度（可當貢獻）
1. 從訓練集（非測試集）自動造替換對，配對設計包含「換」與「不換」，避免模型學成一律改答案。
2. 以替換偵測加上答案正確性當可驗證獎勵做 GRPO；比較加入與不加入注意力輔助獎勵。
3. 評估：VS-Bench、原始基準（不能退步）、與 FSAF 等免訓練法比較，並分析訓練後注意力是否真的回到影像 token。

### 最強反駁與回應
- 反駁：「模型只是學會偵測替換的捷徑（像素統計），沒有真正重看。」
  回應：用語意替換但視覺近似的配對（原文的設計），並報告未見過的替換類型與自然反思情境（不插入替換）下的表現。
- 反駁：已有大量 look-back／visual reflection RL 方法。
  回應：差異在於用反事實替換提供「重看是否發生」的直接監督，而不是只看最終準確率；須在相關工作中逐一對照，撞題風險中。

### 前兩週停損實驗與 GPU 小時
- 第 1 週：在 Qwen3-VL 小型 thinking 版本重現 VS-Bench 落差（小型 thinking 版本的可用性屬〔推估〕）。推論約 10 GPU 小時。
- 第 2 週：SFT 先導，約 5k 對替換資料，2B 到 4B 規模。約 60–100 GPU 小時〔推估〕。
- 停損條件：VS-Bench 改善小於 10 個百分點，或「不換」條件下的答案變動率上升超過 5 個百分點（代表學到的是懷疑偏誤）。

---

## 候選 5：蒸餾推理模型的迴圈，針對循環狀態的訓練期修正（信心：中偏低）

### 一句話題目
依原論文「學習誤差導致迴圈」的機制，在蒸餾小模型上找出容易進入循環的狀態，對這些狀態做針對性的 on-policy 教師修正，降低低溫度下的迴圈率與冗長度。

### 延伸的原論文
- Wait, Wait, Wait... Why Do Reasoning Models Loop?，arXiv 2512.12895，ICML 2026 spotlight〔本機〕。發現蒸餾學生即使教師很少迴圈也會大量迴圈，並提出兩個機制（難學動作的風險趨避、時間相關的錯誤偏誤）；溫度只是權宜之計，結尾只「討論」訓練期介入〔摘要〕。
- 引用者 29 篇〔API〕。相關接手：ProD（2610.19439）針對小模型 agent 的停滯做 progress-aware 蒸餾〔摘要〕；2610.24715 從解碼器角度分析迴圈鎖死〔摘要〕；2606.00206 量化後推理變長，以 logit 懲罰處理〔摘要〕。數學推理的訓練期修正仍空，但周邊擁擠。

### 最強反駁與回應
- 反駁：on-policy 蒸餾本來就能修正分布偏移，這只是 on-policy distillation 的特例。
  回應：必須證明「只在循環狀態修正」在相同教師 token 預算下明顯勝過均勻的 on-policy 蒸餾；做不到就沒有貢獻。這也是信心只有中偏低的原因。

### 前兩週停損實驗與 GPU 小時
- 第 1 週：R1-Distill-Qwen-1.5B／7B 在 AIME、MATH 上量溫度 0 與 0.6 的迴圈率，找出循環起點的狀態特徵。約 20 GPU 小時〔推估〕。
- 第 2 週：小規模針對性修正與均勻 on-policy 蒸餾比較。約 80 GPU 小時〔推估〕。
- 停損條件：在相同教師預算下，迴圈率的降幅沒有比均勻 on-policy 蒸餾多 30% 以上。

---

## 查過但不建議

| 論文 | 會議 | 不建議的理由 |
|---|---|---|
| Neural Thickets（2603.12228） | ICML 2026 spotlight | 引用者 32 篇〔API〕，已有 Revisiting Neural Thickets（2611.01262）、Rethinking … Without Weight Perturbations（2610.40649）、Cross-Task Merging in Neural Thickets（2611.11177）、Adapter Thickets（2610.00991）、Modular Norm RandOpt（2609.25745）等直接接手 |
| Equivalence of Context and Parameter Updates（2511.17864） | ICML 2026 oral | 「把 context 轉成權重」的方法已有 Transmuting prompts into weights（2510.08734）、PatchKV（2609.39329）〔API 標題〕 |
| Error Propagation … Quantized Diffusion（2508.12094） | ICML 2026 oral | 延伸到 flow／影片模型屬增量，研究價值偏低 |
| Language Model Circuits Are Sparse in the Neuron Basis（2601.22594） | ICML 2026 spotlight | 推到視覺可行，但沒有查引用者，本輪未深查（引用數 28〔API〕） |
| Base Models Know How to Reason（2510.07364） | ICML 2026 spotlight | 引用數 34〔API〕，模型差分方向擁擠，未深查 |
| ICML 2025 oral 整體 | ICML 2025 | 已過約 16 個月，常被大量接手（例如 Scaling Collapse 2507.02119 引用者 26 篇〔API〕，含 Scaling with Collapse 2509.25087）；Masked diffusion、Emergent Misalignment、TarFlow 等更擁擠 |

## 沒能確認的事項

- 候選 1：公開的概念抹除權重有哪些、能否直接下載；2609.37537 的評估方式是否已含機率估計（只讀摘要）。
- 候選 2：Wen 等的記憶偵測工作是依記憶提及，本輪沒查證。
- 候選 4：Qwen3-VL 是否有 2B／4B thinking 版本。
- 所有 GPU 小時都是〔推估〕，沒有實測吞吐量。
- ICML 2026 本機資料沒有摘要，初篩只看標題，可能漏掉標題不顯眼的好題。
