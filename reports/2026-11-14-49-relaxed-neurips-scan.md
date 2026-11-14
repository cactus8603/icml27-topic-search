# 放寬條件後重掃 NeurIPS 2026（主會 oral／spotlight 加 workshop）

日期：2026-11-14。沒有用 GPU，也沒有寫實驗程式。
適用條件：使用者在本輪中途兩次放寬。
1. 「CV 相關也可以，不限制任何方向」。
2. 「只要有研究價值，信心高的都可以，不用要贏 SOTA」。

其餘條件不變：ICML 會收、不太冷門、撐得起完整研究、不能是純負面分析、10 週內用 1–4 張 H200 做完。
本報告不列作者姓名或聯絡資訊，論文一律以 arXiv 編號或標題稱呼。

## 來源層級標記
- 〔全文〕：用 curl 抓 arXiv HTML 轉成純文字，自己讀相關段落並做關鍵字搜尋，不是逐頁精讀。
- 〔摘要〕：arXiv API、Semantic Scholar、OpenReview 回傳的原始摘要，或回報 36 快取的 neurips.cc 摘要。
- 〔搜尋摘要〕：只看到搜尋引擎整理的內容。
- 〔API〕：GitHub、Semantic Scholar 引用清單、OpenReview 搜尋的回傳。
- 〔本機〕：本機快取檔案。包括回報 36 的 467 篇 oral／spotlight 摘要，以及另一個子代理抓的 ICLR 2027 投稿 14,245 篇（標題與摘要）。
- 〔推估〕：我自己的估計，或憑記憶、未在本輪核實的內容。

網路操作約 45 次，分配如下：
- Semantic Scholar：約 12 次，其中數次回 429。
- arXiv API：約 12 次。
- arXiv HTML 全文：3 篇。
- GitHub：5 次。
- OpenReview 搜尋：1 次。
- 網路搜尋：2 次。

467 篇主會摘要沿用回報 36 的本機快取，沒有重新抓。

---

## 結論

依最新條件（研究價值加上做得出來、中稿把握高，不必贏 SOTA），提 3 個候選。

| 排序 | 候選 | 類型 | 信心 | 前兩週 GPU 小時〔推估〕 |
|---|---|---|---|---|
| 1 | 能量分布塑形的合成資料視覺預訓練：補上原作延後的「等量預訓練」因果驗證，並把篩選改成生成時直接塑形 | 視覺表徵（CV）＋資料為中心 | **中偏高**（第 2 週停損通過才算） | 80–120 |
| 2 | 多模態大模型的連續數值動作：從單峰高斯走向混合或多峰的座標策略，並修正「離散加連續」混合動作的 GRPO | 多模態推理＋RL 方法 | 中 | 100–200 |
| 3 | 對「學會躲監測器」的模型做防禦：利用逃避機制的低維結構訓練監測器，並偵測逃避本身 | LLM 安全／可解釋性 | 中 | 30–60 |

**為什麼第 1 名能到中偏高：**
- 原作是 NeurIPS 2026 oral，arXiv 2611.11962 在 2026-11-09 才上線，Semantic Scholar 尚無引用者。
- 作者在全文明寫「matched pretraining experiments are deferred to future work」。也就是說，篩選的增益只用離線代理指標驗證過〔全文〕。
- 程式、metric 套件與 checkpoint 都已公開〔API：GitHub README〕。
- 原作的預訓練規模小，全在 V100 上跑。主要的密度模型是 grayscale ImageNet64，n=9 個資料集〔全文〕。4 張 H200 足以把規模和因果實驗都做得更完整。
- 失敗風險主要在「效果量」，不在「做不出來」。

**第 2、3 名停在「中」的原因：**
- 第 2 名：混合動作 RL 已有 Pose-RFT（2508.07804）、HyLaR（2604.20328）等相鄰工作。
- 第 3 名：防禦題天生有「自適應攻擊者再贏一次」的反駁。而且 ICLR 2027 投稿 *The Monitorability Geometry of Language Models*（OpenReview Dpwe23TWWq）已提出「最佳監測子空間」〔本機〕，部分重疊。

這次沒有找到第二個「中偏高」的題目。

---

## 一、廣掃範圍與篩法
1. **主會**：用回報 36 快取的 467 篇 oral／spotlight（132 oral、335 spotlight），逐一看標題〔本機〕。
   - 因為已取消 CV 限制，這次 CV 類也看。
   - 挑出約 60 篇讀摘要〔摘要〕，再挑 9 篇查引用者或全文。
2. **arXiv 近期上線的 NeurIPS 2026 oral／spotlight**：查詢 comment 含 "NeurIPS 2026" 且含 oral 或 spotlight，依日期取 150 筆。用途是找「剛公開、還沒人跟進」的論文〔API〕。
3. **Workshop**：
   - 官方部落格稱有 102 個 workshop〔搜尋摘要〕。
   - 用 OpenReview API 在 `NeurIPS.cc/2026/Workshop` 群組做主題搜尋，以安全／監測為例：375 筆命中〔API〕。
   - 另外從上述 arXiv 清單看到 workshop oral／spotlight。
   - 只做抽樣，沒有逐一看完 102 個 workshop。
4. **撞題檢查**：
   - 本機 ICLR 2027 投稿用 regex 搜。
   - Semantic Scholar 查引用清單。
   - arXiv 關鍵字搜尋 2026 年 9–11 月的論文。

---

## 二、候選 1（建議主線）：能量分布塑形的合成資料視覺預訓練

**一句話題目**：合成影像能不能訓練出好表徵，取決於它在真實影像密度模型下的能量分布。本題要用等量預訓練證明這是因果關係，並提出在生成時直接把能量分布塑形成目標分布的方法。

### 起點：原作
NeurIPS 2026 oral，2611.11962，*Shaping Useful Noise: Energy Distributions Predict Visual Pretraining Quality*。

- **方法**：用標準化的真實影像密度模型，給每張合成影像一個能量值（負 log 機率）。再用 Wasserstein-1 距離（ELD）或百分位版本（pELD），比較合成資料與真實資料的能量分布〔全文、README〕。
- **相關性**：在 DINO／ViT-S 的自監督設定中，ELD 與 ImageNet-1K 凍結線性探測準確率的 Spearman ρ = −0.87（p = 0.002）〔全文〕。
  - 「All sources」這一列標示 n = 9〔全文〕。
- **另一個設定**：Learning-to-See（Baradad 等 2021）的小規模合成資料，用 ImageNet-100 評估協定〔全文〕。
- **開放缺口**：
  - 作者原文：「These are offline proxy results; matched pretraining experiments are deferred to future work to further establish a transfer gain caused by the intervention」〔全文，Section 5〕。所以「用 ELD 篩選或重抽樣能提升遷移」這件事，在 curation 段落只有 NN 距離、有效類別數這類離線代理指標。
  - 密度模型：主圖用 grayscale ImageNet64 密度模型〔全文，Fig. 2 圖說〕。
  - 算力：全部跑在 V100〔全文，附錄 B〕。
- **程式**：github.com/chinglamchoi/shaping-useful-noise，最後推送 2026-11-08。內容包括 pELD 套件、重現主實驗與 rebuttal 實驗的腳本，以及 checkpoint 設定說明〔API：README〕。

### 切入角度與貢獻
1. **因果驗證**（補原作延後的實驗）：
   - 從同一個來源池，在相同張數與相同 FID、recall 區間下，構造 ELD 高低不同的子集。
   - 各自做等量預訓練，用 Learning-to-See 協定加 DINO／ViT-S，至少 3 個種子。
   - 用偏相關控制 FID、LPIPS、recall、功率譜斜率。
   - 這是原作明說沒做的部分，結果正負都有資訊。純負面結果則要搭配下面的方法才成文。
2. **方法：生成時塑形，不是事後篩選**：
   - 把「讓能量分布貼近真實分布」當成生成器的目標。
   - 程序式生成器（dead leaves、隨機 StyleGAN、分形）：直接最佳化少數參數，讓 pELD 最小。
   - 擴散生成器：用溫度、CFG、雜訊排程做能量條件化取樣。
   - 預期的賣點是「同樣張數、不丟樣本」。事後重抽樣會丟掉大量樣本，原作的 curation 是「equal selected-set budget」下從池子裡挑〔全文〕。
3. **理解：能量對應到表徵學到什麼**：
   - 把能量層級對應到稀疏／紋理這類影像統計量。原作引用 Guth et al. 2026，指出 ImageNet 能量近似 Gumbel 分布，低能量是簡單場景、高能量是紋理〔全文〕。
   - 再看不同能量帶分別貢獻哪些下游能力，例如形狀或紋理偏好、ImageNet-C／A 穩健性。原作已報 ImageNet-C、A 的相關性〔全文〕。
4. **（可選）擴展**：
   - 換成彩色、較高解析度的密度模型，檢查 ELD 的結論是否穩定。
   - 從純合成預訓練延伸到「合成暖身加真實資料」，連到 2511.13945 的 procedural warm-up。

### 最接近的工作與區隔
| arXiv | 內容 | 區隔 |
|---|---|---|
| 2611.11962 | 原作。ELD 當診斷指標，加上事後重抽樣 | 本題做等量預訓練的因果驗證、生成時塑形、機制解釋 |
| 2106.05963（Learning to See by Looking at Noise）〔推估：編號憑記憶〕 | 程序式雜訊影像預訓練，比較各種生成器 | 沒有密度或能量的觀點 |
| 2511.13945 *Can You Learn to See Without Images? Procedural Warm-Up for ViT*〔API：arXiv 搜尋〕 | 程序式資料當 ViT 暖身 | 沒有能量分布匹配 |
| 2506.19465 *Stylized Structural Patterns for Improved Neural Network Pre-training*〔API：arXiv 搜尋〕 | 結構化合成圖樣預訓練 | 同上 |

撞題檢查：
- ICLR 2027 投稿用 regex 搜「likelihood／energy／density + synthetic images」：0 筆〔本機〕。
- 搜「synthetic／procedural + pretrain」：沒有相關的視覺預訓練論文〔本機〕。
- arXiv 2026 年 9–11 月的 cs.CV 程序式／合成預訓練搜尋：沒有看到能量相關論文〔API〕。

### 最強反駁與回應
- **反駁 1**：「能量只是影像複雜度或功率譜斜率的代理，Baradad 早就知道要配頻譜。」
  - 回應：主實驗加入「功率譜匹配」和「通用特徵密度能量」兩個基線。原作附錄 B.1 已有「Generic feature-density energies are not substitutes for ELD」的初步證據〔全文，小節標題〕。
  - 我們要做的是在控制頻譜後的偏相關與介入實驗。如果偏相關消失，論文轉成「頻譜才是因」的正面方法：以頻譜塑形的生成器。仍是方法貢獻，但信心會降到中。
- **反駁 2**：「純合成預訓練的天花板低，ICML 不在意。」
  - 回應：同時報「合成暖身加少量真實資料」的設定。不追 SOTA，主張放在「哪些合成資料性質有用」的可控科學問題加上一個可用工具，符合使用者最新條件。
- **反駁 3**：「原作團隊自己會做續作。」
  - 回應：這是最大的不確定性，無法排除。緩解方式是把重心放在生成時塑形與機制解釋，不只是補做原作延後的等量預訓練實驗。

### 前兩週停損實驗
- **第 1 週**：
  - 用原作 repo 重算 4–6 個合成來源的 pELD，對上原作的表值。
  - 選一個程序式生成器家族，做參數掃描：同張數、FID 相近，pELD 分成高、中、低三檔。
- **第 2 週**：
  - 三檔各做 Learning-to-See 協定的等量預訓練，3 個種子。原作的 Table 3 設定，ResNet 加對比學習，ImageNet-100 線性評估。
  - 加一組「頻譜匹配、pELD 不同」的對照。
- **停損條件**（任一成立就停）：
  - 低 pELD 檔對高 pELD 檔的線性探測差距不到 1.0 個百分點，或不超過種子間標準差的 2 倍。
  - 控制頻譜後，pELD 與準確率的偏相關 |ρ| 小於 0.4。
- **GPU 小時**：單次小規模預訓練約 4–8 H200 小時〔推估〕。總計 9 次預訓練加 3 次對照加打分，約 80–120 H200 小時〔推估〕。

**信心：中偏高**，以第 2 週停損通過為前提。
- 做得出來：高。程式公開、規模小、不依賴大模型。
- 中稿：中到中偏高。這題是 oral 的直接後續，補上原作明說延後的實驗，再加上新方法。

---

## 三、候選 2：多模態大模型的連續數值動作，從單峰高斯到多峰策略與混合動作 GRPO

**一句話題目**：NV-CoT 讓多模態大模型直接輸出連續座標當作推理動作，但用的是單峰、共享 σ 的高斯策略。本題研究這個策略在多目標或歧義區域時的失效，並提出多峰座標策略，以及「離散 token 加連續座標」混合動作下一致的 GRPO 比率與裁剪。

### 起點：原作
NeurIPS 2026 oral，2602.23959，*Thinking with Images as Continuous Actions: Numerical Visual Chain-of-Thought*。

- **策略設計**：用 4 個線性頭輸出 bbox 平均，第 5 個頭輸出共享的 σ，σ 取 max(·, ε)。另有 Laplace 變體。用重參數化取樣，所以 GRPO 的重要性比率可以閉式計算〔全文，Section 3〕。
- **RL 規模**：Qwen2.5-VL-7B，GRPO 80 次迭代，32 張 A100，batch 256，G = 16〔全文，4.1〕。
- **結果**：在 V* 上，比 LVR-7B 高 7.8%、比 DeepEyes-7B 高 2.6%〔全文〕。
- **程式與權重**：github.com/kesenzhao/NV-CoT。已支援 Qwen3-VL-2B／8B，HF 上有 2B 與 8B 的 Joint-GRPO checkpoint。訓練資料用 DeepEyes-Datasets-47k，README 寫「Added more numerical-head designs」〔API：README〕。
- **引用者**：Semantic Scholar 只有 5 篇，沒有一篇在做多峰或連續動作 RL 的延伸〔API〕。

### 切入角度與貢獻
1. **診斷**：
   - 單峰策略在「多個合理區域」或「目標很小」時，平均值會落在兩個區域中間，GRPO 訓練過程中 σ 可能塌縮。
   - 用公開的 2B 與 8B checkpoint，先量失敗案例裡這兩類的比例，以及 σ 的軌跡。
   - 這一步只是動機，不是主貢獻。
2. **方法**：
   - 換成混合高斯，或輕量 flow 座標頭，讓模型能同時保留幾個候選區域。
   - 推導混合動作的 GRPO 比率：token 機率比與連續密度比混在同一個裁剪下，尺度不一致〔推估：這是我的分析，原文沒提〕。
   - 據此提出分項裁剪，以及 σ 與熵的控制。
3. **泛化**：把「數值動作」延伸到計數（點集）與影片時間定位（時間戳）。用同一個框架說明哪些數值適合連續頭、哪些適合文字 token。
4. **定位**：照使用者最新條件，不以打贏 DeepEyes 等 SOTA 為目標，主張放在「混合動作 RL 的正確做法加可用的多峰頭」。

### 最接近的工作與區隔
| arXiv | 內容 | 區隔 |
|---|---|---|
| 2602.23959 | 原作。單峰高斯或 Laplace，共享 σ | 本題做多峰策略，並修正混合比率與裁剪 |
| 2508.07804 Pose-RFT〔API：arXiv 搜尋，只看到標題〕 | 多模態大模型產生 3D 姿態的混合動作 RL 微調 | 任務是姿態生成，不是推理過程中的視覺動作；是否處理比率尺度問題未查 |
| 2604.20328 HyLaR〔API：只看到標題〕 | 混合潛在推理，解耦的策略最佳化 | 潛在 token，不是數值座標動作 |
| 2609.36116 Dyad〔摘要〕 | 給 LLM 加上原生的「型別化」離散動作打分 | 動作是離散候選，不是連續數值 |

### 最強反駁與回應
- **反駁 1**：「Qwen3-VL 用文字座標已經很強，連續頭的增益本來就小（V* 對 DeepEyes 只差 2.6%），再改成多峰只是邊際改進。」
  - 回應：主張放在混合動作 RL 的方法論，以及多峰帶來的可解釋候選，不是靠分數取勝。
  - 若診斷顯示多峰或 σ 塌縮的案例很少（見停損），就不做這題。
- **反駁 2**：「Pose-RFT 已經做過混合動作 RFT。」
  - 回應：要在第 1 週讀全文確認它有沒有處理比率尺度與裁剪。若已處理，貢獻縮成「多峰頭加視覺推理場景」，信心降到中偏低。

### 前兩週停損實驗
- **第 1 週**（只推論）：
  - 用公開的 NV-CoT 2B checkpoint 在 V* 與 HR-Bench 上做推論，標註失敗案例中「多個合理區域」與「σ 異常」的比例。
  - 讀 Pose-RFT、HyLaR 全文。
- **第 2 週**：
  - 在 2B 上把頭換成 K = 3 的混合高斯，跑縮小版 GRPO，例如 40 次迭代，batch 128。
  - 對照組是原版單峰。
- **停損條件**（任一成立就停）：
  - 失敗案例中多峰類不到 15%。
  - 混合頭在 V* 上沒有至少 1.5 個百分點的改善，也沒有可量化的 σ 塌縮改善。
- **GPU 小時**：推論約 10 小時，2B GRPO 兩組約 100–200 H200 小時〔推估：依原作 32 A100 跑 7B 等比縮放，不確定性高〕。

**信心：中。**
- 做得出來：中偏高，程式與權重齊全。
- 中稿：中。「thinking with images」很熱門，這是 oral 的後續，但切入角度要靠第 1 週的診斷撐住。

---

## 四、候選 3：對「學會躲監測器」的模型做防禦

**一句話題目**：Neural Chameleons 顯示，模型可以學會零樣本躲過沒見過的激活監測器，而逃避機制是把激活推進一個低維子空間。本題利用這個結構，提出對逃避穩健的監測器，以及不依賴觸發詞、可偵測「模型正在做自我混淆」的偵測器，並用自適應攻擊者評估。

### 起點：原作
2512.11949，*Neural Chameleons*。

- **場次待確認**：
  - 回報 36 快取的 NeurIPS 2026 JSON 把它標成 Oral〔本機〕。
  - 但網路搜尋結果裡有 iclr.cc/virtual/2026 的頁面〔搜尋摘要〕。
  - 我沒有進一步確認是哪一個場次，或兩者的關係。
- **原作結果**：
  - 4 層、8 層 ensemble 與 MLP、attention 監測器「mitigate but do not eliminate evasion」〔全文〕。
  - 機制是「moves activations into a low-dimensional subspace」〔全文〕。
  - Limitations 段寫的未來方向是：更大的模型、自然出現的路徑、更進階的監測器（例如讀 attention head）〔全文，5.1〕。
- **程式與權重**：github.com/mgm52/neural-chameleons，MIT 授權。另有 minimal reproduction repo 提供既有 checkpoint。模型包括 Gemma-2-2b／9b-it、Llama-3.1-8B-Instruct、Qwen-2.5-7B-Instruct〔API：README；全文〕。

### 切入角度與貢獻
1. **逃避子空間擴增訓練**：
   - 用良性概念的逃避樣本估計逃避子空間。
   - 訓練監測器時，沿這個子空間做對抗擴增，或讓監測器對這個子空間的投影不變。
   - 目標是讓監測器在沒見過的安全概念上，觸發逃避時也能維持 TPR。
2. **不靠觸發詞的自我混淆偵測器**：
   - LessWrong 的一篇文章回報，簡單分類器在新概念上全部失效，原因是它們學到的是觸發詞，不是逃避本身〔搜尋摘要〕。
   - 本題用改寫或隱式觸發，加上不同概念交叉留出的方式訓練，正面解決這個問題。
3. **自適應攻擊者評估與成本曲線**：攻擊者知道防禦後再訓練，畫出「逃避率對能力稅」的權衡曲線，把防禦的價值定位成提高攻擊成本。

### 最接近的工作與區隔
| 編號 | 內容 | 區隔 |
|---|---|---|
| 2512.11949 | 原作。攻擊加上簡單防禦：ensemble、非線性 | 本題做利用機制的防禦，加上偵測器 |
| 2412.09565 *Obfuscated Activations Bypass LLM Latent-Space Defenses*〔搜尋摘要〕 | 多種潛在空間防禦都不穩健；生成時的探測與多層探測較好 | 不是針對學到的零樣本逃避 |
| 2609.36490 *LLMs Learn to Evade Latent Monitors from Prior Feedback Alone*〔摘要〕 | 又一種攻擊：rank-1 LoRA 讓 TPR 降到 4%，重訓監測器後仍有效 | 攻擊面，可當我們的自適應攻擊者 |
| ICLR 2027 投稿 Dpwe23TWWq *The Monitorability Geometry of Language Models*〔本機〕 | 用 Jacobian 給出線性監測器盲區的上界與最佳監測子空間（Fisher 子空間） | **重疊最大**：「讀行為相關方向」這條防禦路線已經有理論。本題要在 Neural Chameleons 這類學到的逃避上實測它，並加入擴增與偵測器 |
| ICLR 2027 投稿 MMkYLPv7YO *Adaptive Attacks on Model-Diffing Audits*〔本機〕 | 比對基礎模型與微調模型的稽核也能被躲過 | 排除「直接用 model diffing 當防禦」這個捷徑 |

撞題檢查：Semantic Scholar 引用者 13 篇，全部是攻擊、分析或相鄰主題，沒有專門的防禦〔API〕。arXiv 搜尋「obfuscation／evasion + probe + defense」也沒有找到〔API〕。

### 最強反駁與回應
- **反駁**：「這是貓捉老鼠。2412.09565 已經顯示對抗訓練探測器不夠，任何固定防禦都會被自適應攻擊者打穿。」
  - 回應：不宣稱防禦完全有效，主張「在相同能力稅下，攻擊者需要的訓練量或可達到的逃避率被顯著壓低」，並附偵測器作為第二道防線。
  - 和 Dpwe23TWWq 的區隔要寫清楚，它是理論加上 steer 型攻擊，不是學到的逃避。
  - 審稿人仍可能認為增量偏小。

### 前兩週停損實驗
- **第 1 週**：
  - 用 minimal repo 的 Gemma-2-2b／9b chameleon checkpoint，重現觸發下的 TPR 掉落。
  - 建三種監測器：8 層 ensemble 基線、逃避子空間擴增探測器、Fisher 子空間探測器。
- **第 2 週**：
  - 用 2609.36490 式的 rank-1 LoRA 做一輪自適應攻擊。
  - 訓練不靠觸發詞的偵測器，用留出概念評估。
- **停損條件**（任一成立就停）：
  - 在 1% FPR 下，擴增探測器在留出安全概念上的 TPR 沒有比 8 層 ensemble 高至少 20 個百分點。
  - 偵測器在留出概念上的 AUROC 不到 0.8。
- **GPU 小時**：探測器很便宜，攻擊者的 LoRA 微調約 20–40 小時，總計 30–60 H200 小時〔推估〕。

**信心：中。**
- 做得出來：高。
- 中稿：中。這個題目熱門、有安全價值，但新穎性被 Dpwe23TWWq 部分吃掉，也有貓捉老鼠的反駁。

---

## 五、這次查過、不提的方向

| 方向（NeurIPS 2026 論文） | 不提的理由 | 證據層級 |
|---|---|---|
| Representation Fréchet Loss（2604.28190，oral） | 已有至少 4 篇跟進：2607.13300 醫學影像、2608.00562 自迴歸影像、2607.10421 音訊、2609.35763 統一的分布式訓練 | 〔API：arXiv〕 |
| Drifting 生成模型（2602.04770，oral） | 引用數 145，非常擁擠 | 〔API：S2〕 |
| 影片擴散模型內部的物理資訊（2606.05328），轉成引導或獎勵 | 2603.14294 *Seeking Physics in Diffusion Noise* 已做「探測器→驗證器→軌跡挑選加梯度引導」，另有 2610.03154 等多篇探測研究 | 〔摘要〕 |
| 量化推理模型想太多（2606.00206，oral） | 已有 2607.20129、2607.11317（CUSUM 解碼）、2610.29378（自生成軌跡回復）跟進；回報 31 也判過這一區 | 〔API：S2〕 |
| Activation patching 的多重中介交互作用（2606.27510） | 自然的方法續作「交互感知的電路發現」已有 2610.04017 | 〔API：S2〕 |
| 像素空間 v-prediction 的頻寬問題（2611.12852，spotlight） | 剛上線，沒有引用者。但 RAE 類工作已用加寬的 head 處理 DiT 寬度不足的問題，我判斷切入點偏窄；沒有深查 | 〔推估〕 |
| 回報 36 因擁擠刪掉的方向：迴圈模型 φ、NOPO 加 PU、ECHO 類、衍生資料 η | 放寬條件後仍沒有找到新的切入角度。迴圈模型 2026 年 9–11 月約 25 篇；NOPO 加 PU 的最強反駁（多生成測資）仍成立；η 的算力差距仍在 | 沿用回報 36 |
| 其他只讀摘要就排除的 | Master Key 能力轉移（steering 很擠）、Thinking tokens 與安全（安全推理很擠）、SpectraReward 或 T2I 獎勵模型的視覺推理（Flow-GRPO 類 RL 很擠，且計數題已在第三輪出現過） | 〔摘要〕 |

**Workshop 抽樣中看到、但不夠撐主會的**：
- 2610.07334 *Weight Oracles*（workshop spotlight）：與主會 LoRAcles oral 同一條「讀權重」路線。值得注意，但這輪沒有深查。
- 2609.39838 *Learning Steganography Is Easy, Learning Steganographic Reasoning Is Hard*（workshop oral）。
- 多個 workshop 都收了 *Drop the Act: Probe-Filtered RL for Faithful CoT*。

這幾篇都是單點結果，我沒看到可以直接擴成主會的開放問題〔摘要／API〕。

---

## 六、沒能確認的事項
- Neural Chameleons 的場次：NeurIPS 2026 JSON 標 oral，但搜尋結果有 iclr.cc 2026 頁面，未確認。
- 下列 arXiv 編號是憑記憶寫的，沒有在本輪核實：2106.05963（Learning to See by Looking at Noise）。Pose-RFT 與 HyLaR 只看到標題，沒讀摘要或全文。
- 所有 GPU 小時都是推估，沒有實測。
- Semantic Scholar 多次回 429。2611.11962 的引用清單沒拿到；它 11-09 才上線，預期為 0，未確認。
- Workshop 只做了關鍵字抽樣，沒有看完 102 個 workshop。
- 本報告所有「全文」讀取都是自己轉成純文字後做關鍵字定位，不是逐頁精讀。數字寫進正式文件前要再對原文。
