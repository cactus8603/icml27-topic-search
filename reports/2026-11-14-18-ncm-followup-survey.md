## NCM 之後的文字 VAR／由粗到細生成：文獻調查與 ICML 2027 題目評估

**結論：** NCM 才一個月，學術後續工作幾乎還沒有，所以「沒人做過」的空間確實存在。但它本身的根基比部落格呈現的弱：
- 訓練程式碼還沒釋出。
- OWT 品質輸給兩個 baseline。
- 釋出的評估設定在 LM1B 與 XSum 的最後解碼步加了頻率懲罰與鄰接懲罰，數字因此被修飾過。
- 最明顯的延伸（層內相依、joint training、adaptive hierarchy）多半會被原作者自己做掉。

調查限制：以下論文都經 arXiv 頁面或 API、論文 HTML、GitHub 實際確認。論文 HTML 是透過摘要工具分段讀的，細節可能有遺漏，有疑慮的地方會註明。OpenReview API 被 challenge 擋下，無法直接查投稿。

---

### 1. NCM 本身，以及之後與之前的相關工作

#### 1.0 NCM 本身（可能影響是否以它為基礎的細節）

**Next-Concept Modeling: Fast, Parallel Coarse-to-Fine Text Generation**
- 作者：Kumbong, Khare, Mlauzi, Woen, Ding, Buchanan, Arora, Mirhoseini, Ré
- arXiv 2610.14162，2026-10-09，沒有註明投稿會議
- 連結：https://arxiv.org/abs/2610.14162 ，程式碼 https://github.com/HazyResearch/ncm

**Tokenizer**
- 每一層把 residual 用內插降採樣到長度 ℓ_k，量化後再還原到 L；每層有自己的 codebook；nested dropout。
- **encoder 為 0 個 block**，也就是粗層「concept」其實是 token embedding 經池化後再量化。decoder 只有 1–4 個 block。
- 規模：
  - OWT：L=1024，K=32（長度依平方數 1,4,9,…,1024 遞增），codebook 4096（論文表）／8192（釋出 config）。
  - LM1B：L=128，K=12，codebook 8192。
- 重建準確率 OWT 99.99%、LM1B 99.86%，codebook 利用率 62–90%。

**Generator**
- 約 100M（LM1B 97M）。
- 使用 block-diagonal attention：第 k 層只看累積重建，屬於 Markov 式，與作者自己的 HMAR 相同思路。
- 抗暴露偏差的作法：條件 code 以平均 0.5 的比例均勻隨機替換，替換率隨層數線性遞減。

**關鍵數字**（Table 8，GPT-2-large Gen PPL／entropy）

| 資料集 | NCM | DUO | MDLM | FLM | 資料本身 |
|---|---|---|---|---|---|
| LM1B | 93.05／4.31（12 NFE） | 97.30（128 NFE） | 115.45 | 112.33 | 53.51／4.31 |
| OWT | 83.12／5.42（32 NFE） | 77.69／5.55（1024 NFE） | 105.15 | 62.23／5.33 | 14.15 |

- 在 OWT 上 NCM 輸給 DUO 和 FLM。這些 OWT baseline 數字與 FLM 論文 Table 10 完全相同，看起來是直接引用的。
- 條件任務：XSum ROUGE-L 28.0（8 NFE）；WMT14 BLEU 22.6（AR 為 25.2）。
- Sudoku（hard）：86.1%，對比 MDLM 30.2%、FMLM+ 81.3%。
- 論文沒有回報 likelihood（validation PPL）。

**作者自列的 limitations／future work**
- 固定的層數深度與各層長度；改成 input-dependent schedule（adaptive computation）。
- tokenizer 與 generator 分開訓練；改成 end-to-end（joint）訓練。
- 主實驗模型只有約 100M。
- 粗層之間的過渡處偶爾出現瑕疵。

**我在釋出程式碼中查到、但在讀到的論文內文裡沒找到的事**
- `config/eval/lm1b-*.yaml` 與 `xsum.yaml` 中，最後由 decoder logits 轉成 token 時是 argmax，並套用：
  - `frequency_penalty: 11.5`
  - `frequency_deficit: 2.0`
  - `adjacency_penalty`
  - `frequency_allowance`：以訓練資料 token unigram 為目標，計數超過時扣分，不足時加分
- 實作在 `scripts/eval_shard.py` 的 `pick_tokens`。這等於直接把輸出的 unigram 分佈往資料推，而 unigram entropy 正好是被回報的指標：LM1B 的 4.31 與資料完全相等。OWT 的 config 沒有這些懲罰。
- 論文只寫「Select output tokens by argmax」；Appendix F.4 只提 CFG 與 MBR。
- 這些設定從 2026-10-13/14 首次釋出時就存在。

**GitHub 上的獨立複現與批評**（以下都是 issue，非同行審查）
- Issue #9、#26：品質數字可以重現。
  - LM1B 97M／317M／725M 的 Gen PPL：94.7／89.0／81.3（作者在 LM1B 做了 scaling 實驗，但論文沒列中、大模型的 Gen PPL）。
  - Figure 1 的速度要靠 KV cache、bf16 與 CUDA graphs 才能重現；作者在 issue 中確認了這個設定。
  - XSum 的 NFE 實際是 9，論文寫 8；作者承認是 off-by-one，並說 unconditional 不計入 level-0 那一步。
- Issue #26 的留言（granman3）：與同形狀的 GPT-2 加 KV cache 比，batch 1 時 NCM 快 7–8 倍，但 batch 128 時 AR 反而稍快、每個樣本耗能也較低（0.82–0.97 倍）。
- Issue #19：
  - 罕見人名與數字經 codec 重建後只有 89%（XSum）保留下來。
  - 在一個小任務上，自行重寫的 27M NCM 輸給同大小的 AR：exact match 51.0% vs 57.6%。
  - 他們也指出至今沒有 compute-matched 的 NCM vs AR 比較。
- Issue #16：作者親口表示 codebook restart 會改善重建與利用率，但下游生成反而變差。這是典型的 reconstruction vs generation 權衡。
- Issue #10：**訓練程式碼尚未釋出**。作者回「Yes」，但到 2026-11-14 repo 最後一次 commit 是 10-14，裡面只有 eval 程式。OWT checkpoint 也還沒釋出。
- 社群已有 PR 在做 FSQ、Jacobi self-correction、coupled decoder（PR #7）與 adaptive early exit（PR #12），都未經驗證。
- Repo 熱度：1,864 stars、252 forks（一個月內）。

**引用、審稿與複現**
- Semantic Scholar 目前查到 0 篇引用。
- 搜尋找不到 OpenReview 投稿頁或審稿意見，**未確認**它是否投了 ICLR 2027。
- 沒有正式發表的複現或批評論文；有一些部落格與科普文，沒有技術批評。
- 同期（2026-10 之後）找不到另一篇文字 VAR 或多尺度 residual code 生成的論文。

#### 1.1 NCM 之前的相關工作，以及 NCM 與它們的差別

- **Fast Decoding in Sequence Models using Discrete Latent Variables**
  - Kaiser, Roy, Vaswani, Parmar 等，ICML 2018，https://arxiv.org/abs/1803.03382
  - 方法：把目標序列壓成較短的離散 latent，用 AR 生成這段 latent，再平行解碼出全文。
  - 差別：這是「單層壓縮 latent ＋ 平行解碼」的原型；NCM 是多層 residual 並逐層平行。NCM 沒有引用它（以我讀到的參考文獻為準）。
- **Progressive Generation of Long Text with Pretrained LMs**
  - Tan 等，NAACL 2021，https://arxiv.org/abs/2006.15720
  - 方法：先產生關鍵詞，再分多階段逐步擴寫成全文，明確受「影像由低解析到高解析」啟發。
  - 差別：粗層是顯式文字，不是學出來的 code。
- **Latent Diffusion for Language Generation**（LD4LG）
  - Lovelace 等，NeurIPS 2023，https://arxiv.org/abs/2212.09462
  - 方法：在 encoder-decoder autoencoder 的連續 latent 上做 diffusion。
- **PLANNER**
  - Zhang, Gu 等，NeurIPS 2023，https://arxiv.org/abs/2306.02531
  - 方法：用 latent diffusion 產生段落語意 embedding 作為規劃，再交給 AR 解碼。
- **Large Concept Models**
  - LCM team（Meta），2024，https://arxiv.org/abs/2412.08821
  - 方法：在 SONAR 句子 embedding 空間做 AR，一個句子就是一個 concept。
  - 差別：連續、句子級、序列式；NCM 是離散、全序列、多尺度。
- **Next Semantic Scale Prediction via Hierarchical Diffusion LMs**（HDLM）
  - Zhou, Wang, Zhang, Tong, Wang, Bates, Jaakkola，**NeurIPS 2025**，https://arxiv.org/abs/2510.08632
  - 方法：用預訓練 embedding 分群，建出「字 → 群 → mask」三層詞彙；forward process 讓 token 往祖先移動，並推出閉式 ELBO；MDLM 是它的特例。
  - 數字：OWT 170M、131B tokens，valid PPL ≤23.36（MDLM 重新實作版為 27.39），Gen PPL 144.2 vs 163.7。
  - **這是概念上最接近 NCM 的 prior work**：同樣在講「next semantic scale」，但它是 diffusion 加固定分群，且有似然。
- **Next Concept Prediction in Discrete Latent Space**（ConceptLM）
  - Liu, Song, …, Zhouhan Lin，arXiv 2026-02，https://arxiv.org/abs/2602.08984
  - 方法：用 VQ 把 hidden state 變成 concept 詞彙，先預測 concept 再引導 token 生成，仍是 AR。
  - 數字：70M–1.5B，最多 300B tokens，13 個 benchmark。
- **NCP-ArchPreview**
  - Intern-NCP Team，arXiv 2026-09，https://arxiv.org/abs/2609.10715
  - 方法：同一路線擴大規模，用 product quantization 建 concept，與 NTP 一起端到端訓練。
  - 數字：8.9B 參數、5.73T tokens；用 51.3% 的 tokens 達到 OLMo-3-7B 的最終 loss。
- **Dynamic Large Concept Models**
  - Qu 等，arXiv 2025-12，https://arxiv.org/abs/2512.24617
  - 方法：端到端學出變長 concept 邊界。
  - 數字：R=4 時，12 個 benchmark 平均 +2.69%。
- **Cola DLM**
  - Guo 等，arXiv 2026-05，https://arxiv.org/abs/2605.06548
  - 方法：Text VAE 的連續 latent，加上 block-causal DiT，再接條件 decoder；與約 2B 的 AR／LLaDA 比較。
- **AURORA-LM**
  - Liang 等，arXiv 2026-08，https://arxiv.org/abs/2608.02602
  - 方法：連續 latent 加 block-causal flow matching。
  - 數字：NCM 表中 XSum R-L 26.6（128 NFE）／28.6（加 guidance，184 NFE）。

**NCM 的實質差別**：多尺度 residual 離散 code、每層獨立 codebook、層內完全平行、粗層長度隨層增長。前人要嘛只有單層 latent（Kaiser）、要嘛是連續句子 latent（LCM、PLANNER）、要嘛是 AR 加 concept（ConceptLM），或是固定分群的 diffusion（HDLM）。

---

### 2. 視覺 VAR 系列的已知問題、對應解法，以及在文字 NCM 上是否已有人做

**基準論文：VAR**
- Visual Autoregressive Modeling，Tian, Jiang, Yuan, Peng 等，2024，NeurIPS 2024（會議由我記憶得知，**未經本次開啟確認**），https://arxiv.org/abs/2404.02905
- 數字：ImageNet-256 FID 1.73、IS 350.2；比 AR baseline 快約 20 倍。

**(a) 跨尺度誤差累積與暴露偏差**
- **Infinity**（Han 等，2024，https://arxiv.org/abs/2412.04431 ，會議**未確認**）
  - 方法：bitwise 預測、infinite-vocabulary tokenizer、Bitwise Self-Correction（訓練時隨機翻轉 bit 後重新量化 residual，讓模型學會修正）。
  - 數字：GenEval 0.62→0.73（對比 SD3-Medium）；1024² 影像 0.8 秒。
- **SAR: Rethinking Training Dynamics in Scale-wise AR Generation**（Zhou, Ge, Tan, Liu 等，arXiv 2512.06421，CVF open access 列於 CVPR 2026 Findings，https://arxiv.org/abs/2512.06421 ）
  - 方法：Stagger-Scale Rollout 讓模型看到自己的中間預測；Contrastive Student-Forcing Loss 監督這些自生成條件。
  - 數字：FlexVAR-d16 的 FID 下降 5.2%，只需 10 個 epoch。
- **xAR**（Ren 等，2025，https://arxiv.org/abs/2502.20388 ）
  - 方法：Noisy Context Learning，在帶雜訊的 context 上訓練以避開 teacher forcing。
  - 數字：xAR-H FID 1.24。
- **AID-VAR**（Bi, Huang, Guo, Xu，arXiv 2026-05，https://arxiv.org/abs/2605.24843 ）
  - 方法：對抗式 discriminator 在每個尺度過渡處診斷，搭配輕量 injector 修正；提出跨尺度一致性指標 ISCS。
  - 數字：d20 的 FID 改善 16%，參數只增加 3%。
- **在 NCM 上**：**只做了最簡單的版本**，也就是均勻隨機替換條件 code（類似 BSC 或 xAR，但雜訊不貼近模型實際犯錯的分佈）。模型誤差感知的 corruption、rollout 或 student forcing、跨層誤差傳播分析，**都還沒人做**（PR #7 的「Jacobi residual polish」未經驗證）。

**(b) 同一尺度內 token 條件獨立**
- **HMAR**（**Kumbong**, Liu, Lin, Liu 等，CVPR 2025，https://arxiv.org/abs/2506.04421 ）
  - 方法：Markov 式 next-scale，加上尺度內多步 masked 生成；取樣排程可不重訓就更改。
  - 數字：訓練快 2.5 倍以上、推論快 1.75 倍以上、記憶體省 3 倍以上。
  - **這是 NCM 第一作者自己的前作**。
- **Logit Refiner**（Li, Baumann, Krause, Ommer，**ECCV 2026**，https://arxiv.org/abs/2609.11804 ）
  - 方法：指出層內平行取樣是 mean-field 近似；加一個輕量 AR refiner，在 frozen backbone 特徵上做層內循序取樣。
  - 數字：參數增加約 10%，訓練算力不到 5%；在 310M–2B 都有改善。
- **在 NCM 上**：**沒有正式工作**。間接證據顯示問題確實存在：
  - 釋出的 config 需要頻率與鄰接懲罰。
  - PR #7 提到 non-AR「token stuttering」。
  - 論文自承粗層之間的過渡處有瑕疵。

**(c) Tokenizer 的重建上限與 codebook 利用率**
- **HART**（Tang, Wu 等，2024，https://arxiv.org/abs/2410.10812 ，會議**未確認**）
  - 方法：見 (d)。
  - 數字：rFID 2.11→0.30；MJHQ gFID 7.85→5.38；吞吐量高 4.5–7.7 倍。
- **ImageFolder**（Li, Qiu 等，2024，https://arxiv.org/abs/2410.01756 ）
  - 方法：用語意正則化分支配上細節分支，並用 product quantization。
  - 論點：token 越長，重建越好，但生成不一定越好。
  - NCM 的 nested dropout 正是引自這篇。
- **在 NCM 上**：token 級重建已接近 99.99%，所以瓶頸不是重建，而是：
  - 罕見實體與數字的重建（issue #19）。
  - **重建與可生成性的權衡**（作者自承 restart 有害）。
  - 粗層是否真的帶語意（encoder 0 個 block）。
  
  這三點**都沒人系統研究**。

**(d) 連續與離散混合 residual**
- HART 用離散 token 表示大結構，連續 residual 交給 37M 的 residual diffusion。
- **FlowAR**（Ren, Yu, He, Shen 等，**ICML 2025**，https://arxiv.org/abs/2412.15205 ）：每個尺度用 flow matching，可直接接任意現成 VAE；約 2B 時 FID 比 VAR 好 0.32。
- **在 NCM 上**：沒人做。但文字最後必須是離散 token，而且連續 latent 的文字 diffusion 已經很擠（AURORA-LM、ELF、Cola、CANDI），**價值低**。

**(e) 大模型做粗尺度、小模型做細尺度，以及其他加速與蒸餾**
- **CoDe: Collaborative Decoding**（Chen, Ma, Fang, Wang，2024，https://arxiv.org/abs/2411.17787 ，會議**未確認**）
  - 方法：大模型負責小尺度（drafter），小模型負責大尺度（refiner）。
  - 數字：加速 1.7 倍、記憶體省約 50%、FID 1.95→1.98；推到 2.9 倍時 FID 2.27。
- **FastVAR**（Guo 等，ICCV 2025，https://arxiv.org/abs/2503.23367 ）
  - 方法：在大尺度做 cached token pruning。
  - 數字：加速 2.7 倍，品質下降不到 1%。
- **ScaleKV**（Li, Chen 等，2025，https://arxiv.org/abs/2505.19602 ）
  - 方法：把層分成 drafter 與 refiner，分別給不同的 KV cache 預算。
  - 數字：KV cache 降到 10%。
- **MVAR**（Zhang 等，ICLR 2026，https://arxiv.org/abs/2505.12742 ）
  - 方法：尺度 Markov 加空間 Markov 注意力。
  - 數字：記憶體省 3 倍。
- **Distilled Decoding**（Liu, Ning, Wang, Lin，2024，https://arxiv.org/abs/2412.17153 ，會議**未確認**）
  - 方法：用 flow matching 把 AR 蒸餾成 1 步。
  - 數字：VAR 10 步→1 步，FID 4.19→9.96。
- **在 NCM 上**：都還沒人做。注意 OWT 各層長度總和 Σℓ_k = 11,440，是輸出 1024 的 11 倍以上，算力集中在細層。
  - 這條路最大的風險是「工程味重」，而且跟作者 future work 的 adaptive computation 部分重疊。
  - 社群 PR #12 已經在做 early exit。

**(f) 可變解析度與任意長度**
- **FlexVAR**（Jiao 等，2025，https://arxiv.org/abs/2502.20313 ）
  - 方法：改成預測 ground truth 而非 residual，每一步都能單獨解碼成圖；支援任意解析度與任意步數。
  - 數字：FID 2.08（13 步）。
- **FlexTok**（Bachmann 等，ICML 2025，https://arxiv.org/abs/2502.13967 ）
  - 方法：1–256 個 token 的變長 1D tokenizer，靠 nested dropout 加 rectified flow decoder。
  - 數字：8–128 tokens 時 FID<2。
- **DetailFlow**（Liu 等，2025，https://arxiv.org/abs/2505.21473 ）
  - 方法：1D next-detail prediction，平行推論並帶 self-correction。
  - 數字：128 tokens FID 2.96；比 VAR、FlexVAR 快約 2 倍。
- **在 NCM 上**：NCM 每個資料集用固定的 L，沒有處理變長。
  - 變長或 semi-AR block NCM 是很明顯的下一步，而 block diffusion 那一系已經很擠。
  - 「adaptive hierarchy」在作者 future work 裡明列。
  - 社群（issue #19）已在試 sliding window。

**(g) VAR 與離散 diffusion 的理論連結**
- **Scale-Wise VAR is Secretly Discrete Diffusion**（SRDD，Kumar, Nair, Patel，2025 技術報告，https://arxiv.org/abs/2509.22636 ）
  - 論點：VAR 加上 Markov attention mask 時，數學上等價於離散 diffusion；因此可以引入 iterative refinement 等 diffusion 技巧。
- HDLM（見 §1.1）在文字側給出了階層詞彙 diffusion 的 ELBO。
- **在 NCM 上**：沒人做。
  - NCM 用 block-diagonal attention，正是 SRDD 前提的 Markov 結構。
  - 它的「均勻隨機 code 替換」在形式上就像 uniform-state 離散 diffusion 的雜訊（DUO、GIDD）。
  - NCM 完全沒有似然，這是 DLM 審稿人一定會問的點。

**(h) Guidance 與取樣技巧**
- **CCA: Condition Contrastive Alignment**（Chen, Su, Sun, Zhu，2024，https://arxiv.org/abs/2410.09347 ，會議**未確認**）
  - 方法：用類 alignment 的微調取代 CFG。
  - 數字：只需約 1% 的預訓練 epoch，就讓無 guidance 的品質接近有 CFG，取樣成本減半。
- VAR 本身靠 CFG 搭配 top-k／top-p。
- **在 NCM 上**：NCM 有 CFG（w=0.5 最佳）與 MBR（XSum R-1 +4.48，K 從 1 到 256）。各層溫度與 guidance 排程、guidance-free 微調都還沒人做；作者未公開的最終解碼懲罰本身也值得檢驗。

---

### 3. 離散 diffusion 語言模型子領域的現行標準

**資料集、模型規模與設定**
- **OWT**：GPT-2 tokenizer，L=1024。
- **LM1B**：bert-base-uncased，L=128。
- 主模型是約 110–180M 的 DiT（FLM 用 179M、12 個 block，1M steps × batch 512；HDLM 用 170M／425M，131B tokens）。
- 近期論文常加上條件任務或推理任務：XSum、WMT14、Sudoku、GSM8K。
- 少數論文擴到 1B（AURORA-LM），工業界有 DiffusionGemma（3.8B active，https://arxiv.org/abs/2608.00146 ）。

**指標**
- Gen PPL（以 GPT-2-large 評估），加上 mean unigram entropy（OWT 資料 entropy 5.44，LM1B 4.31）。
- NFE。
- 似然論文會報 validation PPL 上界。
- 有些論文會報 wall-clock，但 NCM 自己也承認 NFE 會掩蓋每次前向的成本差異。

**已知的指標問題**
- **Masked Diffusion Models are Secretly Time-Agnostic…**（Zheng 等，ICLR 2025，https://arxiv.org/abs/2409.02908 ）：32-bit 的 categorical 取樣有數值問題，等於降低了溫度；token 多樣性下降，讓過去只看 Gen PPL 的比較不公平。
- **FLM 論文**（https://arxiv.org/abs/2602.16813 ）：
  - 明說 Gen PPL 可以靠重複 token 刷低。
  - 以 entropy 低於約 4 作為退化門檻。
  - 指出某些 few-step baseline 是靠 entropy 崩潰才拿到低 Gen PPL（例如 Duo+Di4C 1 步：292.9／3.79）。
- **S-FLM**（https://arxiv.org/abs/2605.11125 ）：Gen PPL 高，不代表在可驗證任務上正確。
- 這些都和 NCM 用懲罰把 entropy 推到正好等於資料值的作法直接相關。

**最強 baseline 數字**（FLM Table 10，1024 步）

| 方法 | OWT Gen PPL／entropy | LM1B Gen PPL／entropy |
|---|---|---|
| FLM | 62.23／5.33 | 96.91／4.29 |
| DUO | 77.69／5.55 | 98.14 |
| MDLM | 105.15 | 109.21 |
| SEDD | 101.43（取自 NCM 表） | — |

Few-step：
- FMLM：OWT 1 步 168.3、4 步 111.3。
- DiDi-Instruct：OWT 8 NFE 62.2、128 NFE 18.4。這是搜尋摘要的數字，我**沒有開論文確認**；它的 OpenReview 頁顯示為 ICLR 2026 投稿。

**發表會議**（以下都經 arXiv 註記或官方頁確認）

| 會議 | 代表作 |
|---|---|
| ICML 2024 | SEDD（Oral，https://arxiv.org/abs/2310.16834 ） |
| NeurIPS 2024 | MDLM（https://arxiv.org/abs/2406.07524 ）、MD4（https://arxiv.org/abs/2406.04329 ） |
| ICLR 2025 | Block Diffusion（Oral，https://arxiv.org/abs/2503.09573 ）、Zheng 等、EDLM（https://arxiv.org/abs/2410.21357 ）、Simple Guidance（https://arxiv.org/abs/2412.10193 ） |
| ICML 2025 | DUO（https://arxiv.org/abs/2506.10892 ）、GIDD（https://arxiv.org/abs/2503.04482 ）、Train for the Worst（Kim 等，**Outstanding Paper**，https://arxiv.org/abs/2502.06768 ） |
| NeurIPS 2025 | ReMDM（https://arxiv.org/abs/2503.00307 ）、E2D2（https://arxiv.org/abs/2510.22852 ）、HDLM |
| ICLR 2026 | Soft-Masked DLM（IBM，官方 proceedings PDF）；Loopholing Discrete Diffusion（**只見到 proceedings PDF，未開確認**） |
| ICML 2026 | Eso-LMs（https://arxiv.org/abs/2506.01928 ） |
| NeurIPS 2026 | ELF（https://arxiv.org/abs/2605.10938 ） |

- SDTT（https://arxiv.org/abs/2410.21035 ）的會議**未確認**。
- FLM 的會議**未確認**。

**規模要求**：約 100–170M 加 OWT（以及 LM1B）在 2024–2026 的 ICML、NeurIPS、ICLR 都還是主流且足以被接受。但現在幾乎都要配上：
- entropy；
- 多種 NFE 下的曲線；
- 至少一個條件或推理任務；
- 最好有 scaling 點或 1B 等級的實驗。

以 NCM 的定位，審稿人還會要求似然（val PPL），以及與 AR 的 wall-clock 比較。

---

### 4. 綜合判斷（以下全是我的判斷）

**四個比較像 ICML 題目的方向**，依「機制洞見加方法、撞車風險可控」排序：

1. **NCM 的似然與「它其實是離散 diffusion」**
   - 內容：因為 tokenizer 是確定性的，log p(x) ≥ log p_θ(c(x)) + log p(x|c(x)) 是可直接計算的下界，可以補上 NCM 缺的 val PPL。再用 SRDD 的 Markov 等價，把隨機 code 替換解讀為 uniform-state 雜訊，由此推出有原則的 corruption schedule 與層權重 α，並與 MDLM、DUO 公平比較。
   - 最可能撞到：SRDD、HDLM、DUO／GIDD 的理論，以及原作者自己的補充。
   - 風險：下界可能很鬆，讓 NCM 看起來很差。這本身是發現，但會削弱「方法」的賣點。

2. **貼近模型誤差的 corruption 與跨層誤差傳播**
   - 內容：先量測推論時真正出現的錯誤分佈（是否多為 codebook 鄰近碼，而非均勻隨機），再改用 rollout、student forcing 或鄰近碼替換來訓練，並提出跨層一致性指標。
   - 最可能撞到：Infinity BSC、SAR、AID-VAR、xAR；文字側有 ReMDM、GIDD 的 self-correction。
   - 風險：作者已經有 corruption 這個開關，很容易自己迭代。

3. **粗層到底有沒有語意：tokenizer 的可生成性 vs 重建權衡**
   - 內容：encoder 是 0 個 block，作者也承認 restart 會傷害生成。可以量測各層 code 的語意性與條件 entropy，再用語意對齊（例如對齊預訓練 LM 特徵、ImageFolder 式的語意分支）改造粗層。
   - 最可能撞到：作者 future work 的 joint training、ImageFolder、ConceptLM／NCP-ArchPreview。
   - 風險：撞車機率中到高。

4. **層內相依的量化與輕量修正**
   - 內容：量測各層的 joint vs marginal 誤差，證明那些懲罰在補償 factorization error，然後用 Logit Refiner 式或 HMAR 式的層內多步取樣，**拿掉懲罰**後比較。
   - 最可能撞到：**HMAR 是第一作者自己的方法，被作者自己做掉的機率最高**；另有 Logit Refiner、FMLM+ Posterior Refinement、EDLM。
   - 判斷：洞見清楚，但被 scoop 的風險最大。

5. （較弱）**細層的算力重分配（CoDe／FastVAR 式）**
   - 撞車少，但 ICML 審稿人可能覺得增量。
   - 與 adaptive computation、社群 PR #12 部分重疊。

**太擠或明顯會被原作者做掉的方向**
- 原作者 future work 明列的：joint／end-to-end training、adaptive hierarchy 與 early exit、scaling。
- HMAR 式的層內 masked 生成：作者的前作。
- 變長或 block semi-AR 版本：block diffusion 那一系太擠。
- 少步蒸餾：DUO-DCD、SDTT、Di4C、FMLM、DiDi-Instruct 已經很擠。
- 連續或混合 latent：AURORA、ELF、Cola、CANDI 已經很擠。

**建立在一個月大、未經審查框架上的具體風險**
1. **沒有訓練程式碼，也沒有 OWT checkpoint。** 10 週內要自己重寫 tokenizer 與 generator 訓練，並在 OWT 重現 83 Gen PPL。光這一步就可能吃掉一半時間，而且重現失敗就沒有根基。
2. **主張本身偏弱。**
   - OWT 品質輸給 DUO 與 FLM。
   - 速度優勢建立在優化過的 NCM stack，對手是 1024 NFE 的 baseline。
   - 與 AR 比，大 batch 時沒有優勢（社群量測）。
   - LM1B 與 XSum 的數字套用了解碼懲罰，而我讀到的論文內文沒提。
   
   若 NCM 在審稿或後續檢驗中被削弱，以它為前提的論文會跟著被質疑。
3. **作者動作快、規模大**（Hazy Research 加上 HMAR 的作者），很可能在 ICML 2027 交出 NCM v2，正面撞上任何「顯而易見的下一步」。
4. **社群擁擠。** 一個月就有 252 forks，PR 已經在碰 self-correction、FSQ、early exit；2027 年 1 月 arXiv 很可能同時出現多篇 NCM 延伸。
5. **審稿框架風險。** 「在未發表方法上做增量」容易被打。比較安全的寫法是把題目定義成「文字的多尺度離散階層生成」這個一般問題，NCM 只是其中一個實例，最好再加上 HDLM 式或自行實作的變體，並自己補上似然、entropy 與 AR wall-clock 比較。

---

**主要來源**（未經開啟確認的會議出處已在內文標記）
- https://arxiv.org/abs/2610.14162 ，https://arxiv.org/html/2610.14162
- https://hazyresearch.stanford.edu/blog/2026-10-12-ncm
- https://github.com/HazyResearch/ncm （issues #7, #9, #10, #12, #16, #19, #26；`config/eval/*.yaml`；`scripts/eval_shard.py`）
- 其餘論文連結見各條目。
