## NCM（arXiv 2610.14162）可行性調查：整理、本機驗證與判斷

標記說明：〔論文〕是論文原文寫的；〔驗證〕是我讀程式、讀檔案或在 CPU 上實際跑過確認的；〔推論〕是我的判斷或推估。論文全文（含附錄 A–J）取自 arXiv HTML 版，轉成文字存在 /tmp/claude-1047/-work-ChihChungHsu-cactus8603-icml27-s2-gated-sink-kv/bd65f9a7-961a-4b8d-adce-42df28fbb74b/scratchpad/paper/p.txt。沒有改任何專案檔、沒有送 slurm、沒有用 GPU。唯一的運算是兩支 CPU 小腳本，放在 scratchpad：`tf_check.py`、`unigram_check.py`。

### 先講最重要的五點
1. **〔驗證〕LM1B 的 Gen PPL 是在一套論文沒寫的後處理下量的。** LM1B 的設定（`config/eval/lm1b-small.yaml:130-134`）最後挑 token 時不是 argmax：
   - frequency penalty 11.5；
   - frequency deficit 2.0，搭配從 generator.pt 讀出的資料 token unigram 當配額；
   - adjacency penalty `10,4,0.5`；
   - 逐位置跑的 Python 迴圈，在 `scripts/eval_shard.py:99-124`。

   這套規則會把每個樣本的 token 次數拉向資料的 unigram 分布，直接影響論文拿來當多樣性把關的 unigram entropy H，也會壓掉重複詞。論文 Algorithm 2 第 11 行寫的是「Select output tokens by argmax」，全文 grep 不到 penalty。部落格的內容也沒提（是由一個小模型讀部落格頁面後轉述的）。XSum 也有較輕的同類 penalty（freq 2.0、adjacency `5,4,0.5`）。
2. **〔驗證〕無條件生成的第 0 層 code 不是模型生成的。** LM1B、TinyStories、OWT 的第 0 層是從 checkpoint 裡存的 `level0_unigram` 抽樣（`eval_shard.py:71, 156`），論文沒寫。
3. **〔驗證〕官方 repo 只有推論與評測程式，沒有任何訓練程式**，沒有 loss、optimizer、dataloader。設定檔裡的 `code_corruption`、`fine_drop.dual/w_full/w_trunc` 在釋出的程式中沒有被實作，載入時直接被忽略。要從頭訓練就得自己重寫訓練流程。
4. **〔驗證〕OWT 權重沒有釋出**（HF 上 `owt/TODO.md` 的內容只有「TODO」）。repo 的 `owt.yaml` 寫的 generator 是 16 層、寬 1024、codebook 8192，和論文 Table 6 的 12 層、768、94.65M 及 Table 5 的 4096 都不一致。
5. **〔驗證，小樣本〕粗層 code 幾乎無法從前面各層預測。** 我在 CPU 上用 32 條 LM1B valid 序列做 teacher-forced 檢查：第 1–3 層的 CE 是 8.46／8.71／8.33 nats，codebook 本身的 unigram 基線是 8.21／8.64／8.61（ln 8192 = 9.01）。也就是粗層的生成和直接從邊際分布抽差不多。第 4 層以後模型才明顯勝過 unigram（7.44 對 8.31，到最後一層 1.02 對 7.06）。

---

## 1. 論文精確整理〔論文〕

### Tokenizer（多尺度 residual VQ-VAE）
- **架構**（§3.3、Table 5）：
  - encoder blocks 全部為 0，也就是 encoder 只是 token embedding 查表（vae.py:102-178）。
  - decoder 是 transformer：LM1B 4 個 block，其他資料集 1 個；6 個 head，Sudoku 4 個。
  - 每層先把 residual 用 Down_k 縮到 ℓ_k 個位置，對 codebook 做最近鄰，再用 R_k 放回長度 L，累加成 ẑ^(k)。
  - 程式中 Down 是 `F.interpolate(mode="area")`、Up 是 linear 內插，兩者在 quant.py:395-411。每層再過一個 Phi 卷積（quant_resi ratio 0.5）。LM1B 第 0 層改用學出來的 stride conv（`learned_l1_sampling`）。
- **層數 K／目標長度 L／codebook 大小**（Table 5）：

  | 資料集 | K | L | codebook | latent d | tokenizer 參數 |
  |---|---|---|---|---|---|
  | OWT | 32 | 1024 | 4096 | 384 | 111.2M |
  | LM1B | 12 | 128 | 8192 | 384 | 111.40M |
  | TinyStories | 12 | 128 | 4096 | 384 | 36.43M |
  | XSum | 8 | 48 | 4096 | 384 | 60.63M |
  | WMT14 | 11 | 64 | 4096 | 384 | 144.00M |
  | Sudoku | 12 | 89 | 16 | 256 | 14.90M |

- **每層長度（取自設定檔）**〔驗證〕：
  - LM1B／TinyStories：[1, 8, 16, 24, 32, 40, 52, 64, 76, 92, 108, 128]，合計 641。
  - XSum：[4, 6, 10, 14, 20, 26, 32, 40, 48]，共 9 層，和論文寫的 K=8 不一致。
  - WMT14：[1, 4, 8, 14, 20, 26, 32, 40, 48, 56, 64]。
  - Sudoku：[1, 2, 4, 6, 9, 13, 18, 26, 37, 50, 67, 89]。
  - OWT：平方數 1, 4, 9, …, 1024，共 32 層。
- **訓練目標**（Eq. 9）：在抽到的截斷層 k 上算重建 CE，加上 β·MSE(z, sg[ẑ]) 與 MSE(sg[z], ẑ)。用 straight-through 讓梯度穿過量化，codebook 用 EMA 更新（decay 0.95–0.99）。
- **Nested dropout**：Table 5 寫 0.5，意思是丟掉細層的機率。LM1B 寫「0.7 / 0.3」，指完整階層與截斷階層的損失權重各 0.7、0.3。
- **〔驗證〕程式和論文的兩處差異**：
  - VQ loss 是每一層都算再取平均（quant.py:519-524），不是只在截斷點算。
  - 截斷只作用在送進 decoder 的 ẑ（quant.py:525-531）。
- **重建結果**（Table 1，用完整階層、held-out 資料的精確 token 準確率）：OWT 99.99、LM1B 99.86、TinyStories 99.98、XSum 99.79、WMT14 98.79、Sudoku 100（三種難度）。codebook 使用率 62–90%，WMT14 最低，61.89%。

### Generator
- **大小**（Table 6）：
  - OWT 94.65M；LM1B 97.07M；TinyStories 88.61M；XSum 88.02M。以上都是 12 層、寬 768、12 個 head。
  - WMT14 119.96M（12/768/12）。
  - Sudoku 25.46M（8 層、寬 512、8 個 head）。
  - Prefix 長度：XSum 512、WMT14 96、Sudoku 91，其餘 1。
- **注意力遮罩**：block-diagonal，每一層的位置只看自己這一層加上 prefix；prefix 之間互看，但看不到 target（§3.4、App F.2.1）。〔驗證〕程式在 next_concept.py:266-292。
- **輸入**：把 ẑ^(k-1) 用 area pooling 縮到 ℓ_k 個位置（quant.py:670-694, 696-709），經 `wte` 投影，加上 level embedding 與逐層重置的 RoPE。
- **訓練**：
  - 所有層在一次前向中算完。
  - Loss 是 Eq. 10，權重 w_k ∝ ℓ_k^-α。α=0 時每個 code 等權，α=1 時每層等權。
- **Conditioning corruption**：把作為條件的 code 以各層不同的機率換成同層 codebook 中均勻抽的 code，標籤維持乾淨（§3.4、F.3）。選用的是「線性遞減」排程，TinyStories 消融選了平均 0.5。〔驗證〕LM1B 設定寫的是 `mode: uniform, random_rate: true, low 0, high 1, skip_level0`，和論文描述不完全一樣；訓練程式沒有釋出，無法核對。
- **推論**：每層一次 forward，同一層的位置平行、互相獨立地抽樣（Eq. 5），最後由 VQ decoder 解碼。〔驗證〕各任務的實際設定：
  - 無條件（LM1B、TinyStories、OWT）：temperature 1.0、top_k 0、top_p 1.0，即純 multinomial 抽樣（next_concept.py:487-499）。
  - XSum、WMT14：top_k=1，即 greedy。
  - Sudoku：temperature 0，即 argmax。
  - CFG 依 VAR 的做法隨層線性增加權重（next_concept.py:477-485）。
- **實際 forward 次數**〔驗證〕：
  - LM1B／TinyStories：11 次，第 0 層來自 unigram；論文寫 NFE 12。
  - OWT：31 次；論文寫 32。
  - XSum：`query_level0` 加 8 次，共 9 次；論文 Table 2 寫 8。
  - WMT14：1 次 prefix pass 加 10 次，共 11 次，和論文吻合。
  - Sudoku：12 次，吻合。

### 訓練資料、token 數與算力〔論文〕
- **資料**（Table 4）：
  - OWT（GPT-2 tokenizer，L=1024）。
  - LM1B（bert-base-uncased，L=128，打包序列）。
  - TinyStories（自訓 BPE 8192，L=128）。
  - XSum（GPT-2 tokenizer，文章 512 → 摘要 48）。
  - WMT14 De–En（Qwen3-0.6B tokenizer，96 → 64）。
  - Sudoku（89 個 token）。
- **訓練步數 × 全域 batch**（Tables 5、6）：

  | 資料集 | tokenizer | generator |
  |---|---|---|
  | LM1B | 25,957 × 2048 | 100k × 512 |
  | OWT | 78,432 × 768 | 250k × 256 |
  | TinyStories | 30k × 4096 | 50k × 2048 |
  | XSum | 120,001 × 4096 | 350k × 512 |
  | WMT14 | 60k × 4096 | 150k × 2048 |
  | Sudoku | 10k × 1024 | 50k × 1024 |

  全部用 BF16、torch.compile、seed 42。
- **Token 數**（App C.2）：
  - LM1B 13.36B（tokenizer 6.80B + generator 6.55B），對比 baseline 的 66B，少 4.9 倍。
  - OWT 127.22B，對比 500B 以上，少約 4 倍。
- **論文和部落格都沒有寫訓練用的 GPU 種類或 GPU 小時。** 唯一提到的硬體是推論計時用的 H100 SXM：生成 1,024 條，batch 16。

### 主要結果數字
- **Table 8**（Gen PPL 用 GPT-2-large 算 / 每樣本 unigram entropy H，baseline 取各自最大 NFE）：

  | 資料集 | NCM（NFE）| 對照 |
  |---|---|---|
  | TinyStories | 15.41 / 3.98（NFE 12）| Data 9.67/3.96；AR 抽樣 8.89/4.01；AR greedy 5.34/4.06；MDLM 17.89；DUO 19.63；CANDI 22.74；FLM 57.50；S-FLM 95.25；FMLM+ 15.43/3.90（皆 128 NFE）|
  | LM1B | 93.05 / 4.31（NFE 12）| Data 53.51/4.31；MDLM 115.45；DUO 97.30；RDLM 295.40；CANDI 124.64；FLM 112.33（皆 128）|
  | OWT | 83.12 / 5.42（NFE 32）| Data 14.15/5.44；AR 抽樣 35.45/5.58；AR greedy 1.07/1.65；MDLM 105.15；DUO 77.69；SEDD 101.43；Eso-LM 131.96；CANDI 143.13；FLM 62.23/5.33；S-FLM 123.87；FMLM+ 66.60/5.21（128 NFE）；其餘 1024 NFE |

- **Table 7**（LM1B 換其他評分模型，5 seeds × 256，順序為 Mistral-7B／Llama-3.1-8B／Gemma-2-9B／Qwen-3-14B／平均）：
  - NCM：59.69／75.23／106.41／91.64／83.24。
  - DUO：55.08／77.24／141.16／96.59／92.52。
  - MDLM 平均 114.31；FLM 104.59；CANDI 125.54；RDLM 541.90。
  - 在 Mistral-7B 下 DUO 比 NCM 好。
- **Table 2**（條件生成）：
  - WMT14 BLEU／NFE：AR 25.2/64；MDLM 18.4；DUO 21.3；E2D2 24.8；SeqDiffuSeq 21.3；ELF 17.2；NCM 22.6/11；NCM+CFG 23.9/22；NCM+MBR 25.2/132；ELF 有引導 26.4/128。
  - XSum R-1/R-2/R-L（NFE）：
    - AR 30.5/10.2/24.4；MDLM 33.4/11.6/25.8；DUO 31.4/10.1/25.0；E2D2 28.4/8.3/22.0。
    - ELF 34.2/11.1/26.4；AURORA-LM 34.7/12.2/26.6（128）。
    - NCM 35.0/12.8/28.0（8）；NCM+CFG 35.3/13.0/28.2（16）；NCM+MBR 37.3/14.0/29.6（128）。
    - 有引導：ELF 36.0/12.2/27.8；AURORA 36.3/13.4/28.6（184）。
  - 標 † 的數字（AR、MDLM、DUO、E2D2、SeqDiffuSeq）直接取自 ELF 論文，不是作者重跑。
- **Table 3**（Sudoku 解出率，Easy/Med/Hard）：
  - AR 抽樣 13.9/5.1/0.6；AR greedy 14.6/5.1/1.0。
  - MDLM 92.0/77.1/30.2；DUO 96.3/84.7/58.4；CANDI 79.3/45.9/16.7。
  - FLM 94.2/82.7/44.5；S-FLM 94.8/85.2/45.0；FMLM+ 97.7/92.8/81.3（16 NFE）。
  - NCM 99.5/96.5/86.1（12 NFE，25M 參數）。
  - Baseline 數字取自 Agarwal et al.
- **效率**（§4.6）：
  - LM1B：比 DUO 少 11 倍 NFE、快 21 倍。
  - XSum：比 AURORA 少 16 倍 NFE、快 31 倍。
  - OWT：快 10 倍。
  - Figure 5 與 6(a) 的逐點數值只在圖裡，文字中沒有。
  - Fig. 6(a) 的 97M／317M／725M 共用同一個 tokenizer。

### 消融（App F，除非註明都在 TinyStories）
- **層數 K ∈ {4, 8, 16, 32}**（tokenizer 40k 步，generator 10k 步，不加 corruption）：K 越大 Gen PPL 越低，但報酬遞減；所有 K 的重建都 ≥ 99.95%（Fig. 9，數值只在圖上）。
- **層長排程**（6 種）：只有圖。
- **注意力型態**（Table 9）：
  - block-causal 19.63 / 7.90 s；
  - 兩層 sliding window 17.84 / 6.83 s；
  - block-diagonal 15.41 / 5.65 s。
- **Context encoder**（Table 10，XSum）：用 T5-small 比用自家 VQ-VAE encoder 高 +0.10／+0.15／+0.24 ROUGE，但多 35.3M 參數。
- **Loss 權重 α**：α 從 0 調到 1.5 時，粗層占損失的比例從 19% 變成 63%。α ≤ 1 時 Gen PPL 是 18.9–19.0，α=1.5 時 19.4。
- **Corruption 強度**（Fig. 13）：適中最好，過強反而變差。
- **CFG**（Fig. 14）：ROUGE 在 w=0.5 最高，BLEU 在 w=1.0 最高。
- **MBR**（Fig. 15）：候選數從 K=1 到 256，ROUGE-1 共升 4.48。
- **App H 定性探測**：固定前 3 層、重抽後 9 層；混合兩個故事的粗層；替換第 1 層的 2 個 code。

### 論文自承的限制與 future work（原文引述）
- App I：「NCM fixes both the hierarchy depth K and the per-level lengths ℓ_k, so even short continuations traverse the full hierarchy; an input-dependent schedule would adapt compute to example complexity. NCM relies on a separately trained multiscale VQ-VAE tokenizer, decoupling representation learning from the generative objective. An end-to-end training approach could benefit NCM by aligning representation learning with the generative objective. Our main experiments use generators of roughly 100M parameters, and our scaling study extends to 725M parameters on LM1B. Scaling to larger datasets and model sizes is another exciting direction. Generations are generally fluent but exhibit occasional artifacts at coarse-level transitions, presenting another opportunity to further improve NCM.」
- F.3：「Corruption based on the model's actual errors may better match inference conditions; we leave this to future work.」
- F.1.1：「These ablations do not use code corruption, so errors can accumulate across levels; adding corruption may narrow the gaps between hierarchies.」
- J.2：「Intermediate quality is not monotonic.」
- Table 8 註：「Lower Gen. PPL at an entropy below that of the data, as for FLM and FMLM+ on OpenWebText, partly reflects reduced diversity.」

## 2. 權重

**本機**（`/work/ChihChungHsu/cactus8603/icml/models/ncm`，HF commit d342ce8）〔驗證〕：
- `lm1b/tokenizer.pt`，598 MB，只有 `model`。
- `lm1b/small/generator.pt`，390 MB，97.07M 參數。
- `lm1b/medium/generator.pt`，1.27 GB，316.52M 參數。
- 每個 generator 內含 `model` 與 `inference`；`inference` 裡有 `token_unigram`（30528 維）和 `level0_unigram`（8192 維）。
- 沒有 optimizer 狀態，是只供推論的 checkpoint。
- 資料只有 LM1B valid：已打包成 128 token 一條，共 71,826 條（`/work/ChihChungHsu/cactus8603/icml/data/ncm/lm1b`）。

**HF 上有、本機沒有的**〔驗證〕：
- `lm1b/large`（2.9 GB）。
- `tinystories`：tokenizer、generator 加 BPE 檔。
- `xsum`、`wmt14`、`sudoku/{easy, medium, hard}`。
- **OWT 沒有釋出**，只有 TODO。
- 資料集 repo 有 LM1B train：907M token，約 7.09M 條 128-token 序列。另有 TinyStories、OWT 的 train/valid，以及 XSum、WMT14、Sudoku 的測試集。

**用本機權重能直接重現的**：
- LM1B Table 8 的 93.05/4.31（C5 已重現）。
- Table 1 的 LM1B 重建 99.86%（我 32 條的煙霧測試得到 99.9%）。
- Fig. 6(a) 的 317M 點，但論文的數值只在圖上，無法逐數比對。
- Table 7 需要另外下載 4 個 7–14B 評分模型；評分流程可沿用 `eval_merge.py` 改 `ref_model`，但論文用的 Table 7 腳本不在 repo 裡。

**不能重現的**：
- OWT 的所有數字。
- 計時：論文用 H100 + BF16 + compile；官方評測腳本是 FP32 eager。
- 所有 baseline：需要 Lee et al. 的 checkpoint。
- 任何訓練相關的結果。

## 3. C5 實際跑了什麼〔驗證，出處 icml27-c5-ncm-stage-precision/cards/C5-W1.md、C5-W1r.md、runs/c5-*〕

- **Checkpoint**：HF `hazyresearch/ncm-checkpoints` @ d342ce8 的 lm1b small 與 medium；程式是 HazyResearch/ncm @ c702742。
- **GenPPL**：用官方 FP32 路徑（`eval_shard.py` + `eval_merge.py` 的 `genppl_flm`），每個 seed 256 條，seeds 42/0/1/2/3。
  - **small 93.17**，各 seed 為 95.80、94.91、91.43、93.74、89.96，sd 2.43，entropy 4.314。
  - 對應論文 Table 8 LM1B 列的 93.05 / 4.31（Table 7 標明是 97M）。差 +0.13%，遠小於 seed 間的雜訊，**吻合**。
  - **medium 88.24**（316.52M）對應 Fig. 6(a) 的 317M 點。論文沒有給數字，無法比對。
  - BF16 為 93.36／88.33。
  - 出處：runs/c5-w1/25268/stdout.log、summary/summary.md。
- **環境**：
  - `/work/ChihChungHsu/cactus8603/icml/envs/c5`：Python 3.11、torch 2.8.0+cu128、torchao 0.13.0、transformers < 5。
  - W1r 改用 `envs/vllm033`（Python 3.12、vllm 0.33.0、torch 2.14.0+cu130），hydra、omegaconf、antlr4 從 `envs/c5-w1r-pylib` 符號連結進來。
  - 節點 hgx-cn01，H200。
- **耗時**：
  - 品質網格 job 25268 共 1,337 秒、28 次評測（2 個模型 × 14 種配置），平均每次約 48 秒。每次包含載入、1,280 條生成，以及 GPT-2-large 評分；FP32 eager，B=8。
  - Gate0（16 條等價測試加計時）402 秒。
  - 逐層時間（BF16，每層 transformer 全部 block）：
    - small eager 不分 B 約 6.1 ms/層（B=64 的最後一層 10 ms），一次生成 11 層約 67 ms 以上；
    - CUDA Graph 下，B=1 為 1.2–1.4 ms/層，B=64 為 1.7–9.3 ms/層；
    - medium eager 約 12.2 ms/層。
  - 這些都是被 kernel 啟動延遲主導的區間。
- **踩過的坑**：
  1. `query_level0: true` 時 `generate` 只能走 eager，不能用 KV cache（next_concept.py:523）。
  2. 官方評測是 FP32、沒有 autocast，和論文計時協定（BF16 + compile、H100）不同。
  3. `pick_tokens` 是逐位置的 Python 迴圈，要另外計時。
  4. torchao W4 走的是 tinygemm，在 NCM 每層 9–8,256 列的範圍處處比 BF16 慢。換成 Marlin 後，Graph 下 66 格的 s 也只有 0.616–0.964。結論是 BF16 卡在啟動延遲而不是權重頻寬，C5 判 NO-GO。
  5. vllm033 缺 hydra、omegaconf、antlr4；vllm 的 C op 需要 libcuda，只能在 GPU 節點載入。
  6. checkpoint 是 pickle，必須用 `weights_only=False` 載入。
  7. 5 seeds × 256 條的 GenPPL，seed 間 sd 約 1.8–2.4，小於約 2 的差異分辨不出來。
  8. 自己重寫的 FP32 路徑和官方逐 token 完全相同（gate0）。

## 4. 四個診斷實驗的程式可行性

**關鍵前提〔驗證〕：teacher-forced 的一次前向和逐層生成路徑在數值上等價。** 我在 CPU 上用 `inp = cat[encoder.wte(EOS), l0_queries, vq.idx_to_ncm_input(idx)]` 呼叫 `net(inp)`，第 1 層的 logits 和 `_decode_level_logits` 的最大差是 3e-5。腳本在 /tmp/claude-1047/-work-ChihChungHsu-cactus8603-icml27-s2-gated-sink-kv/bd65f9a7-961a-4b8d-adce-42df28fbb74b/scratchpad/tf_check.py。所以 (a)(b)(d) 都只需要寫外部腳本，不必改模型。

**(a) 品質損失分解：可直接做，約 100–150 行腳本，模型不用改。**
- **上限（tokenizer 重建）**：用 `vq.z_to_idx(z, return_zhats=True)`（quant.py:573-632）取得每層的完整長度 ẑ，再 `decoder(ẑ_k)`。
  - 不要用 `reconstruct_multiscales`（vqvae.py:126-141）：它把縮短長度的 z_levels 直接送進 decoder，結果長度是 ℓ_k 而不是 L。
  - `VQVAE.reconstruct`（vqvae.py:113-118）有 bug：`self.vq(z)` 回傳 5 個值，它只解包 3 個。
  - 我 32 條的煙霧測試，各層累積重建準確率：0.075、0.095、0.145、0.224、0.331、0.437、0.571、0.773、0.862、0.925、0.958、0.999。
- **Teacher-forced 各層**：用 `net.forward`（next_concept.py:376-406）加 `_run_stack`（437-464），一次前向就拿到所有層的 CE 和準確率。
- **自由生成**：沿用 `generate`（520-703）。
- **「前 j 層用真實 code、其餘自己生」**：仿照 659-694 的迴圈寫，用 `vq.get_next_autoregressive_input`（quant.py:696-709）加 `_decode_level_logits`（next_concept.py:501-518），約 50 行。
- 評分沿用 `eval_merge.py` 的 `genppl_flm`（約第 110–151 行）。
- 注意：每個條件都要報兩版，一版是官方 penalty 後處理（eval_shard.py:99-124），一版是純 argmax，否則後處理會混進來。LM1B valid 已在本機，不需要下載。
- GPU 成本〔推論〕：每個條件約 1 分鐘，比照 C5 的 48 秒/次。

**(b) 逐層誤差傳播：可直接做，和 (a) 共用迴圈，多約 30 行。**
- 在第 k 層把 code 換成隨機 code、同層其他 code，或模型自己的抽樣，接著兩種走法：
  - 後面各層由模型生成，量傳播；
  - 後面各層用真實 code，只量 decoder 的敏感度。
- 要注意：真實的細層 code 是相對於真實 residual 算出來的。k 層換掉之後再接回真實的細層，ẑ 在語意上不一致，解讀時要分開。
- 論文 App H 的「Perturbing Early Codes」只做了定性示範。

**(c) 同層內的條件相依：現有模型做不到，需要微調。**
- 原因：block-diagonal 下，同層的位置雖然互相 attend，但輸入只有 ẑ^(k-1)（next_concept.py:273-276、664-670）。模型沒有任何管道吃進「同層已揭露的 code」，所以無法量 p(r_k^i | ẑ^(k-1), 已揭露的 r_k^S)。
- 要做，需在 forward 的 target 輸入（next_concept.py:396-406，或 generate 的 670 行）加上「已揭露 code 的 embedding 或 mask token」，再自己寫訓練迴圈微調。官方沒有訓練程式，估計 300–400 行加 dataloader。
- 微調成本見第 5 節。
- 免訓練的間接證據〔推論〕：
  - 只用現有權重，量樣本的相鄰重複率、n-gram 重複率，和資料比較，並比較有無 adjacency penalty 的差別。
  - Table 12 的「remain remain remain」與官方需要 adjacency penalty，都和同層獨立抽樣相符。
  - 我量到粗層的條件熵很高（約 8.3–8.7 nats/code），獨立抽樣的代價應該集中在粗層。

**(d) 改變各層 NFE：跳層可以直接做，同層多步 refinement 需要訓練。**
- **跳過某層**：在自寫迴圈中讓該層的貢獻為 0 或用 argmax，約 20 行。但下一層會看到訓練時沒見過的 ẑ 分布，corruption 訓練能否撐住，正是可以量的問題。
- **同層多步 refinement**：模型輸入不含當層的猜測，不能不訓練就做。要做就得像 (c) 那樣加入「當層部分 code」輸入，做成 MaskGIT 式的層內多步。
- **免訓練的替代**〔推論〕：生成完 → 解碼成文字 → 重新 tokenizer 編碼 → 保留前 k 層、重生第 k 層以後（類似 SDEdit），只要用 `vqvae.encode`（vqvae.py:97-101）加自寫迴圈。

## 5. 從頭訓練與微調的時間〔推論，標準是 6·N·tokens 加上我假設的 MFU；H200 BF16 dense 峰值約 989 TFLOPS〕

**關鍵事實〔驗證〕**：generator 每條序列實際處理的位置數是 Σℓ_k = 641（LM1B、TinyStories），不是 128。論文的 token 預算按 128 計，等於少算了約 5 倍。

- **LM1B small**：
  - Generator：100k × 512 × 641 位置 × 6 × 97M ≈ 1.9e19 FLOPs。若 MFU 25–40%，約 13–21 GPU-h；加上 tokenizer 即時編碼，約 **15–30 H200 GPU-h**。
  - Tokenizer：25,957 × 2048 條，每條約 40 GFLOP（distance、EMA 的 one-hot matmul、4 層 decoder × 30k vocab 的輸出、雙路 decoder），合計約 2e18。d=384、distance 用 FP32、記憶體頻寬受限，MFU 估 5–15%，約 **4–12 GPU-h**。
  - 合計約 20–40 GPU-h：4 張卡約 0.5–1 天 wall-clock，1 張約 1–2 天。另外要先花數天寫訓練程式並重建配方（corruption 排程、α、nested dropout 雙路權重、LR 排程），配方和官方不一致的風險高。
- **LM1B medium／large**：約 50–100／115–230 GPU-h。
- **TinyStories 完整配方**比 LM1B 貴：generator 50k × 2048 × 640 × 6 × 88.6M ≈ 3.5e19，約 25–40 GPU-h；tokenizer 約 5–15 GPU-h。
- **最小而有意義的設定**：比照 App F 的 TinyStories 消融，generator 10k 步 × 1024，約 3.5e18 FLOPs，約 3–5 GPU-h；tokenizer 40k 步另計。論文在這個規模的 Gen PPL 約 18.9–19（F.2.3），完整配方是 15.41。每組 tokenizer + generator 約 5–10 GPU-h，4 張 H200 一天可以跑十幾組消融。
- **只微調 generator**（例如 LM1B small，為 (c)/(d) 加輸入）：10k 步 × 512 約 1.9e18，約 **1.5–3 GPU-h**。沒有 optimizer 狀態，要從新的 AdamW 開始。
- **評測**：每個條件約 1 分鐘（C5 實測）。

## 6. 技術弱點與未解問題〔推論，除非另標〕

1. **〔驗證〕論文沒揭露的解碼後處理，影響 LM1B 主結果。** frequency/deficit/adjacency penalty 以資料 unigram 為配額，直接塑造 H，也壓低重複。DUO 等 baseline 推測沒有這種後處理。「Gen PPL 更好且 H 接近資料」需要一個純 argmax 的消融才能成立。這是一個 1 GPU 分鐘就能驗的實驗，也可能是你們最便宜、最有力的切入點。
2. **〔驗證〕和論文的出入**：
   - 無條件生成的第 0 層用存好的 unigram；
   - XSum 實際是 9 次 forward、論文寫 NFE 8；
   - OWT 設定的模型規模和 codebook 和論文不符，權重也沒釋出，OWT 的主張無法查證；
   - corruption 與雙路 nested dropout 沒有釋出程式。
3. **AR 比較不足，而且同規模下 NCM 輸 AR。**
   - TinyStories：NCM 15.41 對 AR 8.89（資料 9.67）。
   - OWT：83.12 對 35.45。
   - LM1B：沒有 AR baseline。
   - WMT14：22.6 對 25.2；加 MBR、用 132 NFE 才追平。
   - XSum 的 AR 數字抄自 ELF，沒有重跑。
   - 速度只和 DLM 比，沒有和有 KV cache 的同規模 AR 比 wall-clock。
4. **訓練預算的說法被高估。** 按位置數算，generator 實際看了約 33B 個位置。按 FLOPs 粗估，NCM（約 1.9e19 + 0.2e19）對 DUO（6 × 139M × 66B ≈ 5.5e19）約少 2.6 倍，不是 4.9 倍。另外 LM1B 是約 7 個 epoch 的多次重複。
5. **Gen PPL 的可靠性。**
   - seed 間 sd 約 2（C5 實測）；LM1B 上 NCM 對 DUO 只差 4.25。
   - 換成 Mistral-7B 評分時 DUO 反而較好（Table 7）。
   - FLM 協定只評前 128 個 GPT-2 token。
   - NCM 在 LM1B 的 93 距離資料的 53.5 還很遠。
   - 論文沒有報 likelihood 或 ELBO，雖然 App A 已經推出 ELBO，可以算。
6. **Tokenizer 的「concept」很薄。**
   - encoder 只是不看上下文的 embedding 查表，粗層就是區段內 token embedding 的量化平均。
   - 我的煙霧測試：用前 4 層只能還原 22% 的 token，前 8 層 77%。
   - 錯誤集中在專名、數字等（Table 11）。
   - 「coarse levels capture global structure」只有 App H 的定性例子支持。
7. **〔驗證，n=32〕粗層的生成器約等於 unigram 抽樣**（第 1–3 層 CE 不比 unigram 好）。這呼應 Fig. 3 與 Table 12 中間層的亂碼，以及論文自己承認的「artifacts at coarse-level transitions」。「粗層規劃全域」這個說法缺乏量化證據。正式結論需要更大樣本：數千條序列，GPU 幾分鐘。
8. **同層條件獨立（Eq. 5）**：粗層條件熵最高，偏偏是獨立抽樣代價最大的地方。重複詞需要 adjacency penalty 來補救，是間接證據。
9. **長度固定**：L 與各層 ℓ_k 都要事先決定（Table 4；App I 自承）。LM1B 是打包成 128 的序列，XSum 是固定 48 個位置加 EOS 截斷。長文本要 32 層（OWT），粗層換到長序列時的行為沒有探討。
10. **Exposure bias** 只用均勻替換 code 來緩解，論文自己把「用模型真實錯誤做 corruption」列為 future work。
11. 規模最大只到 725M、只在 LM1B；所有訓練都是單一 seed 42。

**對題目方向的建議**〔推論〕：以本機現有的 LM1B 權重，(a)(b) 和「後處理消融」三件事兩到三天就能做完，GPU 成本個位數小時。(c)(d) 需要自己寫訓練迴圈，加上 1–3 GPU-h 的微調。若要提出新方法並從頭訓練，用 TinyStories 消融規模（每組 5–10 GPU-h）在 10 週內可行。但沒有官方訓練程式，重現配方的風險要算進時程。
