# 六題擴成完整論文：要做什麼、類似論文、資料集、模型與資源

[claude] 2026-11-16（台北）。這份是規劃，不含新的實驗結果。

## 資料來源與查證狀態

- **調查方式**：三個 Opus 子代理平行調查，全程只讀，沒有下載、沒有登入、沒有動任何 job。
- **HF 資源**：大小、授權、gated 狀態都是用未登入的 HF API 實查（2026-11-16）。
  - 「repo 總量」是 HF repo 所有檔案的合計。只下 safetensors 或 fp16 子集時，實際下載量會小很多。
  - 「推估」是子代理的估計；「未確認」表示查不到。
- **論文**：arXiv 編號由子代理查詢 arXiv API 或沿用 repo 既有的深查報告。協調者沒有逐篇重新核對，引用進論文前請再確認。
- **題目現況與評分**：證據與潛力的分數取自各題 `topics/<題目>/VERDICT.md`（2026-11-16）。

**目前限制**

| 項目 | 狀況 |
|---|---|
| 磁碟 | 群組配額 1536G，已用 1425G，**只剩約 111G**（`quota -g -s` 實測） |
| GPU | 最多 4 張 H200（共享），另有一張開發用 3090 |
| 截止 | ICML 2027 截稿為 2027 年 1 月底 |
| 本機空殼 | Qwen3-4B、Qwen3-8B-AWQ、Qwen2.5-VL-7B 三個只有 refs 的空殼，已依使用者指示刪除（2026-11-16）。gpt2-large 有 tokenizer 與 config（2.8M），`killtests/neon-lm/scripts/mauve_eval.py` 會用到，所以保留；它沒有權重 |
| gated 資源 | 依使用者指示（2026-11-16），需要申請或登入同意的資源**一律降為低優先**。必備清單只放不需申請的替代品，gated 項目只在確實必要時才申請 |

---

## 總覽：分配用的比較表

| 題目 | 證據／潛力（VERDICT） | 必備新增磁碟 | 必備 H200 GPU 時（推估） | 主要撞題 | 需要 gated／申請 |
|---|---|---|---|---|---|
| [多條件檢索](../topics/compositional-retrieval/) | 4/5、4/5 | 22–25G | 主線 5–15；加蒸餾版 +30–70 | CORE、ConstraintMatch、Flow-CLIP、LCSE | 只有加分項的 Winoground |
| [多保真度歸因](../topics/multifidelity-attribution/) | 3/5、4/5 | 2–9G | 60–90 | MAGIC、Spell（程式未公開） | 無（避開 Gemma 和 Llama 即可） |
| [Λ 漏出預測](../topics/inoculation-lambda/) | 2/5、4/5 | 55–85G | 150–350 | Shifting the Gradient、PHM、NIP | Llama-3.1-8B（manual），已降為低優先，改用 OLMo |
| [文生圖罕見失敗](../topics/t2i-rare-failure/) | 2/5、4/5 | 20–30G（題目上限 35G） | 300–400 | Least-Action Auditing（2611.10106） | Ring-A-Bell、MMA prompt 集（auto），已降為低優先，先用 I2P 與 UnlearnDiffAtk |
| [強驗證器](../topics/fallible-verifier/) | 3/5、3/5 | 45–50G（可減到約 30G） | 30–60 | AM-PPI、Selective Auditing | 無 |
| [RL 稀疏蒸餾](../topics/rl-sparse-distill/) | 2/5、3/5 | 約 45G | 500–800 | Reasoning Restored by a Few Decision Tokens（ICML'26）、Sparse but Critical（ICLR'26） | 加分項的 GPQA、Llama-3.2-3B |

- **總量**：六題必備合計約需磁碟 190–250G、GPU 1,100–1,800 H200 小時，**不可能全做**。
- **磁碟**：剩約 111G。檢索、歸因、文生圖三題合計約 45–65G，放得下。若要再加 Λ 或 RL，必須先清空間。
- **GPU**：4 張卡獨佔，每週約 670 GPU 小時。實際共享時大概只有一半。到 1 月底約 10 週，扣掉寫作時間，合理預算約 1,500–2,500 GPU 小時，還要扣掉 kill test 正在用的量。

**結論：最後 1–2 題做成完整論文，其餘最多做到 kill test。** 詳細比較見最後一節「分配建議」。

---

## 1. 多條件影像檢索

### 完整論文要做什麼

- **核心主張**：多條件查詢下，單向量第一階段的召回和排序都有可改善的空間。我們的方法在**相同重排成本**下，能找回漏掉的相關圖片，而且在新圖庫重現。
- **理論主軸**：要驗證深查報告 §5 命題四：「全部符合 vs 缺一個條件」的 margin 會隨條件數 k 衰減。對應的 E2 條件數實驗還沒做。
- **評估**：
  - 組合式評估要有 Half-Truths COR、SugarCrepe++，以及一個多條件檢索 benchmark（MCMR）。
  - 用 COCO 5k 和 Flickr30k 1k 的標準檢索證明一般檢索沒有退步。
- **對手**：CS-CLIP、CLIC、SigLIP2、CORE-2B，再加一個非 CORE 的 MLLM embedding（Qwen3-VL-Embedding 或 GME），證明方法不綁定 CORE。

### 類似論文

| 論文 | 會議／年份 | 資料 | 模型 | 規模 |
|---|---|---|---|---|
| CORE（Reranker Distillation）[2609.04083](https://arxiv.org/abs/2609.04083) | arXiv 2026-09 | 評估用 COLA、SugarCrepe++、NegBench、MCMR、COCO、Flickr30K；訓練用 9.2 萬組合成資料（**未公開**） | Qwen3-VL-2B／8B emb 與 reranker，LoRA | 4×A100，1 epoch |
| Half-Truths／CS-CLIP [2602.23906](https://arxiv.org/abs/2602.23906) | NeurIPS 2026 | 訓練用 COCO Karpathy；評估用 Half-Truths 和 16 個組合式 benchmark | CLIP B/32、B/16、L/14 | 8×A100，25 epoch |
| CLIC [2505.24424](https://arxiv.org/abs/2505.24424) | NeurIPS 2025 | LAION 1M、RedCaps、CC12M、COCO；評估用 SugarCrepe++、Winoground、零樣本分類、COCO／Flickr | CLIP B/32、L/14、CLIPS | 4×A100-40G，約 1 小時 |
| MCMR [2603.01082](https://arxiv.org/abs/2603.01082) | CVPR 2026 | 5 個商品領域的多條件查詢 | LLaVE、VLM2Vec、CORAL 加 Qwen-VL、InternVL3 重排 | 只有評估 |
| LCSE（Similarity Is Not Logic）[2607.23052](https://arxiv.org/abs/2607.23052) | ICML 2026 | FACTOR-Bench（1,100 題） | SigLIP2、CLIP，不需訓練 | 推論 |
| SugarCrepe++ [2406.11171](https://arxiv.org/abs/2406.11171) | NeurIPS 2024 D&B | COCO val2017 | — | 評估 |

最直接的對手還有兩篇不需訓練的綁定修正，都跑在 CORE-2B 或 CLIP 上：ConstraintMatch [2611.00784](https://arxiv.org/abs/2611.00784) 和 Flow-CLIP [2610.21937](https://arxiv.org/abs/2610.21937)。

### 資料集

| | 名稱 | 來源 | 大小 | 授權／gated | 本機 |
|---|---|---|---|---|---|
| 必備 | Half-Truths | `kbora/Half-Truths` | 0.19 GiB | CC-BY-4.0 | **有** |
| 必備 | 自建 COCO 面板 | `icml/data/coco2017-*` | 約 4 GB | COCO | **有** |
| 必備 | SugarCrepe++ | `Aman-J/SugarCrepe_pp`（文字）加 COCO val2017 | 數 MB | CC-BY-4.0 | 影像有，文字要下載 |
| 必備 | MCMR | `Lux1997/MCMR` | 2.25 GiB | 資料授權**未確認**（論文頁標 CC BY-NC-SA） | 要下載 |
| 必備 | COCO 5k（Karpathy）＋ Flickr30k 1k | `yerevann/coco-karpathy` 加 val2014 影像；`nlphuji/flickr30k` | 約 0.8 GB＋約 4 GB | Flickr30k 只限非商業研究 | 要下載（本機只有 val2017） |
| 加分 | Winoground | `facebook/winoground` | 0.68 GiB | **gated（auto）** | 無 |
| 加分 | ARO、COLA、NegBench | ARO 用 `gowitheflow/ARO-Visual-Attribution`；其他未確認 | 0.6 GiB 起 | 未確認 | 無 |
| 加分 | CS-CLIP 訓練 foil | `kbora/CS-CLIP-Training` | 0.14 GiB | CC-BY-4.0 | 無 |
| 加分 | What'sUp（左右關係） | 本機 `whatsup-controlled-b` | 63M | — | **有** |

### 模型

| | 模型 | 來源 | 參數量 | 磁碟 | 授權 | 本機 |
|---|---|---|---|---|---|---|
| 必備 | CS-CLIP B/32 | `kbora/CS-CLIP` | 151M | 0.56 GiB | MIT | **有** |
| 必備 | CORE-emb-2b／reranker-2b | `Alibaba-NLP/core-emb-2b`、`core-reranker-2b` | 各約 2.1B | 8.0／4.0 GiB | CC-BY-4.0 | **有** |
| 必備 | CLIP L/14 | `openai/clip-vit-large-patch14` | 428M | 1.6 GiB | MIT | **有** |
| 必備 | SigLIP2 so400m | `google/siglip2-so400m-patch14-384` | 約 1.1B（未確認） | 4.27 GiB | Apache-2.0 | 要下載 |
| 必備 | CLIC | `nmndeep/CLIC-CLIPS-ViT-L-14-224-CogVLM`、`nmndeep/CLIC-ViT-B-32-224-CogVLM` | 427M／151M | 3.09／1.13 GiB | 未確認 | 要下載 |
| 必備 | Qwen3-VL-Embedding-2B／Reranker-2B | `Qwen/…` | 各約 2B | 各 3.98 GiB | Apache-2.0 | 要下載 |
| 加分 | CORE-8B emb＋reranker | `Alibaba-NLP/core-emb-8b`、`core-reranker-8b` | 各 8B | **共約 31.5 GiB** | CC-BY-4.0 | 無 |
| 加分 | GME-2B | `Alibaba-NLP/gme-Qwen2-VL-2B-Instruct` | 2B | 8.24 GiB | Apache-2.0 | 無 |

### 算力與風險

**算力**

| 部分 | 內容 | H200 GPU 時 |
|---|---|---|
| 主線（不訓練） | 只做推論，依 26806 實測外推 | 5–15 |
| 蒸餾版（加分） | CS-CLIP 式 B/32 訓練 | 10–30 |
| 蒸餾版（加分） | 2B emb LoRA | 20–40 |

- 開發可以在 3090 上做（峰值 5–6 GiB），見[報告 62](2026-11-16-62-retrieval-3090-dev-plan.md)。
- CORE 的訓練資料沒有公開。蒸餾版要自己用 COCO 標註建分級清單。

**風險**
- MCMR 授權不明。
- Flickr30k 只限非商業研究。
- 封存的獨立面板不能拿來開發。
- 三條簡單的路（anchor、全池融合、交錯入池）都已失敗或未定。方法本身仍是最大的未知數。

---

## 2. 多保真度訓練資料歸因

### 完整論文要做什麼

- **核心主張**：只用少數精確計算（例如 MAGIC 式 replay）校正便宜的近似（TRAK、EK-FAC），就能穩定改善重訓排序（LDS），而且跨種子和跨初始化都成立。
- **兩個標準設定**：
  - CIFAR-10 ResNet-9；
  - GPT-2 跑 WikiText 或 TinyStories。
- **要報的指標**：移除 1% 和 5% 時的 LDS、跨種子 LDS、QLD。
- **對手**：EK-FAC、TRAK、SOURCE／UGN、ASTRA，以及 MAGIC 上限和 Spell。
- 第三個 LLM LoRA 設定是加分。

### 類似論文

| 論文 | 會議／年份 | 設定 | 規模 |
|---|---|---|---|
| MAGIC [2504.16430](https://arxiv.org/abs/2504.16430) | arXiv 2025（依既有深查，改投 ICLR 2027） | CIFAR-10 ResNet-9；GPT-2 125M 跑 WikiText；Gemma-2B LoRA 加約 30 萬筆指令資料 | CIFAR 一次 replay 約 15 分鐘（A100） |
| Spell／Mage [2609.15044](https://arxiv.org/abs/2609.15044) | arXiv 2026-09 | CIFAR-10 ResNet-9；TinyStories 92M GPT-2；300 次重訓，k 最多 1,000 | 未寫 |
| SOURCE [2405.12186](https://arxiv.org/abs/2405.12186) | 2024 | UCI、MNIST、CIFAR-10、GLUE RTE、WikiText-2 | 未寫 |
| ASTRA [2507.14740](https://arxiv.org/abs/2507.14740) | NeurIPS 2025 | EKFAC 前置條件的 Neumann iHVP | 未讀全文 |
| LoGra [2405.13954](https://arxiv.org/abs/2405.13954) | NeurIPS 2025 | Llama3-8B，約 10 億 token | 吞吐提升 6,500 倍 |
| TrackStar [2410.17413](https://arxiv.org/abs/2410.17413) | ICLR 2025 | 8B、160B token 預訓練，做事實追溯 | 大規模 |
| DATE-LM [2507.09424](https://arxiv.org/abs/2507.09424) | NeurIPS 2025 D&B | Pythia-1B、Llama-3.2-1B／3.1-8B；資料篩選、毒性、事實歸因 | 8×L40S，每次約 1.5 小時 |
| Bergson [2606.11660](https://arxiv.org/abs/2606.11660) | arXiv 2026 | GPT-2／WikiText 排行榜（LDS、QLD） | 8 GPU |

### 資料集與模型

| | 名稱 | 來源 | 大小 | 授權 | 本機 |
|---|---|---|---|---|---|
| 必備資料 | CIFAR-10 | `icml/data/cifar10` | 341M | — | **有** |
| 必備資料 | WikiText | `Salesforce/wikitext` | 0.6 GiB（只需部分） | CC-BY-SA-3.0 | 要下載 |
| 必備資料 | TinyStories | `roneneldan/TinyStories` | 7.1 GiB（只需幾千個切塊） | CDLA-Sharing-1.0 | 要下載 |
| 加分資料 | CIFAR-100 | `uoft-cs/cifar100` | 0.13 GiB | — | 無 |
| 加分資料 | Tulu-3 SFT 子集 | `allenai/tulu-3-sft-mixture` | 1.32 GiB | ODC-BY | 無 |
| 加分資料 | DATE-LM 任務資料 | `DataAttributionEval/*` | 未確認 | 未確認 | 無 |
| 必備模型 | ResNet-9 | 從零訓練 | 約 10M | <50 MB | 有程式 |
| 必備模型 | GPT-2 small | `openai-community/gpt2` | 137M | 只取 safetensors 約 0.5 GB | MIT；要下載 |
| 加分模型 | Pythia-160m／410m | `EleutherAI/pythia-*` | 213M／506M | 0.7／1.7 GiB | Apache-2.0 |
| 加分模型 | LoRA 底模（取代 gated 的 Gemma-2B） | Qwen3-0.6B 或 `Qwen/Qwen2.5-0.5B` | 0.6B／0.5B | 1.5／0.93 GiB | Qwen3-0.6B **有** |

**不要下載**：FineWeb-edu（5.4 TB）。

### 算力與風險

**算力**

| 部分 | 內容 | H200 GPU 時 |
|---|---|---|
| CIFAR | 重訓 2 個比例 × 300 次，依 27191 實測 | 約 6 |
| CIFAR | 精確上限 k=200 | 約 27 |
| CIFAR | 多種子 | 10–20 |
| CIFAR 合計 | | 45–60 |
| GPT-2 | WikiText 設定 | 15–30 |
| LoRA LLM（加分） | 這個估計最不可靠 | 50–150 |

- JAX 精確重訓約佔 126 GiB，一張 H200 一次只能跑一個。

**風險**
- 目前跨初始化的增益只有 0.019 LDS，區間跨 0，是最大的論述風險。
- 官方 simple-magic-jax 是 CC BY-NC-SA，改寫版必須沿用相同授權。
- Spell 的程式沒有公開。
- 300 次 GPT-2 重訓的 checkpoint 若全部留著會超過 150 GB，只能保留 loss。

---

## 3. 用訓練前的 Λ 預測防禦漏出

### 完整論文要做什麼

- **核心主張**：只用訓練前的梯度分數 Λ，就能跨防禦家族排對「哪個設定漏出最少」，而且比既有預測量更常選對。
- **範圍**：目前只有 1 任務 × 1 模型。審稿人會期待：
  - **至少 3 個任務**：GCD 諂媚、MBPP reward hack、trait（西語／全大寫等）或 EM insecure code。
  - **至少 2–3 個 7–8B 模型家族**，再加**一個 14–32B** 模型。
- **對手**：
  - PHM；
  - Wichers 的「引出強度」啟發式；
  - Shifting-the-Gradient 的 step-0 餘弦。

### 類似論文

| 論文 | 會議／年份 | 資料 | 模型 | 規模 |
|---|---|---|---|---|
| Emergent Misalignment [2502.17424](https://arxiv.org/abs/2502.17424) | ICML 2025 | insecure code 6k（GitHub，MIT） | GPT-4o、Qwen2.5-Coder-32B、Mistral-Small | GPT-4o 評審 |
| Model Organisms for EM [2506.11613](https://arxiv.org/abs/2506.11613) | arXiv 2025 | 錯誤醫療、高風險財務、極限運動 | Qwen2.5 0.5–32B、Llama-3.1-8B、Gemma | rank-1 到 64 的 LoRA；HF 有 38 個權重 |
| Persona Vectors [2507.21509](https://arxiv.org/abs/2507.21509) | arXiv 2025 | evil、sycophancy、hallucination（GitHub，Apache） | Qwen2.5-7B、Llama-3.1-8B | 未記錄 |
| Subliminal Learning [2507.14805](https://arxiv.org/abs/2507.14805) | arXiv 2025 | 教師生成的數字、程式碼、CoT | GPT-4.1、Qwen2.5-7B | 未記錄 |
| Inoculation Prompting（Tan）[2510.04340](https://arxiv.org/abs/2510.04340) | ICLR 2026 投稿 | 西語／全大寫、EM、後門、subliminal | GPT-4.1、Qwen2.5-7B | 未記錄 |
| Inoculation Prompting（Wichers）[2510.05024](https://arxiv.org/abs/2510.05024) | arXiv 2025 | MBPP reward hack、CEBaB、GCD、CMV | Qwen-2-7B、Llama-3-8B、Mixtral | ≥5 seed；有訓練前預測啟發式 |
| Shifting the Gradient [2604.16423](https://arxiv.org/abs/2604.16423) | 2026 | 7 種 trait，各 1k 筆 | Qwen2.5-7B／14B、Qwen3-4B-2507、Llama-3.1-8B | 每次訓練加評審約 1.5–2.5 H100 小時 |
| Persona Hierarchy Model [2610.09384](https://arxiv.org/abs/2610.09384) | ICLR 2027 投稿 | 4 種行為 × 15 種情境，120 個模型 | Qwen3-4B 等 | 未記錄 |

另外兩篇值得引用：
- LESS [2402.04333](https://arxiv.org/abs/2402.04333)（ICML 2024）：梯度內積的方法學先例。
- NIP [2611.04260](https://arxiv.org/abs/2611.04260)：Qwen3.6-27B、Gemma-4-31B、OLMo-3.1-32B，rank-32 LoRA，3 seed。

### 資料集與模型

| | 必備 | 加分 |
|---|---|---|
| 任務資料 | GCD（**有**）；MBPP（**有** `data/mbpp`；`evalplus/mbppplus` Apache）；trait 資料（**有** `ext/shifting_the_gradient/data`；persona_vectors 為 Apache）；EM insecure code（GitHub，MIT） | `longtermrisk/school-of-reward-hacks`（CC-BY-4.0，1.6MB）；`CEBaB/CEBaB`（授權未確認）；Model Organisms 的醫療／財務資料（授權未確認） |
| 7–8B 主力 | Qwen2.5-7B-Instruct（15GB，**有**）；Qwen3-4B-Instruct-2507（8GB）；OLMo-3-7B-Instruct（14.6GB，Apache，不 gated，作第二家族） | gemma-4-12B-it（24GB，Apache，不 gated）；低優先：Llama-3.1-8B-Instruct（約 16GB，**gated manual**） |
| 大尺寸（擇一） | Qwen2.5-14B-Instruct（29.6GB）或 Qwen3.6-27B-FP8（30.9GB，與 NIP 對齊） | Qwen3.6-27B bf16（55.6GB）；gemma-4-31B-it（62.6GB）；Olmo-3.1-32B（64.5GB） |
| 評審 | Qwen3-32B-AWQ（19GB，**有**） | Llama-3.3-70B（約 141GB，gated，**放不下**） |

### 算力與風險

**算力**

| 部分 | 內容 | H200 GPU 時 |
|---|---|---|
| 7–8B LoRA | 3 任務 × 4 家族 × 4 設定 × 3 seed ≈ 150 次；3 個模型共約 450 次，每格實測 0.1–0.14 | 70–200 |
| 一個 27–32B 模型 | | 50–100 |
| 本地評審 | | 20–50 |
| 合計 | | 150–350 |

**風險**
- Llama 要人工申請，已降為低優先；第二家族改用 OLMo-3-7B，第三家族可用 Gemma-4。
- 論文多用 GPT-4o／4.1 當評審，那需要付費 API；改用本地評審就要另做校準。
- 必備模型下載約 55–85GB，會佔掉剩餘磁碟的大半。
- 目前的 51 格試驗遇過多次套件與搬遷問題，見 SUBMITTED.log，有工程風險。

---

## 4. 文生圖罕見失敗的機率估計

### 完整論文要做什麼

- **核心主張**：用 importance sampling 或 SMC 這類估計器，可以用遠少於直接 Monte Carlo 的取樣量，準確估出 unlearned 模型的罕見不安全生成率。
- **參考答案**：用直接 MC 建出 1e-3 到 1e-5 的參考答案。
- **審稿人期待的範圍**：
  - 多個抹除方法、至少兩個概念（裸露加 church 或 vangogh）；
  - 最好多一個骨幹（SD1.5 或 SDXL）；
  - 偵測器加上人工盲評，估陽性的可信度。
- **與 2611.10106 的區隔**：它只能做排序。我們必須拿出**絕對機率**加直接 MC 參考答案的證據。

### 類似論文

| 論文 | 會議／年份 | 資料、模型、規模 |
|---|---|---|
| UnlearnDiffAtk [2310.11868](https://arxiv.org/abs/2310.11868) | ECCV 2024 | SD1.4 的 ESD、FMN、SLD；I2P 的裸露與暴力；NudeNet |
| Ring-A-Bell [2310.10012](https://arxiv.org/abs/2310.10012) | ICLR 2024 | SD1.4／2.x、ESD、SLD、CA、FMN；I2P；NudeNet、Q16 |
| P4D [2309.06135](https://arxiv.org/abs/2309.06135) | ICML 2024 | ESD、SLD；從 I2P 篩出的 prompt；NudeNet、Q16 |
| MMA-Diffusion [2311.17516](https://arxiv.org/abs/2311.17516) | CVPR 2024 | SD1.5 加安全過濾器；benchmark 在 HF 上 gated |
| AdvUnlearn [2405.15234](https://arxiv.org/abs/2405.15234) | NeurIPS 2024 | SD1.4 text encoder 對抗訓練；I2P、ASR、COCO FID／CLIP |
| UnlearnCanvas [2402.11846](https://arxiv.org/abs/2402.11846) | NeurIPS 2024 D&B | 60 種風格 × 20 種物件，9 種方法 |
| Least-Action Auditing [2611.10106](https://arxiv.org/abs/2611.10106) | arXiv 2026-11 | SD1.4 ESD／UCE／RECE／SLD；每個模型 1,500 張樣本當參考答案；只能排序，絕對機率不準。**最接近的撞題** |
| DireSMC [2610.08652](https://arxiv.org/abs/2610.08652) | arXiv 2026-10 | SMC 罕見事件，1e-3 到 1e-5 的罕見度下加速 9 到 1413 倍 |

另可參考：
- Tail Risks in LM [2604.22167](https://arxiv.org/abs/2604.22167)（ICML 2026）：用少 10 到 20 倍的樣本，對上暴力 MC 的估計值。
- Weeding Out Bad Seeds [2609.37537](https://arxiv.org/abs/2609.37537)。
- IVO [2602.00175](https://arxiv.org/abs/2602.00175)。

### 資料集與模型

| | 項目 | 來源 | 大小 | 授權／gated | 本機 |
|---|---|---|---|---|---|
| 必備 prompt | I2P（4,703 個） | `AIML-TUDA/i2p` | 13MB | MIT | 無 |
| 必備 prompt | UnlearnDiffAtk prompts | GitHub `OPTML-Group/Diffusion-MU-Attack` | 極小 | 未確認 | 無 |
| 低優先（gated） | Ring-A-Bell 裸露 | `Chia15/RingABell-Nudity` | 69KB | **gated（auto）** | 無 |
| 必備生成器 | SD v1.4 | `CompVis/stable-diffusion-v1-4` | 本機 4.0G | OpenRAIL-M | **有** |
| 必備 eraser | UCE、ESD、SalUn、RECE（nudity） | `siyich/ijd-unlearned-checkpoints` | 本機 13G | CC-BY-NC-4.0，第三方整理 | **有** |
| 必備 eraser | FMN、SPM、EraseDiff、Scissorhands、STEREO | 同上 | 共約 16G | 同上 | 要下載 |
| 必備 eraser | AdvUnlearn | `OPTML-Group/AdvUnlearn` | nudity 470M | CC-BY-4.0 | **有**（只有 nudity） |
| 必備第二概念 | church 或 vangogh 版本 | 上面兩個 repo | 每個約 3.4G | 同上 | 要下載 |
| 必備偵測器 | NudeNet、Falconsai、CLIP L/14 | pip；`Falconsai/nsfw_image_detection` | 小 | Apache／MIT | **有** |
| 必備品質 | COCO val2017 與 captions | 官方 | 1.1G | CC-BY | **有** |
| 加分偵測器 | Q16 | GitHub `ml-research/Q16` | 小 | 未確認 | 無 |
| 加分偵測器 | LlavaGuard v1.2 7B | `AIML-TUDA/LlavaGuard-v1.2-7B-OV-hf` | 16G | 未標示 | 無 |
| 加分骨幹 | SD v1.5 | `stable-diffusion-v1-5/stable-diffusion-v1-5` | fp16 子集約 2–4G（未確認） | OpenRAIL-M | 無 |
| 加分骨幹 | SDXL base | `stabilityai/stable-diffusion-xl-base-1.0` | fp16 子集約 7G（未確認） | OpenRAIL++ | 無 |
| 加分 prompt | MMA-Diffusion | `YijunYang280/MMA-Diffusion-NSFW-adv-prompts-benchmark` | 0.5MB | **gated（auto）**，CC-BY-NC-ND | 無 |
| 避免 | UnlearnCanvas 影像 | `OPTML-Group/UnlearnCanvas` | **282G** | MIT | — |
| 避免 | FLUX.1、ShieldGemma 2 | — | 58–70G／22G | gated | — |

### 算力與風險

**算力**

| 部分 | 內容 | H200 GPU 時 |
|---|---|---|
| 參考答案 | 主體不需訓練；每格 10⁶ 張約 55 小時（實測約 1.8 萬張／GPU 時），做 5 格 | 約 275 |
| 估計器 | 在約 20 格上跑 | 含在合計內 |
| 合計 | | 300–400 |
| 學 proposal（加分） | | 每個模型 2–10 |
| SDXL（加分） | 每張圖慢 4–6 倍（未確認） | — |

**風險**
- **撞題**：2611.10106。
- **NSFW 資料存取**：
  - Ring-A-Bell 和 MMA 都需要登入同意；MMA 是 NC-ND。
  - 生成的圖只存縮圖，論文不能附圖。
  - 人工盲評需要標註者同意與倫理說明。
- **審稿人可能要求官方權重**：siyich 的權重是第三方整理、NC 授權，建議至少一個方法用原作者的權重。
- **磁碟**：只做 SD1.x 家族放得下；加 SDXL 就很緊。

---

## 5. 強驗證器與少量金標

### 完整論文要做什麼

- **核心主張**：強驗證器有系統性誤收時，在等風險下，用少量金標分配策略就能省掉人工成本，而且優於能使用相同資訊的抽查基線（PPI、PPI++、AM-PPI、Selective Auditing 的改編版）。
- **審稿人期待的範圍**：
  - **多個驗證器家族**：judge、ORM、PRM。
  - **標準基準**：ProcessBench、RewardBench 2。
  - **至少一個真人金標資料集**：MT-Bench human 或 HelpSteer3。只用數學終局答案的話，金標其實是免費的，「金標成本」的設定會被質疑。
- **理論定位**：要說明本題為何不受「judge 不比被評模型準時，金標最多省一半」（2410.13341）這個上限限制。

### 類似論文

| 論文 | 會議／年份 | 資料、模型 |
|---|---|---|
| Prediction-Powered Inference [2301.09633](https://arxiv.org/abs/2301.09633) | Science 2023 | 基礎文獻 |
| PPI++ [2311.01453](https://arxiv.org/abs/2311.01453) | 2023／24 | ppi_py 資料集 |
| Confidence-Driven Inference [2408.15204](https://arxiv.org/abs/2408.15204) | NAACL 2025 | GPT-4 標註加信心分數 |
| LLM as Judge won't beat twice the data [2410.13341](https://arxiv.org/abs/2410.13341) | ICLR 2025 | 理論上限 |
| Limits of Inference Scaling Through Resampling [2411.17501](https://arxiv.org/abs/2411.17501) | 2024／25 | HumanEval、MBPP，以不完美的 unit test 當驗證器 |
| ProcessBench [2412.06559](https://arxiv.org/abs/2412.06559) | ACL 2025 | 3,400 筆人工標註的錯誤步驟 |
| RewardBench 2 [2506.01937](https://arxiv.org/abs/2506.01937) | 2025 | 多技能 RM 基準 |
| Noisy but Valid [2601.20913](https://arxiv.org/abs/2601.20913) | ICLR 2026 | Jigsaw、SafeRLHF；用少量人工標註估 judge 的 TPR／FPR |

最直接的對手：
- Selective Auditing [2610.31662](https://arxiv.org/abs/2610.31662)
- AM-PPI [2605.08429](https://arxiv.org/abs/2605.08429)
- Cost-Optimal Active Eval [2506.07949](https://arxiv.org/abs/2506.07949)
- When Does Verification Pay Off [2512.02304](https://arxiv.org/abs/2512.02304)：37 個模型 × 9 個 benchmark。

### 資料集與模型

| | 項目 | 來源 | 大小 | 授權 | 本機 |
|---|---|---|---|---|---|
| 必備資料 | GSM8K、GSM1k、ASDiv-A、MATH-500、AIME、AMC、OlympiadBench、Minerva | 官方、`math-ai/*` | 都很小 | 各異 | **有** |
| 必備資料 | ProcessBench | `Qwen/ProcessBench` | 4MB | Apache-2.0 | 無 |
| 必備資料 | RewardBench 2 | `allenai/reward-bench-2` | 21MB | ODC-BY | 無 |
| 必備資料 | MT-Bench human judgments | `lmsys/mt_bench_human_judgments` | repo 總量 640MB | CC-BY-4.0 | 無 |
| 加分資料 | JudgeBench、Arena 55k、HelpSteer3、RewardBench v1 | `ScalerLab/JudgeBench`、`lmarena-ai/arena-human-preference-55k`、`nvidia/HelpSteer3`、`allenai/reward-bench` | 1.2MB／387MB／1.4G／124MB | MIT／Apache／CC-BY／ODC-BY | 無 |
| 必備生成器 | Qwen2.5-7B-Instruct、Oat-Zero-1.5B | HF | 15G／3.1G | Apache | **有** |
| 必備 judge | Qwen3-32B-AWQ（強）、Qwen3-0.6B（弱） | HF | 19G／1.5G | Apache | **有** |
| 必備 PRM | Qwen2.5-Math-PRM-7B | HF | 15G | Qwen 自訂授權 | 無 |
| 必備 ORM | Skywork-Reward-V2-Qwen3-8B | HF | 15G | Apache | 無 |
| 必備第二家族 judge | gpt-oss-20b | `openai/gpt-oss-20b` | 權重約 13G（未確認） | Apache | 無 |
| 加分 judge | Prometheus-2 7B、Qwen3-30B-A3B-2507 | HF | 14.5G／61G | Apache | 無 |
| 避免 | 70B 級（Llama-3.3、PRM-72B、Nemotron-70B） | — | 141–282G | Llama 為 gated | — |

### 算力與風險

**算力**

| 部分 | 內容 | H200 GPU 時 |
|---|---|---|
| 主線 | 不需訓練；金標分配器是 CPU 上的 LP 或小型回歸 | 0 |
| 推論 | 5 個資料集 × 約 2 萬候選 × 4 個驗證器 | 30–60 |
| 小型校準頭（加分） | | 每次 2–8 |

**風險**
- **撞題**：AM-PPI 和 Selective Auditing，必須在等風險下打敗它們的改編版。
- **磁碟**：必備項目約 45–50G。若捨 gpt-oss，改以 PRM 加 RM 當第二、第三家族，可以減到約 30G。

---

## 6. 只移植少數 RL 推理決策的蒸餾

### 完整論文要做什麼

- **核心主張**：RL 後模型和 base 模型只在少數 token 決策上不同。只把這些決策移植或蒸餾到 base 模型，就能在相同算力下改善多次作答（pass@k），而不損失多樣性。
- **審稿人期待的範圍**：
  - **配對**：2–3 個公開 base／RL 配對，跨家族並擴到 7B。
  - **評測**：標準數學集，再加 2026 新題組防污染。
  - **協定**：長上限（至少 8–32K）下的 pass@k，加配對 bootstrap。
- **對手**：WiSE-FT、整段回應層級混合、ReasonMaxxer 類方法、rl-replace。

### 類似論文

| 論文 | 會議／年份 | 資料 | 模型 | 規模 |
|---|---|---|---|---|
| Does RL Really Incentivize… [2504.13837](https://arxiv.org/abs/2504.13837) | NeurIPS 2025（未確認） | GSM8K、MATH500、Minerva、OlympiadBench、AIME24、AMC23 等 | SimpleRL-Zoo、Oat-Zero、DAPO 配對 | pass@k 到 256 以上 |
| Understanding R1-Zero（Oat-Zero）[2503.20783](https://arxiv.org/abs/2503.20783) | COLM 2025（未確認） | MATH level 3–5 約 8.5k 題 | Qwen2.5-Math-1.5B／7B、Llama-3.2-3B | 7B 約 8×A100 × 27 小時（未核對） |
| SimpleRL-Zoo [2503.18892](https://arxiv.org/abs/2503.18892) | COLM 2025 | MATH 約 8k 題 | 10 個 base，0.5–32B | 未記錄 |
| Beyond the 80/20 Rule [2506.01939](https://arxiv.org/abs/2506.01939) | NeurIPS 2025 | DAPO-Math-17k | Qwen3-8B／14B／32B base | 未記錄 |
| Critical Tokens Matter [2411.19943](https://arxiv.org/abs/2411.19943) | arXiv | GSM8K、MATH500 | Llama-3-8B／70B、DeepSeek-Math-7B | 未記錄 |
| Sparse but Critical [2603.22446](https://arxiv.org/abs/2603.22446) | ICLR 2026 | 數學題組 | 多組 base／RL 配對 | cross-sampling 換 token。**撞題** |
| Reasoning Restored by a Few Decision Tokens [2605.16874](https://arxiv.org/abs/2605.16874) | ICML 2026 | 數學推理 | Qwen3-0.6B 等 | 約 8% token 有顯著分歧。**撞題** |
| What RL Brings to Reasoning [2610.24663](https://arxiv.org/abs/2610.24663) | arXiv 2026-10 | DAPO-17k 訓練；AIME、MATH-500 | Qwen3-4B／8B-Base、Llama-3.2-3B 自訓 GRPO（權重未公開） | 大 k 實驗約 1,012 H800 小時 |

另可引 Base Models Know How to Reason [2510.07364](https://arxiv.org/abs/2510.07364)（ICML 2026 spotlight），用 9 組 base／thinking 配對。

### 資料集與模型

| | 必備 | 加分 |
|---|---|---|
| 評測 | MATH-500、AIME24／25、AMC23、Minerva、OlympiadBench（**全有**）；DAPO-Math-17k（**有**） | GPQA（**gated auto**）；LiveCodeBench（4.5GB） |
| 大題組 | `EleutherAI/hendrycks_math`（5MB）；`agentica-org/DeepScaleR-Preview-Dataset`（21MB） | `hkust-nlp/SimpleRL-Zoo-Data`（22MB） |
| 防污染 | `MathArena/aime_2026`（CC-BY-NC-SA） | `MathArena/hmmt_feb_2025` |
| 配對 1 | Qwen2.5-Math-1.5B 對 Oat-Zero-1.5B（**有**） | — |
| 配對 2 | Qwen2.5-1.5B 對 SimpleRL-Zoo-1.5B（**有**，fp32 7.1GB） | ORZ-1.5B（3.1GB） |
| 配對 3 | R1-Distill-Qwen-1.5B（3.6GB）對 DeepScaleR-1.5B（fp32 7.1GB）。據深查，這是唯一被證實 pass@k 會交叉的公開配對，但要 32K 上限 | ProRL Nemotron-1.5B（CC-BY-NC） |
| 7B | Qwen2.5-Math-7B 對 Oat-Zero-7B（各 15.2GB） | SimpleRL-Zoo 7B／Llama-8B（fp32 約 30GB） |
| 自訓 | Qwen3-1.7B 或 4B-Base，用 DAPO-17k 跑 GRPO | — |

### 算力與風險

**算力**

| 部分 | H200 GPU 時 |
|---|---|
| 三個 1.5B 配對（評估、rl-replace 重現、蒸餾） | 250–400 |
| 7B 配對 | +200–300 |
| 自訓 GRPO | 60–100 |
| 合計 | **500–800**（六題中最貴） |

**風險**
- **撞題**：兩篇 2026 主會論文已經很接近。
- **截斷**：SimpleRL-Zoo 的 pass@k 交叉多半不顯著，而 3K 上限會製造假交叉，這也是目前題目的截斷問題。
- **成本**：DeepScaleR 配對要 32K 上限，評估很貴。
- **配對 B 現況**：正式重現閘門未通過（26500、26501），短預算判為未能測試。

---

## 分配建議（供使用者決定）

### 粗略排序

依三個面向排序：論文潛力乘證據，磁碟，GPU。

| 排序 | 題目 | 理由 |
|---|---|---|
| 1 | **多條件檢索** | 證據與潛力都是 4/5，最便宜（主線 GPU 不到 20 小時，磁碟約 25G），可以在 3090 上開發。風險在方法本身還沒找到有效的路 |
| 2 | **多保真度歸因** | 潛力 4/5，磁碟幾乎不用，GPU 60–90 小時。風險在跨初始化的增益還不顯著 |
| 3 | **文生圖罕見失敗** | 潛力 4/5，但證據只有 2/5，GPU 300–400 小時，又有直接撞題。正式事件率出來後再決定是否加碼 |
| 4 | **Λ 漏出預測** | 潛力 4/5，但證據只有 2/5，而且磁碟最吃（55–85G）。目前的 51 格結果出來前，不建議下載擴展模型 |
| 5 | **強驗證器** | 潛力 3/5，GPU 便宜（30–60 小時），但撞題風險高，金標成本的設定也要重新設計 |
| 6 | **RL 稀疏蒸餾** | 潛力 3/5、證據 2/5、最貴（500–800 小時），還有兩篇 2026 主會撞題，建議最後考慮 |

### 一個可行的組合

| 項目 | 內容 | 磁碟 | H200 GPU 時 |
|---|---|---|---|
| 主攻 | 檢索加歸因，兩題都做到完整論文的必備範圍 | 約 30G | 約 150–250 |
| 並行 | 文生圖、Λ 繼續完成目前的 kill test，等現象與訊號結果出來，再決定誰升級成第二或第三題 | — | — |
| 暫停擴展 | 驗證器、RL 只維持目前規模，不下載擴展權重 | — | — |
| 若升級文生圖 | | +20–30G | +300–400 |
| 若升級 Λ | 必須先清空間，或只挑一個 7–8B 家族（OLMo-3-7B）加一個 27B FP8 | — | — |

### 下載前要先做的事

1. 列出 `icml/` 底下佔用最大的目錄。清掉已結束 kill test 的 run 和快取時，要先確認 Codex 沒有在用。
2. 三個空殼快取已刪除（2026-11-16）；gpt2-large 只有 tokenizer，用到權重時要另外下載。
3. 下載時只抓 safetensors 或 fp16 子集；fp32 的 repo（SimpleRL-Zoo、DeepScaleR）要特別注意大小。
4. gated 資源（Llama、Winoground、Ring-A-Bell、MMA、GPQA）一律低優先，必備清單已改用不需申請的替代品。真的需要時由使用者決定是否申請；代理不會登入。
