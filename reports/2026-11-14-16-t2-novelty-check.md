## T2 文獻新穎性快篩（2026-11-14，只讀不改檔）

**結論先講**：收斂後的主張「同來源／同領域請求聚成 batch，expert 聯集會變小」已有人發表，而且是在同一個模型（Qwen3-30B-A3B）上量的。連 offload 情境、用 prefill 路由分群取代領域標籤，也都已有人做過，並量了端到端收益。剩下能主張的新穎性很窄，以目前的形式投 ICML，適配度偏低。

### A. 本題複述
MoE 每一步 decode 要放上 GPU 的，是 batch 裡所有請求所選 expert 的聯集 U。expert 放在 CPU、GPU 只留快取時，U 決定要搬多少。本題主張只用整體 expert 熱度估 U（假設各請求獨立）會估不準：batch 若由同一領域的請求組成，用領域熱度預測 U/E 比獨立熱度式準 5–14 pp（B = 8–32，OLMoE 與 Qwen3-30B-A3B），打散 trace 後這個增益就消失，代表增益來自請求間的相關。W1 也發現隨機混領域的 batch 增益 ≤ 1 pp，所以主張收斂成「同來源請求聚成 batch」。W2 打算用 prefill 路由直方圖做 k-means 推斷領域（不靠標籤），再在 MoE-Infinity 上量 bytes 與 TPOT。目前還沒有任何時間或 bytes 的實測。

### B. 最接近的工作（依重疊程度排序）

| # | 工作 | 方法一句話 | 它報告的關鍵數字 | 與本題重疊 | 本題仍不同之處 |
|---|---|---|---|---|---|
| 1 | **"Sort your prompts: a free speed-up for MoE batch inference"**，A. Badaoui（Hugging Face 部落格，2026-10-19），https://huggingface.co/blog/badaoui/sort-your-prompts | 按領域把 prompt 排序後再分 batch；比較三種分法：領域標籤、prefill 路由指紋做 k-means、文字 embedding 做 k-means；也測了串流用的分窗排序 | Qwen3-30B-A3B 單一領域 batch 每步少讀 19–25% expert（B = 8–128）；gpt-oss-20b 少 8–11%；UltraChat 用 embedding 分群少 7–12%。stock vLLM、MI300X 實測吞吐：Qwen3 +11.6～17.3%，gpt-oss-120b +6.9～12.2%，2 卡 EP +8.7～12.9%；路由指紋 k-means 還原 6 個領域，純度 100% | 幾乎完全重疊 W1 的收斂主張，也重疊 W2a 的 A1（路由分群）。資料（MBPP、GSM8K）和模型（Qwen3-30B-A3B）都相同 | 非同行審查的部落格；expert 全部常駐 GPU，沒有 offload；沒有預測 U/E 的模型；沒有打散對照 |
| 2 | **Saber: Sparsity-Aware Scheduling for Efficient MoE-Based LLM Serving on Edge Servers**（MobiCom '26），https://cis.temple.edu/~wu/research/publications/Publication_files/mobicom26-paper974.pdf | 情境就是 offload：expert 在 CPU 與 GPU 間換入換出。把「activation path 相似」的請求聚成 group 再組 batch；activation pattern 離線分群，線上用輕量分類器分組，另有效用驅動的優先度排程 | Qwen1.5-MoE 每 batch 啟用 expert 比例從 60.7% 降到 37.7%，每 batch 少搬 6.1 GB；I/O 最多減少 1.95×，batch 執行時間最多快 3.8×，同延遲下吞吐最多 3.27×。硬體 RTX 3090；模型 Switch Transformers、LLaMA-MoE、Qwen1.5-MoE | 與本題「offload 下把同類請求聚成 batch」的系統版本實質相同，而且已量端到端 | 模型較舊；分群依據是 activation path，不是領域；沒有聯集預測模型，也沒有混領域、相近領域的對照。作者從 PDF 抽出的文字中未取得（未確認） |
| 3 | **ELDR: Expert-Locality-Aware Decode Routing for PD-Disaggregated MoE Serving**，Choi, Cho, Xiong, Yang, Kwon, Cheng（SIGOPS ATC 2026），https://arxiv.org/abs/2607.00466 | 見下方詳述 | Task 工作負載同領域 batch 少用 17–21% expert；TPOT 降 5.4–14.5% | 同領域 batch 的 U 較小（實測）；用 prefill 簽章預測 decode 的 expert | 不做 offload；做的是跨 decoder 的路由，不是 batch 內的聯集預測 |
| 4 | **ExpertFlow**，He et al.（arXiv 2410.17954 v2；註記為 DAC'26），https://arxiv.org/abs/2410.17954 | 單張 GPU 加 CPU offload，以預測器預測 routing，token scheduler 把「預測路徑相似」的 token 聚在一起，搭配預測式 expert 快取 | 摘要報告 GPU 記憶體最多省 93.72%，吞吐最多 10×（只讀了摘要） | offload 下按路由相似度重組 batch | 粒度是 token，不談領域，也沒有聯集預測模型 |
| 5 | **Scaling Multi-Node MoE Inference Using Expert Activation Patterns**，Bambhaniya et al.（arXiv 2604.23150，2026-04），https://arxiv.org/abs/2604.23150 | 用 prefill（或 decode）的 activation 向量做 K-Means，再依群組 micro-batch，並配合 expert 擺放 | prefill 與 decode 的逐層相關係數：Maverick 0.82、Qwen3-235B 0.68、DeepSeek-V3 0.55；請求類型分類準確率 94–99%；all-to-all 最多減少 20%，該層 runtime 降 5.5–6% | 已有「同領域請求的 expert 高度相關」與「prefill 可預測 decode」的實證，並據此分群 | 情境是多節點 EP 的通訊；不做 offload，不談聯集 |
| 6 | **Sem-MoE / Semantic Parallelism**，Y. Li et al.（ICLR 2026），https://arxiv.org/abs/2503.04398 | 依「輸入類別會啟用哪些 expert」把請求重新分 batch 到持有那些 expert 的裝置，並依共同啟用把 expert 分群放置 | 摘要沒有具體數字 | 依 expert 親和性路由請求，而且發表在 ML 會議 | 情境是 EP 的 all-to-all，不是 offload |
| 7 | **DoMoE**，Mu et al.（IJCAI 2026），https://www.ijcai.org/proceedings/2026/657 | offload 下建立領域專屬的 routing table 來預測並預取 expert | 吞吐是 FMoE 的 1.32×、命中率是其 1.22×；在 Mixtral 上約為 MoE-Infinity 的 1.9–2.1×；硬體 L40 | 以領域資訊改善 offload 下的 expert 預測與快取 | 逐請求預測，不把請求聚成 batch |
| 8 | **Doubleword 部落格 "MoE expert co-activations"**，J. Cowan（2026-05-15），https://blog.doubleword.ai/moe-expert-coactivations | 重排 prompt 順序，使同一 batch 的 expert 重疊；比較 oracle、BGE embedding 分群、微調過的 embedder、類別 | Qwen3.5-35B-A3B 的 unique expert loads：oracle 少 21.3%，BGE 少 12.4%，微調後少 15.6%；WildChat 少 10.8–12.3%。牆鐘時間省約 5.4% 是推估值，不是實測 | 與 #1 相同 | 非同行審查；沒有 offload |

**另有幾篇是正交方向（改 routing，不改 batch 組成），均已開啟摘要確認**：
- OEA（arXiv 2511.02237，Oncescu、Dao 等）：B = 16，MoE 層延遲 Qwen3-30B 降 39%、Qwen3-235B 降 15%。
- XShare（arXiv 2602.07265）：expert 啟用最多減少 30%；作者說混合資料集的 batch 也有效。
- Lynx（arXiv 2411.08982）：吞吐 1.30×。
- Bevy（arXiv 2611.04794）：offload 情境，按執行狀態（不是內容）把請求分 batch；吞吐最多是 llama.cpp 的 3.46×；指出有 GPU 快取時預取可能反而有害。

#### ELDR 詳述（讀了 arXiv HTML v1 與 v3）
- **設定**：情境是 prefill／decode 分離（PD-disaggregated）的 serving，expert 全部常駐 GPU HBM，完全沒有 offload。它的論點是 decode 受頻寬限制，batch 啟用越多不同 expert，就要從 HBM 讀越多權重；在單張 MI300X 上，Qwen3-30B-A3B 的啟用 expert 從 16 增到 128，MoE 層延遲約變為 4.7×。
- **方法**：用 prefill 簽章預測 decode 的 expert 使用；balanced K-means 擬合每個 decoder 的中心並在背景重新擬合；locality-band 路由在相近的候選中挑負載最輕的 worker；以 block 粒度快取讓簽章在 prefix caching 下仍精確。輸出與 token→expert 的指派都不變（lossless）。
- **資料與模型**：Qwen3-30B-A3B、GPT-OSS-120B、Gemma-4-26B-A4B（另有 235B 的 EP 實驗）；工作負載是 12 個公開 benchmark（code、math、醫學、法律），加上 14k 筆多語 WildChat。
- **特性刻畫數字**：在相同 batch size 32–128 下，同領域 batch 每步少用 17–21%（Task）、3–10%（Language）的 expert。prefill 與 decode 的逐請求相關平均 0.69（隨機配對 0.18）；按領域合併後的相關：Qwen3 0.90、GPT-OSS 0.70、Gemma 0.92。
- **端到端**：最多 40 張 MI300X、實作在 vLLM。中位 TPOT 平均降 5.4–14.5%，峰值 17.8%；P99 降 3.4–5.6%；TTFT 大致不變。基線有 Random、RR、JSQ、P2C，以及使用 oracle 領域標籤的 "Domain" 基線。
- **與 repo 記載不符**：repo 寫「ELDR 報告同領域 batch 少用 19–25%」。我在 ELDR 的 v1 與 v3 都沒有找到 19–25%；這個數字其實出自 HF 部落格（Qwen3、B = 8–128）。ELDR 本身是 17–21%。v2 沒有讀。

#### repo 引用文獻的實際內容
- **SlimWise（2609.34117）**：HTML §2.1 只有均勻路由的式子 m[1 − (1 − k/m)^B]，沒有獨立熱度式 Σₑ[1 − (1 − pₑ)^B]。repo 把獨立熱度基線標成「SlimWise 式」，出處不對。它也指出 skew 只會延後飽和；Qwen3.6-35B-A3B 在 B = 64 時每步約讀到 75% 的 expert。
- **MoESD（2505.19645，NeurIPS'25 spotlight）**：只讀了摘要，主題是 MoE 上的 speculative decoding。聯集閉式是否在正文中**未確認**。
- **2608.07911（單作者，Yu Zhang）**：trace 模擬的 expert 快取評測方法論，涵蓋 replay 語意、prompt 模板污染、建議回報「聯集／容量比」。沒有報告聯集隨 B 的變化，也不研究 batch 組成。repo 所引的「Qwen3 聯集 8.0/15.0/26.6/42.6/59.2」在摘要中**未確認**。
- **AdaOffload（2611.10219，Stoica 等）**：多 GPU 的 CPU/GPU 混合執行，吞吐 1.9–15.5×；摘要沒提到領域或請求分組。
- **2505.16056（SRP／SCH）**：讀的 arXiv 頁面沒寫發表場地，repo 標的「ICLR'26」**未確認**。它發現「領域專精的 expert 對 routing 一致性貢獻較大」。
- **MoE-Infinity**：只看到 ELDR 引用的 arXiv 2401.14361，原文沒開。

**只看到搜尋摘要、沒有開原文（未確認）**：fMoE（2502.05370）、eMoE（2503.06823，task-aware 的請求排程與 expert 預測）、Read-ME（2410.19123）、Diff-MoE（SC'25，網頁 403）、IEEE "Two-Stage Expert Offloading for Domain-Aware MoE Inference"（網頁 418）、DALI（2602.03495）、OSDI'26 的本地 MoE 論文（2606.10493）、SERE（2602.07616）、HOBBIT、ProMoE、AdapMoE、SiDA-MoE。另有 vLLM RFC #38256，我只從搜尋摘要看到：它說在 OLMoE 上 6 個 prompt 的 batch 一次 forward 要用到 57/64 個 expert，使用量分布有很強的領域依賴。

### C. 有沒有人已做了實質相同的事
**有，分兩塊：**
1. **「同領域 batch → 聯集較小」並量到端到端收益**：HF 部落格（同一模型、同樣的領域資料、也有路由分群與 embedding 分群）、ELDR（ATC'26）、Doubleword 都做了，只是都在 expert 常駐 GPU 的情境。
2. **offload 情境下「把 expert 相似的請求聚成 batch」**：Saber（MobiCom'26）已做，並量了 I/O、執行時間與吞吐；ExpertFlow 在 token 粒度做過。

**剩下的新穎性（很窄）**：
- 一個預測 U/E 的模型：用 batch 的領域混合熱度預測聯集，並和獨立熱度式比較，再用打散 trace 證明增益來自請求間相關。這屬於分析或量測方法的貢獻。
- 負面與邊界結果：隨機混領域 ≤ 1 pp；相近領域會不會也消失（W2a 的 A3，尚未做）。
- 用新模型（OLMoE、Qwen3）系統性比較 offload 快取重播策略（W2a 的 A4，尚未做）。

這幾塊單獨看都不足以成為 ICML 的主貢獻。W2a 的 A1（無標籤路由分群）已被 HF 部落格、ELDR、Bambhaniya、Saber 涵蓋，不再是新的。

### D. 衝突或已隱含，以及審稿人可能的質疑
- **沒有衝突，結論已被隱含或直接報告**。ELDR、HF、Saber 的數字方向與 W1 一致，量級也相近：HF 19–25%、ELDR 17–21%、Saber 啟用比例 60.7% → 37.7%。2505.16056 與 Bambhaniya 已報告 expert 有領域分工、prefill 可預測 decode。
- **與先前研究一致的地方**：混領域無增益和 ELDR 的 Language 工作負載只有 3–10% 方向一致；相近領域增益可能很小，HF 的 UltraChat 只有 7–12%、吞吐約 +5%。
- **最可能的「這不就是 X」**：「這不就是 Sort your prompts、ELDR 或 Saber？」，以及「expert 有領域分工（OLMoE 論文、2505.16056），同領域 batch 聯集較小是顯然的」。
- **針對預測模型的質疑**：「把熱度換成條件分布當然更準」。本題的獨立熱度基線也站不住：SlimWise 根本沒有那個式子，基線出處要改，否則會被說是稻草人基線。
- **其他**：沒有時間與 bytes 數據；把請求聚成 batch 會增加排隊延遲（HF 部落格自己就提了）；W1 每個領域只有 32 個留出請求，B = 64 那格還有重複請求。

### E. 審稿人預期
- **模型**：至少 3 個現代 MoE，包含大型者。對手論文用的是 Qwen3-30B-A3B、GPT-OSS-120B、Gemma-4-26B-A4B、Qwen3-235B、DeepSeek-V3、Llama-4 Maverick、Qwen3.5-35B-A3B。只有 OLMoE 加 Qwen3-30B 會被認為不夠。
- **資料**：多領域 benchmark 之外，要有真實混合流量（WildChat、LMSYS、UltraChat），最好有到達時間的 trace（例如 Azure trace），並在沒有領域標籤的條件下評估。
- **系統**：offload 情境要在真實 serving stack（vLLM／SGLang 的 offload 路徑，或 MoE-Infinity）上實作。基線至少要有 MoE-Infinity、fMoE、ExpertFlow、Saber、DoMoE、按領域排序（HF 部落格 / ELDR 的 Domain 基線），並說明與 OEA、XShare 這類改 routing 方法的關係。
- **指標：要端到端，只報 expert 計數不夠。** 所有同類論文都報了 TPOT／TTFT／吞吐（ELDR 還報 P99），Saber 報了 I/O bytes 與吞吐。必須量：TPOT（中位與 P99）、TTFT、為了聚 batch 增加的排隊延遲、相同 SLO 下的吞吐、PCIe bytes。只報 U/E 預測誤差，會被視為不完整。

### F. 發表場地與 ICML 適配度
- **實際場地**：ELDR 在 ATC'26，Saber 在 MobiCom'26，ExpertFlow 在 DAC'26，DoMoE 在 IJCAI'26，Sem-MoE 在 ICLR'26，MoESD 在 NeurIPS'25（spotlight），Diff-MoE 在 SC'25（未確認）。其餘多為 arXiv 或部落格。主流是系統場地（OSDI、ATC、EuroSys、MLSys、SC、MobiCom）。
- **ICML 適配度：低。** ML 會議接受這類工作的前提，是有可推廣的方法或理論（Sem-MoE 是聯合排程的 formulation，MoESD 是理論模型），不是量測。本題若要投 ICML，需要一個超出「條件熱度 → 聯集」、能被證明或可學習的貢獻，例如有保證的 batch 組成或快取最佳化，或是從聯集模型推導出最佳分組與排隊延遲之間的取捨。同時仍要有端到端數字。以目前的形式，較像 MLSys 或 workshop 論文。

### 給 repo 的更正
1. 「ELDR 19–25%」應改為：ELDR 是 17–21%（Task）／3–10%（Language）；19–25% 出自 HF 部落格。
2. 獨立熱度式不是 SlimWise 的；SlimWise 只有均勻路由的式子。
3. 撞題清單要加入 Saber（MobiCom'26）、Bambhaniya（2604.23150）、Sem-MoE（ICLR'26）、DoMoE（IJCAI'26）、Doubleword 部落格。

暫存檔：ELDR PDF 在 /tmp/claude-1047/-work-ChihChungHsu-cactus8603-icml27-s2-gated-sink-kv/bd65f9a7-961a-4b8d-adce-42df28fbb74b/scratchpad/eldr/eldr.pdf，Saber 抽出的文字在 /tmp/claude-1047/-work-ChihChungHsu-cactus8603-icml27-s2-gated-sink-kv/bd65f9a7-961a-4b8d-adce-42df28fbb74b/scratchpad/saber.txt。
