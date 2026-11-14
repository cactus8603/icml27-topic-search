## K3 文獻新穎性快篩（2026-11-14，唯讀，沒改任何檔案）

**結論先講**：沒有找到已經做完同一件事的論文，但這題的核心容量模型已有 ICML 2026 的先例。剩下的新穎性很窄，主要剩兩件事：把那個模型用到 VLM 的視覺 token 上，以及補上 preemption 重算項。另外，KV 池是人為設小的，審稿人很可能直接說「這很顯然」或「設定不真實」。

### A. 本題複述
- 本題不提新的剪枝法，問的是「每張圖少掉一批視覺 token（降解析度或真剪枝）之後，KV 受限的 vLLM 服務容量（req/s）會變多少，能不能事先算出來」。
- 模型：
  - 每個請求的 KV 佔用用「KV × 時間面積」表示：occ = s + E[o²]/(2E[o])。
  - 並行數 N = min(max_seqs, u·P/occ)。
  - 容量 λ = N / (N·τ_pre + E[o]·τ_dec)。
  - 另加一項事先預測的 preemption 重算 prefill 量（κ·occ·pressure）。
- 主張有三點：
  - 視覺 token 減少帶來的容量收益可以很大：在 Qwen2.5-VL-7B、H200、固定 65,536 token 的小池上，s 從 4,096 降到 256，容量變 13.1×（輸出 32 token）與 4.9×（輸出 1,024 token）。
  - 收益隨輸出變長而縮小。
  - 這個模型能在新題目上把留出格誤差壓到 7.8%，優於只看 prefill 的模型（19.0%）。
- W2 打算再驗證開環 Poisson 到達、GlimpsePrune 真剪枝、品質代價與 InternVL3。

### B. 最接近的工作（依重疊程度排序，全部都開過頁面確認）

| # | 標題／作者／年份與發表處 | 方法一句話 | 它報的關鍵數字 | 與本題重疊 | 本題仍不同之處 |
|---|---|---|---|---|---|
| 1 | A Queueing-Theoretic Framework for Stability Analysis of LLM Inference with KV Cache Memory Constraints — Nie, Si, Zhou，**ICML 2026**，https://arxiv.org/abs/2605.04595 | 用排隊論推導 KV 記憶體受限時的穩定服務率 μ = M/(b̄·E[g(s,o)])，其中 g ≈ [(1+s/ŝ)s + 2os + (1+o)o]/2 就是每個請求一生累積的 KV 用量（即記憶體–時間面積）；結果拿來配置 GPU | vLLM v1、Llama-3-8B、A100；預測誤差 2.25–8.38%，LongBench v2 為 8.03%，整體「通常 <10%」 | **很高**：本題的 occ／面積模型和容量公式與它在結構上幾乎相同，同樣用 Little's law 式的面積論證，同樣用 vLLM 驗證，誤差也在同一量級 | 它只有文字、沒有視覺 token 或剪枝；假設 swap 不花成本，不建模 preemption 重算；沒有 max_num_seqs 上限 |
| 2 | Service-Induced Congestion in Memory-Constrained LLM Serving — Ao, Dong, Luo, Simchi-Levi，arXiv 2606.15555（2026，v3 於 10 月） | 離散時間動態模型，描述 KV 隨生成成長、被 evict 後重啟的過程 | 同質負載下 eviction 造成的吞吐損失可達 50%；不同 decode 長度互質時較穩定 | 中高：正好是本題 W1 的失敗來源（preemption 與重算） | 純理論加少量 GPU 實驗（我沒看到具體數字）；不談視覺 token |
| 3 | ERA: Entropy-Guided Visual Token Pruning with Rectified Attention — Wang 等，arXiv 2606.31982（2026） | 視覺 token 剪枝法，並附一節 vLLM 部署量測 | vLLM、LLaVA-NeXT-7B、batch 256：每請求 KV 0.901→0.182 GB（5.0×），峰值記憶體 4.1× 更低，攤銷端到端 3.3×，decode 吞吐幾乎不變（68.3→70.4 tok/s）；單請求時 FLOPs 減 11.8× 但延遲只快 2.7× | 中：已在 vLLM 量到「KV 變小、batch 下端到端收益變大」 | 只量固定 batch，沒有容量模型、輸出長度掃描或到達率 |
| 4 | HAP: Head-Adaptive Visual Token Pruning — Sun 等，arXiv 2608.23921（頁面標 EMNLP 2026） | 剪枝法，附錄 I 用 FLOPs 模型分析收益如何隨 decode 長度衰減（gain limit） | 剪 75% 時 prefill 省 33.6%；收益掉到 1% 的 decode 長度在 Qwen2.5-VL-7B 約 217K、LLaVA-1.5 約 65K；端到端加速 1.9×（BS=1）→ 2.4×（BS=8） | 中：已經把「收益隨輸出長度縮水」寫成閉式式子 | 只算 FLOPs，不算 KV 容量；沒有服務容量或排隊 |
| 5 | Are We Evaluating VLM Token Pruning Right? (PAA-Bench) — Li, Chen, Shan, Yang，arXiv 2610.24740（2026-10） | 提出 Preservation-Constrained Speedup：在保留一定品質的前提下比較最大端到端加速 | 8 種方法、3 個骨幹；**batch 1**、輸出上限 128；CDPruner 4.37×、SparseVLM 在 Qwen3-VL 只有 1.23×、VisionZip 0.95×（反而變慢） | 中：屬於「token 數不等於實際效率」的批判 | 只量單請求延遲；沒有 KV、batch、並行或輸出長度。repo 寫它「已測端到端加速」是對的，但它沒碰服務層 |
| 6 | When Do Fewer Visual Tokens Accelerate Multimodal Inference? A Break-Even Study — Dou, Tian，arXiv 2608.03649（2026） | 用 break-even 協議拆解決策開銷與 encoder 成本 | Qwen2.5-VL-3B、3090／A100、30 題；A100 上 token 少約 8 倍的策略反而省得較少 | 中低：同屬批判類，用的也是 Qwen2.5-VL | 只有單請求；沒有吞吐或 KV |
| 7 | ModServe — Qiu 等（Microsoft），arXiv 2502.00937v3（頁面未列發表處） | 多模態 serving 的特徵分析，並把不同模態與階段的資源拆開 | 圖像 token 每請求 1,280–7,290；encoder 佔 TTFT 25–79%；吞吐為 vLLM 的 3.3–5.5× | 低中：VLM serving 的容量規劃 | 容量靠 profiling 加外插，沒有閉式 KV 面積模型；不討論剪枝 |
| 8 | SparseVILA — Khaki 等（Song Han 組），arXiv 2510.17777 | 把 prefill 剪枝與 decode 檢索拆開 | prefill 4.0×、decode 2.5×、端到端 2.6×；KV 少 72.5% | 低：單請求延遲 | batch 1、用 TinyChat；不談容量 |

補充（都確認過）：
- **與服務容量相關的旁支**：
  - Ramani & Tantawi，arXiv 2609.20957：近似排隊模型，H100 上 ITL 誤差 5–8%、TTFT 誤差 14–16%，不建模 KV 容量。
  - Vidur（MLSys 2024，2405.05465）：模擬器，延遲誤差 <9%。
  - Ao 等，arXiv 2504.11320：fluid 模型加 KV 記憶體約束的排程。
- **VTC-Bench**（2510.07143，ACL 2026 Main）：只看精度，結論是降採樣勝過多數壓縮法，與 repo 描述相符。
- **GlimpsePrune**（2508.01548）：arXiv 預印本，剪 92.6%，與 repo 描述相符。
- **Fewer Eyes**：是 NeurIPS 2025 **workshop** 論文（Efficient Reasoning），不是主會，repo 的標註要改。方法是生成幾步後撤掉視覺 token，TFLOPs 最多減 55.9%。
- **VL-Cache**（2410.23317）：只看到摘要（KV 留 10%、decode 7.08×）；吞吐與 batch 的細節未確認。

### C. 有沒有人做過實質相同的事
- **沒有**找到「視覺 token 減少 → KV 受限下 vLLM 服務容量 → 閉式模型事先預測並驗證」的論文。剪枝論文最多做到固定 batch 的 vLLM 量測（ERA）或 FLOPs 隨 decode 長度衰減的分析（HAP）。容量／排隊模型（Nie 等）則只處理文字。
- **剩下的新穎性很窄**，大致是三塊：
  1. 把 Nie 等的面積／穩定服務率模型用在「視覺 token 數 s」這個干預軸上，在 VLM 上驗證，量化收益隨 o 縮小的曲線。換句話說，是「多模態加剪枝的實例化」，不是新的模型。
  2. 事先預測的 preemption 重算項。Nie 等假設 swap 不花成本，Ao 等只有理論，所以這一塊相對最新。但本題的 κ 是看過 W1r 結果才定的，模型形式也是事後挑的。
  3. 真剪枝（每題保留數不同）和品質約束下的容量，W2 還沒跑。

### D. 衝突、隱含與審稿人最可能的質疑
- **「這不就是 Nie 等（ICML 2026）把 s 換成視覺 token？」**
  - 本題的 occ = s + E[o²]/(2E[o]) 和容量式，在數學上就是 E[g(s,o)] 的面積論證。
  - 本題文件完全沒有引用這篇。這是最大的撞題風險，必須正面引用並說清楚差別：重算項、max_num_seqs、視覺 token 與 encoder 成本。
- **「這很顯然」**
  - 每個請求的 KV 正比於 s + o；s 從 4,096 降到 256 時，在 o 小的情況下 occ 縮約 13 倍，容量比 13.1× 幾乎是算術結果。
  - 收益隨 o 縮小的結論，HAP 的 gain limit 已在 FLOPs 層面隱含；ERA 也已顯示 batch 下 KV 縮 5× 能換到 3.3× 攤銷收益。
- **「KV 受限是人造的」**
  - 本題的收益靠 65,536 token 的小池才出現，H200 預設約 1.7M token。
  - 7B 模型在 max_num_seqs 128、s=4,096 時只需約 0.53M token，預設池下視覺 KV 根本不是瓶頸。審稿人會要求能自然產生 KV 壓力的設定：大模型或 TP、多圖或影片、長上下文、較小的卡。
- **與既有結論的衝突**：沒有直接衝突。
  - PAA-Bench、break-even 研究、ERA 都說「token 少不一定省延遲」，前兩者只看單請求。本題的說法是服務層收益存在而且可以算出來，這是互補，不是矛盾。
  - 但若只用降解析度當干預，會被說「降解析度同時改變資訊量，等於只是改 prompt 長度，跟剪枝無關」。
- **方法面的疑點**：
  - 只有一個模型、一張卡；閉環一次送入；強制輸出長度。
  - W1 曾因 Little's law 量法錯誤判 NO-GO；重算項的形式是事後定的。
  - big 池長輸出的誤差 17.7–23.3%。

### E. 審稿人預期的規模與設定
- **模型**：至少 2–3 個 VLM 家族（Qwen2.5／3-VL、InternVL3、LLaVA-OV），含 32B／72B 的 TP 多卡。
- **輸入**：多圖、影片，或高解析度文件——要讓視覺 token 自然主導 KV。
- **剪枝**：真剪枝方法整合進 vLLM 或 SGLang，至少 VisionZip 這種 LLM 前剪、FastV／PDrop 這種層內剪各一，外加降解析度基線；encoder 與剪枝開銷要算進去。
- **負載**：開環 Poisson 或真實 trace，報 SLO goodput、TTFT／TPOT 的 p50／p99、最大可持續到達率。
- **系統設定**：預設 KV 池、chunked prefill、prefix cache 開啟的情況都要有。
- **比較對象**：Nie 等的 μ 公式、Vidur 或其他模擬器、profiling 基線。
- **品質**：品質與容量的 Pareto（MathVista、DocVQA、Video-MME 等）。
- **硬體**：至少兩種 GPU。

### F. 發表場地與 ICML 適配度
- 剪枝與 benchmark 類多在 CV／NLP 會議：VTC-Bench 在 ACL 2026，HAP 頁面標 EMNLP 2026，其餘多是 arXiv 或 CVPR。
- serving 系統類在 MLSys、OSDI、SoCC、EuroSys：Vidur 在 MLSys 2024；ModServe 頁面未列發表處。
- 排隊與容量理論已進到 ICML：Nie 等在 ICML 2026，Ao 等列在 math.OC。
- **ICML 適配度：中偏低。**
  - 理論貢獻要能撐起 ICML，就得超越 Nie 等。例如對 preemption／重算給出可證明的修正，或給出「何時剪枝值得做」的閉式判準加界。
  - 若定位是「VLM 的實證容量研究」，比較適合 MLSys 或 EMNLP／ACL 的 efficiency track，或 workshop。
  - 以目前單模型、人造池的證據，送 ICML 主會風險高。

### 未確認
- VL-Cache 的吞吐與 batch 細節。
- EffiVLM-Bench（ACL 2025）的具體速度數字。
- Ao 等 2606.15555 的 GPU 實驗數字。
- ModServe 的正式發表處。
- Semantic Scholar 查詢被限流（HTTP 429），沒用到。
