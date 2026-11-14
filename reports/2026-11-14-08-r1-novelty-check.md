## R1 文獻新穎性查核（2026-11-14，唯讀，沒有改任何檔案）

讀過的資料：repo README、go-topics/R1.md 與 R1-詳細.md（cards/ 的內容透過詳細版間接讀到；repo 裡沒有 paper/ 和 references）。搜尋只用技術關鍵字。以下標「已開」的，是我實際打開過 arXiv HTML 或摘要頁、核對過內容的論文。

### A. 本題複述
在同步 GRPO／DAPO 的 RL 裡，固定一批 rollout 的輸出長度、policy、GPU 與 kernel，問排程本身最多能省多少 decode 時間。

做法是把每步時間的下界設成仿射形式 t(b, KV) ≥ α + βb + γKV，三個係數用 LP 從 microbench 格子擬合。不論怎麼排程，步數 ≥ L_max−1，總 token 數 W 和總 context 讀取量 A 都不變，所以得到與排程無關的下界 LB2 = α(L_max−1) + βW + γA，可省比例的上界就是 1 − LB2/T_obs。

結果：在 Qwen3-1.7B、單張 H200、每批 48 條同時開始、不發生 preempt 的設定下，上界是 4.1–8.0%。天真估計「1 − 平均長度／最長長度」是 18–68%，題目據此主張天真估計高估 4–17 倍。第二輪（W2）會把 4 批合成一池，在 vLLM 上重播 tb（RollPacker 式）、fcfs、lpt 等排程，看它們實際拿到上界的幾成。

### B. 最接近的工作（依重疊程度排序）

| # | 論文（已開／未確認） | 方法一句話 | 報告的收益 | 與本題重疊 | 本題仍不同之處 |
|---|---|---|---|---|---|
| 1 | SortedRL: Accelerating RL Training for LLMs through Online Length-Aware Scheduling，Zhang、You、Qiu 等（MSRA），arXiv 2603.23414（已開）。搜尋結果有 ICML 2025 頁面，可能是 workshop，未確認 | 依輸出長度重排 rollout，短的先更新；另用快取保存未完成樣本 | 完全 on-policy 的版本吞吐只 +7.57%；partial（跨版本）版本 +39.48%。bubble ratio 從 74% 降到 6% 以下 | 「純長度感知排序」在 on-policy 下只拿到約 7.6%，量級和本題上界幾乎一樣。它的 bubble ratio（式 4）= Σ(Q−r_k)Δt_k/(TQ)，是時間加權的閒置槽位比例，本質接近本題批評的天真估計 | 沒有上界，也沒有步時模型；不談槽位比例和實際時間的落差 |
| 2 | Beat the long tail: Distribution-Aware Speculative Decoding for RL Training，Shao、Srivatsa 等（UIUC、Together、UCSD、Prime Intellect），arXiv 2511.13841（已開，未註明發表處） | 長尾階段的推測解碼預算最佳化 | 數學 RL rollout 省 >50%，程式 RL 約 25%（8×H100，16K） | **建模幾乎同構**：t_fwd = c_base + c_tok·n_toks，總時間 = c_base·N_fwd + c_tok·N_toks + C，也明說最長的序列決定 N_fwd。這就是 LB2 去掉 γA 那一項 | 它拿這個模型做 SD 預算，不當成排程無關下界；沒有 context 項；沒有 LP 擬合下界 |
| 3 | RollPacker，Gao Wei 等，NSDI'26，arXiv 2509.21009（已開 v1） | tail batching：短輪發 1.25 倍的題目和回應，只取先完成的，其餘丟棄；被中止的題目延到專門的長輪。另有平行度規劃、reward 排程、stream trainer | 只開 tail batching 時端到端 1.30×／1.48×／2.21×（7B/8k、14B/16k、32B/32k）；整套比 veRL 2.03–2.56×；短輪平均 rollout 最多 7.8× 快。最多 128 張 H800 | 同樣以「長尾讓 GPU 閒置」為動機；本題的 tb 是它的固定輸出 oracle 版 | 它會改變每步的批次組成並丟棄回應，而且是多 GPU 加上大 batch（128×8）。量測方式是每步時間，不是固定工作量的 makespan。沒有上界 |
| 4 | Seer，Qin 等，OSDI'26，arXiv 2511.14617（已開 v2） | divided rollout（切 chunk、跨 instance 負載平衡、全域 KV 池）、近似最長先的 context-aware 排程、grouped SD | 消融結果：divided rollout 1.27–1.35×，加上排程後 1.33–1.47×（排程本身約 +5–13%），再加 SD 1.77–1.87×。另外，近似排程達到 oracle 長度最長先的 95%。最多 256 張 H800、最長 98K | 有「oracle 長度排程」的參考點；排程本身的增益只有個位數到十幾 % | 主要收益來自多 instance 負載不均和 KV preempt，以及 SD；沒有排程無關下界 |
| 5 | APRIL，Zhou 等（AMD、CMU、LMSYS、UCLA），arXiv 2509.18521（已開 v3）。ICLR'26 未確認 | 超額發 rollout，湊夠就停，未完成的放進 buffer 下一步續跑 | 吞吐平均約 +22.5%，單一設定最多 37.7%，引言寫「最多 49.5%」（Qwen3-8B）。單節點 8×MI300／H100，slime | 對象相同（DAPO-Math-17k、Qwen3） | 跨 policy 版本，約 40% 的 token 來自續跑；吞吐以 tokens/s 計，跨步的工作量不固定。屬於本題排除在外的比較類別 |
| 6 | BubbleSpec，Xu 等（SJTU、ByteDance），arXiv 2605.08862（已開，發表處未確認） | 用長尾泡泡時間預先生成 draft，做推測式 rollout | rollout 時間省 30–45%（1.4–1.8×）；泡泡占 rollout 時間「1/3 以上」（Qwen3-1.7B 平均 61.9 s／306 s，約 20%，文中分母不清） | 模型和規模跟本題最像：Qwen3-1.7B、單節點 8 卡、verl。「泡泡占比」也是同一類量 | 增益來自改變解碼（SD），不是排程 |
| 7 | Laminar，Sheng 等（HKU、ByteDance Seed），arXiv 2510.12633（已開）。ACM DL 有 DOI，發表處未確認 | 非同步 RL，動態 repack，把長尾軌跡合併到少數 rollout instance；用 roofline 定 decode batch 上限 | repack 讓生成吞吐 +26%；整體比 verl 2.56×，最多 5.48×，1024 張 H800 | 「decode 在 memory-bound 區，加 batch 幾乎不加延遲」（roofline），正是本題 β 很小這個事實的通用說法 | 跨版本、多 instance；不給上界 |
| 8 | Understanding Stragglers in Large Model Training Using What-if Analysis，Lin、Li 等（NYU、ByteDance），OSDI'25（已開搜尋摘要與 USENIX 頁） | 以模擬「沒有 straggler 時會多快」量化損失 | 42.5% 的 job 因 straggler 慢 ≥10% | 方法論的先例：用 what-if 給收益上限 | 對象是預訓練，不是 RL rollout |
| 9 | Speculative Decoding: Performance or Illusion?，Liu、Yu、Park、Stoica、Cheung，MLSys 2026 oral，arXiv 2601.11580（已開摘要頁） | 在 vLLM 上系統量測 SD，對照理論上限 | 實測和上限差很遠 | 「宣稱收益 vs 上限」型論文被頂級系統會議接受的近期先例 | 對象是 SD |
| 10 | 推論服務的排程理論（全為已開摘要）：Fu 等 Learning to Rank（NeurIPS'24，2.8× 延遲、6.5× 吞吐）；Jaillet 等 Online Scheduling with KV Cache Constraints（arXiv 2502.07115，以 hindsight IP 為基準）；Ao、Luo、Simchi-Levi、Wang 的 fluid-guided（NeurIPS'25）；Li、Dai、Peng 的 Throughput-Optimal（arXiv 2504.07347） | SJF 近似、KV 受限下的競爭比、fluid 基準、work-conserving 吞吐最優 | — | 都有離線最優或理論基準的概念 | 目標是延遲或穩態吞吐，不是固定一批的 makespan；沒有用實測步時做仿射下界 |

其他只做定位、與本題重疊較低的：TLT（ASPLOS'26，arXiv 2511.16665，自適應 drafter，1.7–2.1×）、EfficientRollout（arXiv 2606.18967，自我推測解碼，rollout 最多省 19.6%，並指出 batch 縮小會讓 decode 轉向 memory-bound）、NVIDIA 的 SD for RL（arXiv 2604.26779，有 Amdahl 型的 step 上界，但是針對 SD）、RhymeRL（arXiv 2508.18588，用歷史長度排序做負載平衡，2.6×；ASPLOS'26 只在搜尋結果的 URL 看到，未確認）、RL in the Wild（arXiv 2509.25279，特性分析，不給 what-if 上界）、SpecActor（arXiv 2511.16193）。DARTS（2605.30859）、SAGC（2606.02218）只看過搜尋摘要，內容未確認。

### C. 判斷：沒有人做過實質相同的事，但剩下的新穎性很窄
- 我沒找到任何工作給出「固定輸出的 RL rollout makespan、與排程無關、用實測步時擬合的下界」，也沒有人正式指出 1 − 平均／最長會高估。撞題搜尋（rollout makespan lower bound、tail batching optimality、step-time model）都沒有結果。2026-09 的 rollout 效率綜述（arXiv 2609.25463）摘要也沒有把這件事列為已解決或開放問題。
- 但各個組成都已經有人做過：
  - 仿射或線性步時模型與「最長序列決定步數」已在 Beat-the-long-tail 中出現。
  - 「小 batch 時 decode 是 memory-bound，閒置槽位幾乎不花時間」是 roofline 常識，Laminar、EfficientRollout 都寫明。
  - oracle 排程的參考點 Seer 有。
  - SortedRL 的 on-policy 排序只得 7.6%，其實已經間接顯示純排序的空間小。
- 剩下的新穎性具體只有三塊：
  1. 一個用 LP 擬合、可驗證的排程無關下界，包含 context 項 γA。
  2. 量化指出常用的天真估計或 bubble ratio 與可省時間之間差了一個數量級。
  3. 在同一池輸出上做配對重播，量出各排程拿到上界的幾成。
- 第 3 塊目前還沒做，是 W2 才要做的。

**有一個結構性問題要先講（這是我根據 repo 的定義推得的，不是文獻的結論）**：在 W1r2 的設定下，每批 48 條、併發上限 48、全部同時開始、不 preempt。這時候 seq 本身就已經是步數最少的排程（K = L_max−1），而且所有 token 都在最早的時間點被處理。所以在「固定輸出、單批」這個比較類別裡，真實可省的量大概接近 0，「4–8%」比較像是仿射下界本身的鬆弛量，而不是排程還有多少空間。審稿人很容易看出「單一封閉批次沒有別的工作可以拿來填長尾」，因此天真估計和這個上界回答的是不同的問題：天真估計隱含「還有積壓的工作可以補進閒置槽位」，上界則是封閉批次的 makespan。

W2 的池化設計才有真正的排程自由度，而 repo 自己的模擬顯示，自由度一放開，數字就往天真估計靠近：
- 池上界：15%
- lpt：11.7%
- lpt256（上限放寬到 256）：18.5%
- 池的天真估計：36%

### D. 是否與既有論文的收益衝突
表面上衝突（30–50%、2× vs 4–8%），實質上大致不衝突。差異來源依影響大小排列：

1. **工作量不固定或可以重新分配**。APRIL、SortedRL-partial、Laminar 跨 policy 版本續跑；RollPacker 超額發送後丟棄，並把長題延到別的步。它們都把長尾用的槽位拿去做其他步的工作，這正是本題排除的比較類別。直接證據是 SortedRL：on-policy 7.57%、partial 39.48%，同一篇論文裡差了 5 倍。
2. **多 GPU 或多 instance 的負載不均**。Seer、RollPacker、RL-in-the-Wild 的主要損失是「少數 instance 還在跑長序列，其他 instance 整台閒置」（SM 利用率降到 0），這相當於跨卡的 makespan。本題只量單張卡，同一張卡上縮小的 batch 本來就很便宜。
3. **KV 受限與 preempt**。32B 模型、32K–98K 長度、1024 條以上的 rollout 時，seq 本身就會 preempt 或重算，排程能省掉這些。本題刻意調成 0 preempt。
4. **把推測解碼或其他改變解碼的技術算進收益**。Seer 有 30–44% 來自 SD；TLT、BubbleSpec、SpecActor 主要也是 SD。
5. **量測單位不同**。有的報 tokens/s 吞吐（APRIL），有的報端到端、包含 reward 和訓練重疊（RollPacker 的 stream trainer），有的報短輪的平均 rollout 時間（RollPacker 的 7.8×），本題只量 decode 時間。
6. **模型和硬體區間**。1.7B 在 H200 上極度 memory-bound 或受 overhead 限制（t(256)/t(1)≈3.1，β·48 ≈ 0.37 ms，遠小於 α 2.58 ms）。這是天真估計最失真的區間；換成 32B、TP 多卡或 MoE，結論可能不同。

**審稿人最可能的質疑與需要的回應**
- 「單批 seq 本來就是最優，上界只是下界的鬆弛，結論近乎同義反覆。」需要的回應：主結果改用池化或多批、併發上限小於總量的設定，或改成多 instance 設定，並報告真實排程達到上界的比例。
- 「比較類別把實務上真正有效的手段（跨版本、超額發送、多 instance 平衡、SD）全部排除，結論沒有實用價值。」需要的回應：明確定位成「排程本身的空間 vs 跨版本或改輸出帶來的空間」的分解，並用同一套下界框架擴充到多 instance 和 KV 受限的情況。
- 「天真估計是稻草人，沒人用它宣稱加速。」需要的回應：引用 SortedRL 的 bubble ratio、BubbleSpec 的「泡泡占比」這類實際被拿來當動機的指標，示範它們與可省時間的落差。
- 「仿射下界外插沒有保證；只有 1.7B、單卡、33 題、48 條。」
- 「事後設計」問題，repo 自己已列出。
- 「和 Beat-the-long-tail 的線性步時模型有什麼不同？」要明確區分 context 項、LP 擬合，以及下界的用途。

### E. 審稿人預期看到的規格
- **模型**：至少 7B–32B 的 dense 模型（Qwen2.5／Qwen3、Llama-3.1-8B），最好再加一個 MoE（如 Qwen3-30B-A3B）；Kimi-K2、72B 級是系統會議的上限。
- **資料**：DAPO-Math-17k、DeepScaleR、DeepMath-103K；另加程式（DeepCoder）或工具使用任務；最大長度 16K–32K，系統會議論文有到 64K–98K。
- **系統**：verl、slime、OpenRLHF 或 NeMo-RL，rollout 引擎用 vLLM 或 SGLang；8 卡單節點是最低要求，多節點（32–256 張 H100／H800）是 NSDI／OSDI 的常態。
- **指標**：每步 rollout 時間、端到端步時、吞吐（tokens/s）、GPU SM 利用率或泡泡占比、與 oracle 的比例、準確率曲線（AIME24 等）以證明收益不傷訓練品質；要在真實 RL 迴圈中多步量測，不能只是單批。

### F. 投稿場地適配度
- 同類論文的發表處：RollPacker NSDI'26、Seer OSDI'26、TLT ASPLOS'26、Stragglers what-if OSDI'25、SD「Performance or Illusion」MLSys'26 oral。RL rollout 效率這一支幾乎都在系統會議，而「實測 vs 理論上限」型的批判論文有 MLSys 的先例。
- ML 會議這邊：理論味較重的推論排程論文在 NeurIPS（Learning to Rank'24、fluid scheduling'25）；SortedRL 有 ICML 2025 頁面，但性質未確認。
- ICML 適配度偏低。目前的「理論」只是一個仿射下界加小型 LP，對 ICML 太淺；實驗規模（單卡、1.7B）又達不到系統會議的標準。
- 建議：最合適的是 MLSys（分析加上界，配上多規模實測）。若能補上多 instance、KV 受限和真實 RL 迴圈，可以考慮 NSDI／EuroSys。若要留在 ML 場地，NeurIPS 比 ICML 稍合適，但需要更強的形式化（例如有競爭比或近似比的排程結果），或把它包裝成「RL 效率指標誤用」的評測論文。

### 文件本身記錄錯誤之處
- APRIL 文件寫「最多 +44%」，原文我只看到 37.7%（表）和 49.5%（引言）。
- Seer 的 arXiv 摘要和 OSDI 頁面數字不同：74–97% vs 最多 2.04×。

### 連結
- RollPacker https://arxiv.org/html/2509.21009v1 ・ https://www.usenix.org/conference/nsdi26/presentation/gao-wei
- APRIL https://arxiv.org/html/2509.18521v3
- Seer https://arxiv.org/html/2511.14617v2 ・ https://www.usenix.org/conference/osdi26/presentation/qin
- SortedRL https://arxiv.org/html/2603.23414v1
- Beat the long tail https://arxiv.org/html/2511.13841
- BubbleSpec https://arxiv.org/html/2605.08862v1
- Laminar https://arxiv.org/html/2510.12633v1
- SD for RL https://arxiv.org/html/2604.26779
- EfficientRollout https://arxiv.org/abs/2606.18967
- TLT https://arxiv.org/html/2511.16665v1
- RhymeRL https://arxiv.org/html/2508.18588v1
- RL in the Wild https://arxiv.org/html/2509.25279v1
- SpecActor https://arxiv.org/abs/2511.16193
- Stragglers what-if https://www.usenix.org/conference/osdi25/presentation/lin-jinkun
- SD Performance or Illusion https://arxiv.org/abs/2601.11580
- Learning to Rank https://proceedings.neurips.cc/paper_files/paper/2024/hash/6c8985579293e0209bdaa4f21bb1d237-Abstract.html
- KV-constrained online scheduling https://arxiv.org/html/2502.07115v5
- Fluid-guided scheduling https://arxiv.org/html/2504.11320v2
- Throughput-optimal https://arxiv.org/html/2504.07347v1
- Rollout 效率綜述 https://arxiv.org/abs/2609.25463
