# 實驗硬體與顯存紀錄

涵蓋本輪已建立實驗的 10 題。下一步的設備判斷更新至 2026-11-15 10:58（台北）。歷史表保留 04:38 的現場紀錄。

## 下一步用 24 GB 還是 H200

**24 GB 優先試跑**的四項是多條件檢索、兩個時鐘的階層評估、LLaDA 平行解碼、Attention gate 小模型。表內 GPU 實測來自 H200。24 GB 欄是依同配置量測或程式配置推估，尚無同配置的 24 GB 實測。

| 題目 | 下一步 | H200 已實測峰值 VRAM | 24 GB 估算與設備安排 |
|---|---|---|---|
| 多條件檢索 | `26534` 完成新庫排名，`26568` 測實際重排 | 新庫 `26534` **6.41 GiB、4 分 29 秒**，主機 RAM 約 **4.52 GiB**；正反關係 `26514` **6.89 GiB、1 分 24 秒**。新候選庫實佔 **1.9 GiB 磁碟**。 | **CORE 2B 固定索引有望用 24 GB。** 此作業用圖像 batch 8、文字 batch 32。新留出池 5,200 對交叉編碼先用 H200，估 10–45 分鐘、上限 90 分鐘，另量峰值。新圖檔與清單實佔約 0.63 GiB。[新庫預登記](experiments/retrieval-conjunction/PREREG.md) · [獨立重排](experiments/retrieval-rerank/PREREG.md) |
| 兩個時鐘 | `26523` 完成，`26559` 開始跑唯一本數對照 | 三臂回播 `26523` **2.41 GiB、35 分 41 秒**，主機 RAM 約 **33.85 GiB**。正式評估 `26383` **2.87 GiB、57 分 7 秒**。 | **同類作業可先試 24 GB 顯存，主機 RAM 另計。** 新兩臂在 H200 估 **45–70 分鐘**，暫存 checkpoint 估 **6.5 GiB**。CIFAR 依預登記暫緩。[新對照](experiments/two-clocks-count/PREREG.md) |
| Inoculation 防禦漏出預測 Λ | `26562`／`26563` 路徑修正後重跑作者訓練 | Phase A v2 7B FP32 梯度與有限差分 `26397` 為 **71.7 GiB**。作者 LoRA VRAM 待量。鎖版環境實佔 **5.9 GiB 磁碟**。 | **訓練與生成先留 H200。** BF16 7B 權重約 14 GB，訓練另需活化、梯度和 optimizer 狀態。訓練每臂估 **45 分鐘**，兩個 LoRA 合計估 **1.20 GiB 磁碟**。生成估 **15–45 分鐘**。複核材料與 GCD 信度評分用 CPU，複核材料估低於 10 MiB。[作者訓練](experiments/lambda-author-training/README.md) · [複核流程](experiments/lambda-author-review/README.md) |
| 多保真度資料歸因 | `26535` 正跑獨立 seed，`26574` 等候 CPU 跨軌跡比較 | `26517` **126.45 GiB、28 分 28 秒**，含 JAX 預留，系統 RAM 約 **6.76 GiB**。CPU 評分 `26449` 3 秒。 | **精確探針留 H200，評分用 CPU。** `26535` 含 100 次重訓，單卡估 **65–80 分鐘**、暫存約 **0.8 GiB**。24 GB 需另量未預留峰值。`26574` 的 Slurm 請求無 GPU，上限 10 分鐘。[新軌跡協定](experiments/multifidelity-second-seed/PROTOCOL.md) · [跨軌跡比較](experiments/multifidelity-cross-seed/PROTOCOL.md) |
| Neural Chameleons 防禦 | `26571`／`26572` 比較兩個 clean 面板 | BF16 Gemma 2 9B 全量抽取 `26391` 為 **115.0 GiB**，快取重分析 `26392` 為 **0.98 GiB**。 | **核對用 CPU，快取分析可試 24 GB，原配置重抽留 H200。** 新 clean-only 試驗估 **25–40 GiB、2–8 分鐘**，硬上限 20 分鐘，試跑含最長 352-token 序列，現有估算先留 H200。[抽取程式](../killtests/chameleon-defense/scripts/extract_acts.py) · [快取分析](../killtests/chameleon-defense/slurm/03_analyze.sbatch) · [清潔面板對照](experiments/chameleon-clean-panel/PREREG.md) |
| RL 稀疏決策蒸餾 | `26569` 等待文字停止配對診斷 | B 正確性 `26499` **12.73 GiB、32 秒**。B 16K `26500` 約 **127 GiB**，兩臂分別 **10 分 28 秒／3 分 46 秒**，含預留。CPU `26525` **11 秒**，無 GPU。 | **單模型低併發可試 24 GB，現有長批與雙模型介入留 H200。** 1.5B BF16 權重約 2.9 GiB，每條近 16K 的 BF16 KV cache 約 0.44 GiB。雙模型各批 256 條且近 3,000 token 時，KV 約 41 GiB。`26569` 留 H200，上限 30 分鐘、估暫存低於 2 GiB。[新停止診斷](experiments/rl-author-stop/PROTOCOL.md) |
| 文生圖罕見失敗 | `26570` 量三偵測器資源，`26582` 同分確認完成，`26585` 排隊比較次鍵 | `26516` **36.14 GiB、1 小時 31 分 18 秒**，單卡 30,000 張約 **5.48 張／秒**，含 CLIP。完整 SDv1.4 bench `26288` **52.9 GiB**，batch 32 **5.09 張／秒**、batch 64 **5.25 張／秒**。 | **現有 batch 32 留 H200。** 依完整 SDv1.4 batch 32 實測換算階段 1 約 **9.17 GPU 小時**，六模型吞吐待量。`26570` 隔離診斷上限 15 分鐘。CPU `26582` **1 秒、約 19.8 MiB RAM**，無 GPU。`26585` 的 Slurm 請求無 GPU，上限 5 分鐘。24 GB 縮批須同時通過同種子翻轉率 ≤1% 和吞吐 ≥4.5 張／秒。[重驗預登記](experiments/t2i-gate-b-precision/PREREG.md) · [候選修復核對](experiments/t2i-gate-repair/README.md) · [偵測器資源診斷](experiments/t2i-detector-probe/README.md) |
| 大型平行解碼提交集合 | `26530` 完成 640 題，暫停此版規則 | LLaDA-8B 全程 `26530` **16.58 GiB、18 分 39 秒**。兩題 `26529` **15.10 GiB PyTorch 已配置、15.22 GiB 已保留、23 秒**。 | **同配置有望用 24 GB。** 權重約 14.93 GiB。H200 已覆蓋本輪完整面板，換卡的顯存及執行時間待實測。CPU 可彙總。[新試驗預登記](experiments/large-commit-sets/top-two-selector/PREREG.md) |
| 不完美驗證器 | `26565` 使用同一面板重跑真實開發診斷 | `26484` **110.44 GiB、5 分 45 秒**，含 vLLM 預留。短合成作業 `26520` 的 Slurm 取樣峰值 **19.46 GiB、1 分 7 秒**，可能漏掉瞬間尖峰。 | **整個推論作業留 H200。** 0.6B 單獨有望用 24 GB。強端權重約 18 GiB，現行 0.78 顯存比例在 24 GiB 約給 18.72 GiB，另需 KV cache 和運行資源。開發作業上限 60 分鐘，CPU 已核對每端 4,494 個唯一請求。CPU 評估用正式面板大小的合成資料實測 **7.20 秒、約 122 MiB RAM**。[開發預登記](experiments/fallible-verifier/real_holdout/PREREG_dev_likelihood.md) |
| Attention gate × optimizer | `26521` 的四個 GPU 分段及 CPU 彙總 `26522` 均完成 | 四段各 **3.72 GiB、18 分 00 秒至 18 分 11 秒**，單段主機 RAM 約 **2.02–2.04 GiB**。CPU 彙總約 **336 MiB RAM、3 秒**，無 GPU。 | **同配置可優先試 24 GB，估約 4–6 GiB。** 序列 256、microbatch 8，換卡需量吞吐。[48 種子驗證](experiments/gate-optimizer/round3/PREREG.md) |

已送出的工作依原 Slurm 佇列執行。24 GB 試跑先核對同一份權重、資料、精度與輸出，並記錄目標卡的取樣峰值和耗時。只有縮批或降併發才放得下的階段，須確認配置仍符合該題的預登記，並另驗吞吐。[逐題進度](reports/2026-11-15-topic-status-codex.md) · [Slurm 提交紀錄](../killtests/SUBMITTED.log)

## H200 各階段實測

- `h200q` 有一個 128 CPU、8 張 NVIDIA H200 的共享節點。每張 H200 的原廠規格為 141 GB HBM3e。本研究列出的 GPU 作業每個執行分段用一張卡，陣列可以把不同 seed 或分段平行派到多張卡。共享 `lab_limit` QoS 合計 4 GPU、64 CPU、680 GiB 主記憶體，單一作業時間上限 48 小時。[NVIDIA H200 規格](https://www.nvidia.com/en-in/data-center/h200/) · [研究清單](../shortlist.md)
- **腳本申請**是 Git 內的 `#SBATCH` 預設值或提交時指定的時限。抽查表內作業的 Slurm 紀錄，實際申請均為 16 CPU、170 GiB 系統 RAM，包含表內列出的 CPU 作業。`--mem` 是系統 RAM；GPU 由 `--gres` 申請。[送件紀錄](../killtests/SUBMITTED.log)
- **實測 VRAM**用 Slurm `.batch` 的 `TRESUsageInMax` 內 `gres/gpumem`，原值換算成 GiB。叢集約每 30 秒取樣，數字包含執行框架的顯存預留，可能漏掉短暫尖峰。**實測 RAM**取同一 `.batch` 步驟的 `MaxRSS`，是步驟中單一 task 的最高常駐記憶體。多作業同時執行的合計 RAM 須另計。表內耗時取 `Elapsed`，以單作業或單陣列分段計算，排隊及其他分段時間需另計。各題目前可核對的 GPU 紀錄來自 H200，其他卡型的最低顯存仍待驗證。[Slurm GPU 計量](https://slurm.schedmd.com/gres.html#Accounting) · [Slurm 作業欄位](https://slurm.schedmd.com/sacct.html)

| 題目 | 每作業的 GPU／腳本申請 CPU、RAM | 實測顯存、耗時與 RAM | 下一階段的排程與注意事項 |
|---|---|---|---|
| 條件擴散偏差泛化的兩個時鐘（`biased-gen-cfg`） | 1 張，8 CPU／40 GiB | CelebA 六個訓練分段各 **14.1 GiB**、約 **14 分鐘**（`26279`）。階層訓練的 K=6、K=1 約 **2.7 GiB、39 分鐘／分段**（`26380`、`26381`），正控制已完成分段約 **1.8 GiB、35 分鐘／分段**（`26382`）。訓練代表 RAM 約 3.3 GiB，CelebA 評估 `26294` 為 7.6 GiB。 | 第 2 週評估 `26383` 上限 4 小時，仍依賴訓練。Checkpoint 磁碟於 04:38 約 **46 GiB**，其中包含舊階段評估仍要用的階層權重。[執行腳本](../killtests/biased-gen-cfg/scripts) · [環境與磁碟](../killtests/biased-gen-cfg/README.md) |
| Inoculation 防禦漏出的 Λ（`inoculation-lambda`） | 1 張，8 CPU／96 GiB | Phase A v2 梯度閘門 `26397` 使用 Qwen2.5-7B fp32，**71.7 GiB、1 小時 32 分**。早期 PG 層段作業 `26335` 為 **81.6 GiB、2 分 41 秒**。v2 的 RAM 為 28.9 GiB。 | v2 腳本預設 6 小時，`26397` 送件上限改為 3 小時。階段使用的模型大小、精度及梯度累積會改變顯存。環境及兩組權重的原磁碟估算為約 52 GB。[v2 入口](../killtests/inoculation-lambda/phasea_v2/run.sbatch) · [環境與磁碟](../killtests/inoculation-lambda/README.md) |
| 文生圖罕見失敗（`t2i-rare-failure`） | 1 張，GPU 基準與 Gate B 16 CPU／64 GiB，大量生成 16 CPU／96 GiB。合成統計閘門在 CPU，8 CPU／32 GiB。 | SD v1.4 基準 `26288`（測 batch 32、64）整趟 **52.9 GiB、5 分 14 秒**。Gate B `26289` 結束前量到 **36.1 GiB、1 小時 32 分**，Slurm 標記失敗。基準 RAM 為 3.7 GiB。 | 大量生成腳本每分段上限 4 小時，待完整實測。權重、環境、輸出原規劃磁碟上限 35 GB。[基準腳本](../killtests/t2i-rare-failure/slurm/bench.sbatch) · [階段腳本](../killtests/t2i-rare-failure/slurm) · [磁碟規劃](../killtests/t2i-rare-failure/README.md) |
| 多保真度資料歸因（`multifidelity-attribution`） | 1 張，8 CPU／64–128 GiB。後續精確探針 8 CPU／96 GiB，CPU 評分 2 CPU／12 GiB。 | 第 0 階段重訓 `26326`、精確歸因 `26327` 都量到 **126.4 GiB**，含 JAX 預留，分別約 **12–15 分鐘／分段**與 **11 分鐘／分段**。便宜方法 `26374` 為 **76.0 GiB、10 分 13 秒**。這三階段代表 RAM 約 5.7–6.6 GiB。 | JAX 設定預留 90% 顯存。另有 [TRAK 投影矩陣約 42 GB 的單項配置測試](../killtests/multifidelity-attribution/scripts/s0_bench.sbatch)。新精確探針 `26448` 每分段上限 45 分鐘，兩段依序用一張卡。全題原規劃磁碟上限 80 GB。[JAX／PyTorch 環境](../killtests/multifidelity-attribution/scripts/common.sh) · [精確探針](../killtests/multifidelity-attribution/exact_probe/exact.sbatch) · [磁碟規劃](../killtests/multifidelity-attribution/README.md) |
| RL 稀疏決策蒸餾（`rl-sparse-distill`） | 1 張，16 CPU／170 GiB | 配對 A 長批 vLLM 抽樣 `26292`、`26293` 約 **127.2–127.3 GiB**，各 **1 小時 22 分、51 分鐘**，RAM 分別為 10.4、7.2 GiB。 | vLLM 預設 `gpu_memory_utilization=0.90`，含 KV cache 預留。抽樣上限 6 小時，部分逐步介入腳本上限 24 小時。磁碟原估 ≤25 GB，含 A、B 各一組 base/RL 權重與輸出。[vLLM 參數](../killtests/rl-sparse-distill/scripts/gen_vllm.py) · [分段腳本](../killtests/rl-sparse-distill/slurm) · [環境與磁碟](../killtests/rl-sparse-distill/README.md) |
| Neural Chameleons 防禦（`chameleon-defense`） | 1 張，8 CPU／64 GiB（基準）或 96 GiB（抽取、分析） | BF16 Gemma 2 9B 激活抽取 `26391` **115.0 GiB、6 分 7 秒**。沿用快取的分析 `26392` **1.0 GiB、7 分 37 秒**。兩者的 RAM 分別為 3.6、4.9 GiB。 | 抽取與分析上限分別 3、2 小時。重跑磁碟主要是約 18 GB 模型與每份約 2 GiB 激活快取。顯存峰值取自完整抽取，換模型或改 batch 後須重新量測。[抽取腳本](../killtests/chameleon-defense/slurm/02_extract.sbatch) · [分析腳本](../killtests/chameleon-defense/slurm/03_analyze.sbatch) · [磁碟規劃](../killtests/chameleon-defense/README.md) |
| 多條件檢索的單調證據聚合（`compositional-retrieval`） | 1 張，8 CPU／32 GiB（基準）、48 GiB（主要流程）、56 GiB（補量） | `26314` 探針 **2.1 GiB、1 分 10 秒**。多模型補量 `26393` **6.0 GiB、10 分 24 秒**，RAM 為 4.6 GiB。 | 新檢索作業 `26450` 上限 1 小時，預先估算 5–25 分鐘，兩個模型逐次載入。保留 COCO、Half-Truths 和各模型快取，首輪預登記磁碟上限 40 GB。[補量腳本](../killtests/compositional-retrieval/slurm/headroom.sbatch) · [新檢索配置](../killtests/compositional-retrieval/PREREG_prospective_rank.md) · [原預登記](../killtests/compositional-retrieval/PREREG.md) |
| 擴散 LM 平行解碼（`large-commit-sets`） | 1 張，4 CPU／48 GiB。第二輪 CPU 評分 4 CPU／12 GiB。 | 第一輪 BF16 LLaDA-8B `26376` **16.3 GiB、13 分 35 秒**，RAM 為 1.0 GiB。 | 第二輪試跑／訓練候選收集／鎖定測試時限分別 30 分、2 小時、3 小時，CPU 評分另限 30 分鐘。模型約 16 GB 磁碟。[第一輪說明](experiments/large-commit-sets/README.md) · [第二輪說明](experiments/large-commit-sets/round2/README.md) |
| 不完美驗證器（`fallible-verifier`） | 第一輪 CPU 重播 1 CPU／4 GiB；第二輪推論 1 張、4 CPU／48 GiB，分析 1 CPU／4 GiB。 | CPU 重播 `26371` **27 秒**，Slurm 系統 RAM 峰值約 **80 MiB**。第二輪截至查詢時尚未跑，GPU 顯存待量。 | 推論 `26444` 上限 1 小時。Qwen3-0.6B BF16 與 Qwen3-32B AWQ 在 vLLM 依序推論，`gpu_memory_utilization=0.78` 為框架配置。CPU 分析上限 10 分鐘。[第一輪說明](experiments/fallible-verifier/README.md) · [第二輪說明](experiments/fallible-verifier/round2/README.md) · [推論設定](experiments/fallible-verifier/round2/judge.py) |
| Attention gate × optimizer（`gate-optimizer`） | 1 張，8 CPU／32 GiB | 第一輪 `26372` **3.7 GiB、8 分 47 秒**，RAM 為 2.4 GiB，使用 BF16 autocast。 | 第二輪 `26446` 上限 2 小時，程式另限 6,500 秒。OpenWebText 語料與執行快照保留在共用磁碟。[第一輪執行](experiments/gate-optimizer/study.sbatch) · [第二輪配置](experiments/gate-optimizer/round2/PREREG.md) |

## 複查 Slurm 紀錄

在叢集登入節點用表內的 job ID 查詢。使用作業列判斷狀態、實際申請與時間上限，使用 `.batch` 列讀實測顯存與 `MaxRSS`（主記憶體）。

```sh
sacct -j 26393 -P \
  --format JobID,State,Elapsed,Timelimit,ReqCPUS,ReqMem,AllocTRES,MaxRSS,TRESUsageInMax%240
```

大檔和可重建快取在叢集留存。Git 記錄設定、可用的執行入口、預先登記、提交作業 ID 和小型研究紀錄。磁碟預算含環境、模型、資料及 checkpoint，與顯存分列。[協作與磁碟規則](../COORDINATION.md)
