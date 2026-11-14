# 實驗硬體與顯存紀錄

涵蓋本輪已建立實驗的 10 題。下一步的設備判斷依 2026-11-15 04:55（台北時間）的配置和 Slurm 紀錄。下方各階段實測表保留 04:38 的資料。

## 下一步用 24 GB 還是 H200

**24 GB 優先試跑**的四項是多條件檢索、兩個時鐘的階層評估、LLaDA 平行解碼、Attention gate 校準。表內所有 GPU 實測都來自 H200。對 24 GB 卡的判斷是配置估算，這批實驗尚無同配置的 24 GB 成功紀錄。排程前以目標卡的實際可用顯存為準。

| 題目 | 下一步 | H200 已實測峰值 VRAM | 24 GB 估算與設備安排 |
|---|---|---|---|
| 多條件檢索 | 固定候選圖的新排名評估 `26450` | 前一輪多模型補量 `26393` 為 **6.0 GiB**。新排名配置待量。 | **優先試 24 GB。** CS-CLIP 與 CORE 2B 逐一載入，圖像 batch 8、文字 batch 32。先用相同精度與批次核對代表查詢的分數及排序。資料建構與彙整可用 CPU。[排名腳本](../killtests/compositional-retrieval/scripts/prospective_rank.py) · [凍結配置](../killtests/compositional-retrieval/configs/params_headroom.json) |
| 兩個時鐘 | 第 2 週五組正式階層評估 `26383` | 第 1 週階層評估 `26295` 為 **1.64 GiB**。第 2 週階層訓練峰值 **1.8–2.7 GiB**。完整第 2 週評估待量。 | **優先試 24 GB。** 第 2 週同一組最多載入六個小模型並保留配對預測，會比第 1 週增加用量，仍有顯存餘裕可試。保留完整 checkpoint、時間點和種子，驗證分數一致。最後彙整可用 CPU。[評估入口](../killtests/biased-gen-cfg/scripts/w2_hier_eval.sbatch) · [評估程式](../killtests/biased-gen-cfg/scripts/eval_hier_oracle.py) |
| Inoculation 防禦漏出預測 Λ | Phase B 無防禦／IP 重現與 LoRA 信度試驗 | Phase A v2 7B FP32 梯度與有限差分 `26397` 為 **71.7 GiB**。Phase B 待量。 | **Phase B 先留 H200。** 7B BF16 權重約 14 GB，訓練須再加活化、梯度與 optimizer 狀態。Phase B 的訓練配置尚未凍結，24 GB 容量待量。協定檢查與合成測試可用 CPU。[Phase B 預登記](../killtests/inoculation-lambda/PREREG.md#3-phase-b-現象訊號與空間閘門-13-皆過才判) · [Phase A v2](../killtests/inoculation-lambda/phasea_v2/README.md) |
| 多保真度資料歸因 | 兩段各六個查詢的 JAX 精確歸因 `26448`，再做 CPU 評分 `26449` | 先前精確歸因 `26327` 為 **126.4 GiB**，含 90% JAX 預留。便宜方法整批 `26374` 為 **76.0 GiB**。新探針待量。 | **精確探針留 H200，評分用 CPU。** 六欄依序分段仍要重播原訓練軌跡、保留中間狀態並做向量化反向微分。24 GB 要另量實際運算需要多少顯存。原 TRAK 的投影矩陣本身約 42 GB，原配置也留 H200。[精確探針協定](../killtests/multifidelity-attribution/exact_probe/PROTOCOL.md) · [JAX 作業](../killtests/multifidelity-attribution/exact_probe/exact.sbatch) |
| Neural Chameleons 防禦 | 核對作者基準，分析既有快取，必要時完整重抽 | BF16 Gemma 2 9B 全量抽取 `26391` 為 **115.0 GiB**，快取重分析 `26392` 為 **0.98 GiB**。 | **核對用 CPU，快取分析可試 24 GB，原配置重抽留 H200。** 抽取使用 eager attention 和 batch 16，收集多層激活。若要改用 24 GB，先用含長序列的小批試跑，逐例核對分數與檢出判定。[抽取程式](../killtests/chameleon-defense/scripts/extract_acts.py) · [快取分析](../killtests/chameleon-defense/slurm/03_analyze.sbatch) |
| RL 稀疏決策蒸餾 | 配對 A 既有輸出重判，配對 B 原作 16K 重現及短預算資格 | 配對 A 長批 vLLM `26292`／`26293` 為 **127.2–127.3 GiB**，含 90% 框架預留。B 尚無實測。 | **A 重判用 CPU，B 的單模型低併發可試 24 GB，原配置的大批抽樣與雙模型介入留 H200。** 1.5B BF16 權重約 2.9 GiB，每條近 16K 的 BF16 KV cache 約 0.44 GiB。先量低併發的峰值與吞吐，保持原作長度、取樣與模板。雙模型每批 256 條若都接近 3,000 token，光 KV 就約 41 GiB。[配對預登記](../killtests/rl-sparse-distill/PREREG.md) · [vLLM 配置](../killtests/rl-sparse-distill/scripts/gen_vllm.py) |
| 文生圖罕見失敗 | CPU 尾端模擬與既有 CLIP 分數同分診斷 | 篩選閘門 `26289` 量到 **36.1 GiB**，Slurm 記為失敗。原 batch 32／64 吞吐測試 `26288` 峰值 **52.9 GiB**。 | **現在用 CPU。** 若篩選達標，後續原批次的大量生成留 H200。24 GB 可另驗 SD v1.4 的縮批配置，須同時通過同種子翻轉率 ≤1% 和吞吐 ≥4.5 張／秒。[預登記](../killtests/t2i-rare-failure/PREREG.md) · [吞吐測試配置](../killtests/t2i-rare-failure/slurm/bench.sbatch) |
| 大型平行解碼提交集合 | 第二輪兩題試跑、480 題訓練候選、1,000 題鎖定測試 `26440`–`26442` | 同一 BF16 LLaDA-8B 的首輪 `26376` 為 **16.3 GiB**（Slurm），PyTorch 配置峰值 **15.11 GiB**。第二輪待量。 | **優先試 24 GB。** 單題依序執行，每次 forward 的 batch 為 1，關閉 KV cache，仍使用 256-token 解碼配置。先跑凍結的兩題並量顯存和吞吐，再決定整批能否在 2／3 小時的時限內完成。同一速度比較的兩臂用同卡型。評分 `26443` 用 CPU。[第二輪說明](experiments/large-commit-sets/round2/README.md) · [首輪 forward](experiments/large-commit-sets/pilot.py) |
| 不完美驗證器 | Qwen3-0.6B 與 Qwen3-32B AWQ 依序判決 `26444`，再用 CPU 分析 `26445` | 第一輪是 CPU 重播。第二輪 GPU 顯存待量。 | **現有整個推論作業留 H200。** 0.6B 單獨有望用 24 GB。強端權重約 18.0 GiB，現行 vLLM 比例 0.78 在可見 24 GiB 時只給 18.72 GiB，還得放 KV cache 和運行空間。兩端的推論成本比較維持同卡型。[執行入口](experiments/fallible-verifier/round2/run.sbatch) · [vLLM 配置](experiments/fallible-verifier/round2/judge.py) |
| Attention gate × optimizer | 33M 模型的 3×／4×／6× 壓力校準 `26446` | 第一輪 `26372` 為 **3.7 GiB**。第二輪待量。 | **優先試 24 GB，估約 4–6 GiB。** 沿用序列長度 256 與 microbatch 8，種子和各臂依序執行。整輪約 20,400 步，先驗證吞吐能否在凍結的 6,500 秒內完成，避免只跑完部分配置。[訓練配置](experiments/gate-optimizer/config.json) · [第二輪預登記](experiments/gate-optimizer/round2/PREREG.md) |

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
