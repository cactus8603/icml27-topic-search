# 大型平行解碼集合的交互作用會不會累積成答題錯誤

日期 2026-11-14（台灣）。範圍包含公開論文、ICLR 2027 討論、作者釋出程式與逐題輸出；已在 CPU 重算公開結果。端到端模型實驗留待下述先導實驗。這份回報只評估 masked diffusion LM 的平行解碼，候選方法和下方判準均為研究提案。

## 結論

**只值得做一次有上限的先導實驗，暫列備案，信心中偏低。** 最新的 SetCommit 已做「同一解碼狀態、更換相同大小的提交集合、跑到最終答案」的因果對照，也有可落地的免訓練選擇器。仍可主張的空間很窄：**在實際解碼選出的 ≥8-token 集合中，成對影響非加成所造成的判斷誤差，有多少能透過保留同樣提交量的介入轉成答對，並且在計入探測成本後改善延遲–準確率前緣？** 論文已能證明局部判斷分歧和大 block 準確率差各自存在。目前還沒有證據證明前者解釋後者。 [SetCommit §4.3、附錄 A/F](https://arxiv.org/html/2610.31120v1) · [Measured Coupling 附錄 H、J、K](https://arxiv.org/html/2611.08307v1)

這題和先前停損的「估資訊曲線、找每輪最佳解多少 token」不同，先固定 block、提交量、原始 argmax 和其餘解碼策略，只更換**同一步一起提交哪些位置**。如果效益來自多跑幾步、延後 EOS，或一般置信度調參，就回到已很擁擠的排程題，不升主線。 [先前深查 43](../../reports/2026-11-14-43-deepcheck-dlm-parallel-decoding-ruler.md) · [Serial Depth](https://arxiv.org/abs/2608.25505) · [Simple DLMs](https://arxiv.org/abs/2609.33947)

## 1. 三個已核實的現象

### CoCom 的 rebuttal 有更公平的基線，理論仍止於單步

ICLR 2027 #20666 *Contextual Commitment* 在 11/14 晚快照由 `[3,3,2]` 升至 `[4,3,2]`。GwZc 的疑問是原 Native 用 full sequence，CoCom 用 block 64。作者在 LLaDA-1.5 固定 256-token canvas、prompt、預測規則及 Native 的 256 次 forward 預算，另跑 Native block 64。GSM8K 的 Native 從 57.24% 回到 82.03%，CoCom 則 80.97%、61.39 NFE。GwZc 引用這個品質和效率取捨，明確升到 4。這個結果證明 block 設定本身就能大幅影響結果，也要求先導實驗固定 block；61.39/256 衡量 forward 次數。 [GwZc review](https://openreview.net/forum?id=BJw3PRAyUN&noteId=TOQtQdQVu0) · [作者回覆](https://openreview.net/forum?id=BJw3PRAyUN&noteId=48ycds756K) · [調分留言](https://openreview.net/forum?id=BJw3PRAyUN&noteId=vsCviUfBFZ) · [Radar 固定快照](https://github.com/dipta007/iclr-2027-rebuttal-radar/blob/a67aefa2c4c2d231e471abab4c36cb771c672ec9/docs/data/papers.json)

另一位 reviewer 3FAD 問的是不同的問題。把 soft input 的插值寫成參考向量的期望，未控制真實解碼連續多步後的預測。作者承認等價只在輸入層成立，解碼每步重算邊際，因此只能在新狀態重新套當步界。作者也以不截斷 FP64 檢查達到零超額，LLaDA-1.5 GSM8K 只從 80.97% / 61.39 NFE 變為 80.89% / 61.56，HumanEval 從 43.90% / 64.76 變為 43.29% / 64.62。3FAD 在該快照仍給 2。**僅把局部預算檢查做得更精確，已知很可能沒有端到端收益。** CoCom 的 soft-input 預算和下文的 signed logit margin 是不同量，兩篇的數字各自解讀。 [3FAD review](https://openreview.net/forum?id=BJw3PRAyUN&noteId=JZgbmUtVqz) · [作者 W1/W3](https://openreview.net/forum?id=BJw3PRAyUN&noteId=eqkcrIzC3d)

### Measured Coupling 已量到非加成，量的是門檻分歧

對原狀態 (x)、固定的候選 token (v_i)，令 (c_i(S)) 是填入集合 (S) 的原始候選後，模型給 (v_i) 相對於最佳替代 token 的 logit 差。比較 (c_i(S)) 和成對加總 (c_i(\varnothing)+\sum_{j\in S}[c_i(\{j\})-c_i(\varnothing)])。單一與聯合探測都採完整 forward 時，差值可排除作者加速 kernel 的近似。原作的上線解碼器另有未探測欄位、上一步提案失效和局部 KV 近似，必須另行拆解。對即將提交的 (i\in S)，要用 (S\setminus\{i\})，保持 (i) 為 mask，否則會把答案本身透露給模型。 [Measured Coupling §3、附錄 A/J/K](https://arxiv.org/html/2611.08307v1)

| LLaDA-8B-Instruct，block | 量測對象 | 一起 commit 2–3 | 4–7 | ≥8 | 全部 |
|---|---|---:|---:|---:|---:|
| GSM8K，20 | 下一步仍 masked 的位置 | 3.7% | 7.8% | 15.6% | 10.0% |
| MBPP，20 | 下一步仍 masked 的位置 | 4.5% | 7.3% | 14.2% | 9.4% |
| GSM8K，32 | 下一步仍 masked 的位置 | 2.4% | 5.7% | 9.2% | 7.0% |
| GSM8K，64 | 下一步仍 masked 的位置 | 1.4% | 3.7% | 10.8% | 7.2% |

來源為附錄 J Table 8，各列是「評估同一原候選時，成對預測和實際聯合探測落在 τ 門檻兩側」的比例。原作還發現兩個方向都會飽和，例如有利訊號被加總預測為約 5.5、實測約 3.5；衝突約 −16、實測約 −3.5。**這個比例有三個不能省略的限制。** 第一，分母為位置，受到長題、多步及殘餘 mask 數影響。第二，評估的是尚未提交的位置、含正負方向，15.6% 不能轉成已提交錯誤率。第三，block 64 的整體分歧率反而低於 block 20，不能用這些跨設定的表格說明 block 64 掉分原因。 [Measured Coupling 附錄 J，Table 8](https://arxiv.org/html/2611.08307v1#A10)

附錄 K 對**已提交位置**做 leave-one-out 時，成對與聯合門檻同側的比例約為 GSM8K 96%、MBPP 97%。對每個已提交 token，把其餘同組 token 填入後，原值不再是 argmax 的比例，在作者 LLaDA block-20 預設約 0.8%（GSM8K 和 MBPP，各 50 題）。這兩個統計各有不同門檻和分母，後者也不等於任務錯誤。反過來，錯一個數字也可能拖垮整道題，因此需要實際介入續跑才能決定要不要做。 [Measured Coupling 附錄 K，Tables 9–10](https://arxiv.org/html/2611.08307v1#A11)

### 公開逐題輸出重算：大 block 確實掉分，資料沒有可連結的步級軌跡

下載作者公開 GitHub 釋出的逐題 JSONL，使用 `flexible-extract`、按 `doc_id` 配對全數 1,319 題 GSM8K，核對 32/64 設定一致覆蓋。以下 MC 都用 12 個融合探測、τ=.90，MF 為其成對相似度基線、τ=.85。NFE 是每題平均。這個重算沒有新的模型推論。 [作者程式及輸出](https://github.com/TalOved1/measured-coupling/tree/main/outputs) · [原文附錄 H](https://arxiv.org/html/2611.08307v1#A8)

| 設定 | 答對題數 | 準確率 | NFE |
|---|---:|---:|---:|
| MC block 20 | 1006 | 76.27% | 40.11 |
| MC block 32 | 957 | 72.55% | 34.23 |
| MC block 64 | 893 | 67.70% | 29.09 |
| MF block 32 | 1008 | 76.42% | 56.44 |
| MF block 64 | 995 | 75.44% | 48.32 |

MC20 相較 MC64，有 211 題答對／後者答錯、98 題反向，配對差 **+8.57 pp**（配對 Wald 95% CI **[+6.00,+11.14]**，exact McNemar (p=1.17\times10^{-10})）。MC32 對 MC64 是 **+4.85 pp [2.26,7.45]**；MF64 對 MC64 是 **+7.73 pp [5.11,10.36]**，但 MF64 多用 19.23 NFE。原文把 MC block 32 的探測增至 K=20，可從 72.55% 恢復至 76.19%；block 64 增至 K=40 仍 67.70%，NFE 降至 27.4。這支持 64 的問題比單純 K 不足更值得查，尚無該 K40 設定逐題檔或同組 wall-clock 可重算。檔案只有最終答覆、正誤、NFE 等欄位，沒有同一步的 (S\)、單／聯合 logits 或 probe 時間。 [原始 MC64 逐題檔](https://github.com/TalOved1/measured-coupling/blob/main/outputs/llada_gsm8k__full__fusedm_signed_k12_t0.9_b64.jsonl.gz) · [原文 Table 7](https://arxiv.org/html/2611.08307v1#A8)

程式還放出一個 `flip_rho0.05_b20`：用下一步的 logit 資訊線上估 argmax 翻轉，仍以 pairwise feature 加總，這檔有 1012/1319＝76.72%、53.12 NFE。與 MC20 的配對差只有 +0.45 pp（95% CI [−1.77,+2.68]），卻多 13.0 NFE。它在論文主表之外。若改用自然的下一步觀測訓練殘差，這個公開變體須列對照。 [公開 flip 資料](https://github.com/TalOved1/measured-coupling/blob/main/outputs/llada_gsm8k__full__flip_rho0.05_b20.jsonl.gz) · [lagged/flip 程式](https://github.com/TalOved1/measured-coupling/blob/main/measured_coupling/llada/lagged.py)

## 2. 最接近的工作已經解決到哪裡

| 工作 | 已做到 | 對本題的實質限制 |
|---|---|---|
| [SetCommit](https://arxiv.org/html/2610.31120v1)，10/22 | 免訓練、沿用 host 提交量，依 entropy × 位置距離的 capped pairwise penalty 選 token，再用同 quota 的 remask 交換。兩模型四任務，在 32 NFE 有實際增益。LLaDA/GSM8K 的同狀態續跑有 240 prompts × 4 anchors，固定狀態、候選值、集合大小、後續策略和預算；對 confidence-matched control 增益只有 +1.86/+0.88/+0.07/+0.64 pp（進度 12.5/25/50/75%）。另外測 spacing、stratified random、shuffled risk、remask on/off。 | 「免訓練選集合」「固定 quota 的同狀態因果續跑」「飽和／capped pairwise penalty」「加 remask」已占位。尚可區分的是直接量聯合與成對的 signed margin 差，及其在 matched-latency 的獨立增益。Dream 原生 baseline 採抽樣，SetCommit 插件用 argmax；需加共用 proposals 的 selector-only 版本。 [ICLR #4399，已撤稿](https://openreview.net/forum?id=LxhKURNgpg) · [可重現程式](https://github.com/Kissmetothemoon/SetCommit) |
| [RPD](https://arxiv.org/html/2609.36452v1)，09/29 | 在 2,000 條 GSM8K-Aug 正解推理上控制 mask 數，按上游熵比較下游錯誤；再用跨層穩定度和前方剩餘 masked token 累積熵挑 commit。兩模型同有 block/full-canvas 與實測 TPS。 | 已涵蓋「上游未解使後面錯」「用廉價 gate 避免早鎖答案」「大 canvas 可保品質」。它用上游剩餘不確定度，沒有直接量合組後變動或高階殘差。需把 RPD 的現成規則當有競爭力的替代解釋和速度基線。 [公開程式及示例軌跡](https://dlm-rpd.github.io/) · [ICLR #22425](https://openreview.net/forum?id=ZVmtOl3rIt) |
| [DEMASK](https://arxiv.org/html/2604.02560v1#S5.SS4)、[PUNT](https://arxiv.org/abs/2510.21961) | DEMASK 學 pairwise TV 影響，以次可加假設控制集合取樣的 TV 誤差；Dream 上以 5,000 篇 × 5 次 mask、隨機集合大小 1–6 比聯合與成對，違反率 <6%，也已明寫可放鬆次可加條件。PUNT 直接用近似條件獨立測試拆相依 token。 | 不能聲稱首次量成對與集合差異、首次證明聯合誤差界。TV 次可加失效和有正負號的 logit margin 飽和是不同性質；小隨機集合與實際走到的大集合是可檢驗的區隔。 |
| [CoCommit](https://arxiv.org/html/2607.04469v2)、[PURL](https://arxiv.org/abs/2611.11551) | CoCommit 用訓練的 marker 加局部 forward 讓同組候選互通，在 LLaDA2.1-mini 近似 joint mode，並討論多回合擴充。PURL 的摘要描述聯合學 unmask 與 revision，該預印本已從 arXiv 撤下。 | 聯合預測已有可讀的正面結果。先以固定候選 token 值、固定不可逆寫入的控制實驗定位 selection 的增益。PURL 只列為曾提出的相鄰方向。 |
| [Mean-field](https://arxiv.org/abs/2606.15805)、[DAPD](https://proceedings.mlr.press/v306/kim26t.html)、[DOS](https://aclanthology.org/2026.findings-acl.861/)、[LoPA](https://proceedings.mlr.press/v306/hersche26a.html) | 成對相似度、注意力與依賴估計、近似 joint prediction 都已做過。DOS 已掃 block size，LoPA 用小 AR 模型做較一致的 joint 取樣。 | 定義簡單但夠強的 baselines，避免把普通 dependency-aware 或跨 block 的提升誤認成高階殘差的收益。 |
| [SPOT](https://arxiv.org/abs/2610.22543)、[SpecCTMC](https://arxiv.org/html/2611.08362v1) | SPOT 驗證提交順序以保留 target trajectory；SpecCTMC 保留參考 CTMC 的抽樣分布。後者在 B200 2,048 tokens 報 sequential eval 減 3.97 倍、depth-4 每步成本 4.02 倍，總體時間比約 0.99。 | 以分布精確性替代任務品質會擴大範圍。真正可用的效率要攤計所有探測與驗證時間。 |
| [Schedule-Calibrated Commitment Order](https://arxiv.org/html/2611.15234v1)，11/11 | 說明 sequential block 會灌高全序列 rank correlation。LLaDA 500 prompts，64→128 的 global Spearman 0.9845→0.9149，隨機區塊內的參考本來就從 0.9375→0.7500，校正後 within-block 0.751→0.660。缺原始逐題序列，無品質或 runtime 結果，平行 ties 應另處理。 | 若診斷涉及位置和時間，應在同一 block/progress 配對並直接記錄並列；不拿整串左到右的相關當依賴或因果。 |

**投稿狀態與文獻的效力。** 公開 OpenReview 可核實 SetCommit #4399 的 `venueid` 為 `Withdrawn_Submission`，沒有證據判定撤稿原因，arXiv v1 和程式仍公開，必須列基線。RPD #22425 為 active，11/14 晚 `[2,2,2]`，公開索引未找到可核實的 rebuttal 或個別理由。Measured Coupling 和 Schedule-Calibrated 的 ICLR 2027 精確標題查詢沒有結果，本文將兩篇當獨立的公開研究。SetCommit、RPD、Measured Coupling 的 arXiv 摘要均未標為 withdraw。 [SetCommit OpenReview](https://openreview.net/forum?id=LxhKURNgpg) · [RPD OpenReview](https://openreview.net/forum?id=ZVmtOl3rIt) · [SetCommit arXiv](https://arxiv.org/abs/2610.31120) · [RPD arXiv](https://arxiv.org/abs/2609.36452) · [Measured Coupling arXiv](https://arxiv.org/abs/2611.08307)

## 3. 第一個值得跑的實驗

**第一階段只問「知道真實聯合反應後，有沒有更好的同大小集合可選」，暫時不做新 sampler。** 使用公開的 LLaDA-8B-Instruct 及固定 256-token GSM8K 設定，把 block 32/64 的原始實作與 K=12、按比例 K=20/40 先對齊。把 75–100 道訓練題做程式和門檻開發，測試用固定 240 道 GSM8K 題。每題從原始大 block 的自然 rollout，按進度先後抽最多兩個含 8 個以上非 EOS 候選的位置集合，另留少量 2–7 的控制；同時記下全部路徑中符合資格的題數，用它把效果換回整批 prompt 的分母。對確定性的貪婪解碼，重跑 seed 只量時間，統計抽樣單位仍是 prompt。 [作者重現說明](https://github.com/TalOved1/measured-coupling#run) · [SetCommit 配對設計](https://arxiv.org/html/2610.31120v1#A1.SS2)

- **先分離三種誤差。** 在同一個 (x)、同一候選 argmax 和距離下，依序記錄融合近似成對探測、精確單 token probes 的加總、精確多 token 揭露的直接值。第一段差是 probe 沒測、stale 和快取近似；第二段才是非加成。用逐題叢集統計閾值翻轉、幅度、正負方向、EOS 與非 EOS、答案區與中間推理位置。即將 commit 的 token 使用 leave-one-out，下一步仍未 commit 的位置另外統計。
- **接著做同狀態、同 quota 的分叉。** (A) 原 selection；(B) 以 confidence、位置距離與集合大小匹配的簡單交換；(C) 精確 pairwise selection；(D) 固定少量候選集合（原集合及 2–3 個等大小交換集合），用精確 joint/leave-one-out 量測排序。B/C/D 共用候選池、argmax、時間位置和後續 decoder；加一組 shuffled residual。這是「在預先限定候選集合、以模型自身訊號重排序」的強診斷，不是全部可能集合的上界，更不是真值 oracle。事後才用任務評分，不能用正解選哪個集合。
- **用兩個讀數決定因果與實用性。** 診斷階段允許記錄額外 probe，不把它算成後續 NFE，固定後續規則及剩餘 forward cap，並另記實際使用量。記錄預先固定 anchor 的 paired final accuracy／可執行測試、掉分救回／反向損失以及每題真實能介入幾次。確認階段讓重排或校正真的介入整段解碼，同時計入全部探測與 kernel/host 開銷。回報 wall-clock 秒/答案、有效長度、實際 forwards 和 peak memory。單次干預效果不直接相加當作多步累積效果。
- **過關才做廉價校正。** 先試 sign-aware saturation/size calibration（固定額外查詢數 0），再試把 K 個 probes 中少數換成 joint rows，只針對滿足 ≥8 且靠近門檻的集合。也可用原本就會執行的下一步學殘差，但要和公開 `flip/lagged` 比，且只能使用當下可取得訊號。對照同 block 掃 τ、同時間調 K、改小 block、SetCommit selection-only 和 full、RPD，最後用 Dream 或 MBPP 驗證跨設定方向。若廉價公式只是 SetCommit 的 capped pairwise penalty 改寫，缺乏區隔。

### 預先決定的停損與綠燈

| 判準 | 如何處理 |
|---|---|
| **第一道：找得到可定位且可介入的失敗。** 預設 240 題，只看固定測試與非 EOS；如果精確聯合重排比精確成對或 quota-matched 交換連方向都不穩、介入覆蓋率太低，或把 τ/K 重新校正後差距就消失，停。若觀察到 ≥4 pp 整批 prompt 的配對改善，且主要落在可重現的聯合殘差 strata，再跑確認。 | 4 pp 約是目前 block64 與公開 K12/MC20 差的一半。若 240 題估計區間很寬且方向、機制未明，列不確定，可擴一次到完整 1,319 題或獨立任務；不把不顯著視為沒效。 |
| **第二道：保留 probe 成本後仍值得做。** 於盲測中，相對同台 GPU 上重調參的成對方法、SetCommit 和 RPD 的最佳品質–時間前緣，至少取得同時間 **+2 pp**（配對 95% CI 下界 >0），或同品質平均每題延遲 **−15%**（品質非劣 margin 1 pp，明示 CI）。在第二個模型或任務保留方向，也驗證改善並非只改 EOS 或輸出長度。 | 兩道都過才考慮當主題；只有位置級預測較準、只在調參集或特定極端 budget 有效，就不上升。 |

依公開 20/64 比較的 309/1319 配對不一致率粗估，240 題的 95% 區間半寬約 6 pp；這個樣本多半只能篩掉明顯沒效果的想法，無法確認 2 pp。確認時應以 prompt-cluster 方式配對、在獨立資料統計，必要時報告小效應無法分辨。原始 8.57 pp 只適合作為要解釋的量級，不能當校正可兌現的收益。

## 4. 實作成本與排序

- **已核對的資源。** Measured Coupling 是 MIT，提供 LLaDA/Dream 評估、設定、融合 kernel 和輸出。論文在單 A100 80GB、BF16、batch 1 測，LLaDA 權重 14.9 GB，K12 融合 probe 多約 0.14–0.26 GB。現有完整單 token probe 可作離線起點，測聯合集合仍需寫 adapter；GitHub 逐題輸出不包含可直接重播的步級 logits。SetCommit 為 MIT，有 pinned LLaDA checkpoint 與 Native/Fast/EB selector、fake host smoke，GitHub 釋出的 tarball 沒找到完整同狀態分叉逐題資料與 Dream 端到端重現說明，故它是可跑的 LLaDA 基線，Dream 整合待核。RPD 有釋出入口和示例軌跡。本次只讀程式與解析輸出，未安裝 CUDA 環境或跑 smoke。 [Measured Coupling 程式](https://github.com/TalOved1/measured-coupling) · [SetCommit 重現說明](https://github.com/Kissmetothemoon/SetCommit/blob/main/REPRODUCE.md) · [RPD 作者網站](https://dlm-rpd.github.io/)
- **先用單卡小樣本量成本。** 原作者 LLaDA GSM8K block 20 約 5.86 秒／答案（40.1 NFE、A100）。用每題 8–12 條完整或部分續跑作保守基準，240 題約 3–5 GPU-h，另預留精確 singleton、leave-one-out、不同 block 與編譯/時間重複。建議先在 20 題量實際成本，再決定 240 題 cap；若成本跨過約 10 GPU-h，先縮 anchor 或候選集合。數字是按公開端到端時間估算，H200、不同 prompt 長度和多 probe kernel 時間尚未實測。 [原文 Table 1、附錄 A/D](https://arxiv.org/html/2611.08307v1)
- **研究排序。** 已停損的「把資訊曲線接到真模型、固定排程」沒有因本調查重新成立。這個新候選有自己可測的局部誤差及大 block 掉分，足以支撐一次可快速否決的試驗。SetCommit 已在同狀態下把大部分看似可歸因的改善縮到約 0–2 pp，且 RPD 的上游不確定度、EOS 提早提交等替代解釋都很強，故現階段列備案，優先級低於已找到清楚識別問題且可以低成本跑完驗證的主題。只有「大集合的聯合殘差修正跨模型穩定、並改善完整品質–時間前緣」才有足夠獨立性。 [SetCommit 附錄 F.4](https://arxiv.org/html/2610.31120v1#A6.SS4) · [RPD §4](https://arxiv.org/html/2609.36452v1#S4)
