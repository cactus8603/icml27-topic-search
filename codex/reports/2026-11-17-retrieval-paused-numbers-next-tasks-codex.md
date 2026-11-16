# [codex] Retrieval 暫停交接：已確認數字與接續任務

[codex] 2026-11-17 00:22 起（台北時間）。使用者要求「先等等，把數字和接續任務記錄起來，想要統整 repo」；目標已由使用者清除。本文件記錄暫停點，不授權恢復實驗、搬遷、推送或新 GPU 工作。

## 已採取的暫停動作

- 全部子代理已收到停筆指示；不啟動新測試、下載、分析或 GPU。
- 唯一中央 `central-v4b` 的父程序已用 pidfd 精確暫停（SIGSTOP），停止後續派工。沒有 signal 既有 worker、模型程序或程序群組；已在跑的工作可以自行完成。
- 中央 PID `2921085`，start_ticks `272000661`，boot_id `f34b259e-fe82-4dd1-ad7e-82b9e76bf226`。暫停者仍持有中央 singleton lock；不可另開第二個排程器繞過。
- 中央來源：`icml27-topic-search@f6780742ec62d7e50773054ea747c3692fce43dc`；schedule `/ssd4/cactus8603/icml/codex/retrieval_matrix_dispatch_batches/central-v4b.json`，SHA256 `1638788fee9d62abafe9cf3bc6f4ba9e7b2a17b255885451701c1102de02ae8c`。
- 持久暫停快照：`/ssd4/cactus8603/icml/dl/retrieval-matrix-dispatch/pause-20261117-0022.json`，SHA256 `6ae57a3dd33409f26f695c0b43c31d32d73b6336a5f78d1c242eb3dab0d57ede`。含完整 state、父程序 argv/token 與原因。
- 已向中樞 `killtests/SUBMITTED.log` 登記 hold-parent；恢復須使用者明示，再由 root 重核程序 token、來源路徑、worker receipts／leases。不要直接按舊數字 PID 恢復。

## 已確認的正式 P0 數字

以下沿用已完成機械判讀與非實作者獨立重算的 CORE2 結果，百分比四捨五入至小數六位；本交接沒有重新讀取或計算其他模型效果。

| CORE2 embedding 原生評估 | 實測 % | 論文目標 % | 差異 pp | 固定 1 pp 規則 |
|---|---:|---:|---:|---|
| COCO text→image R@5 | 79.255617 | 79.3 | -0.044383 | 通過 |
| Flickr text→image R@5 | 96.120000 | 96.2 | -0.080000 | 通過 |
| NegBench COCO retrieval R@5 | 76.581115 | 76.6 | -0.018885 | 通過 |

- 原生三資料集皆在門檻內；真 `qualification.make → admit` 已通過。資格檔 `/ssd4/cactus8603/icml/dl/retrieval-matrix-analysis/results/fixed-embedding-core2/qualification.json`。
- 獨立重算涵蓋原生正反向七個 IR 欄位，與原判讀差距 ≤ `1e-12`。獨立審查 SHA256 `c5b312f3ea2332a7bcae146df8f6b84bfc7ade6f1ce59afbccb72077bca33693`。
- 全部 producer 位元組保留，來源 `icml27-topic-search@99431594893bba94337b3befd10a7a483ba7374f`；分析來源 `icml27-topic-search@7b4855f5d1b7f6e6c1f3850b8a2a9e41b5ac82ed`。

| CORE2 reranker 原生 MCQ | 實測 % | 論文目標 % | 差異 pp | 固定 1 pp 規則 |
|---|---:|---:|---:|---|
| COLA | 72.380952 | 72.4 | -0.019048 | 通過 |
| NegBench MCQ | 64.562007 | 64.6 | -0.037993 | 通過 |
| SugarCrepe++ | 83.818721 | 83.9 | -0.081279 | 通過 |

- 完整 `15,912` 例／`58,891` pairs／`9,425` 圖／八個 subsets，逐題 strict/native FP32 與 macro 均已獨立核對；三資料集通過。
- 資格檔 `/ssd4/cactus8603/icml/dl/retrieval-matrix-analysis/results/mcq-core2/p3-reranker-qualification.json`；獨立審查 SHA256 `1ddf6e2d34c6ac8eea58c037180cc06a890967cc28645fa5862457590d590e71`。
- Producer 來源 `icml27-topic-search@25f7bd515875a0c698202d0047d7b7952fd6846b`。

完整既有數字與判讀界限：[CORE2 獨立重算報告](/ssd4/cactus8603/icml/codex/reports/2026-11-16-core2-independent-numeric-review-codex.md)。兩項均為固定資格規則的機械通過；使用者的研究判斷仍為 `PENDING`，不能寫成方法已成功或論文已定案。

## 暫停時 GPU 與佇列

暫停快照共 34 個任務：4 completed、4 running、12 pending、10 failed、4 cancelled。父程序停住後 state 不會持續 reconcile；下表是暫停時狀態，稍後應以原 worker 的終態收據核對。

| 當時執行中任務 | 卡 | worker PID / start_ticks | child PID / start_ticks | 原來源 |
|---|---:|---|---|---|
| `mcq-core8` | 3 | 2972508 / 272085656 | 2972531 / 272085678 | `icml27-topic-search@25f7bd515875a0c698202d0047d7b7952fd6846b` |
| `native-paired-clic_b32-einops-v1` | 0 | 3017994 / 272133385 | 3018033 / 272133406 | `icml27-topic-search@5675ef59cee99fa6d28e12f07501eeb157c321bd` |
| `native-paired-clic_l14-einops-v1` | 1 | 3115095 / 272233335 | 3115132 / 272233357 | `icml27-topic-search@5675ef59cee99fa6d28e12f07501eeb157c321bd` |
| `native-rerank-jina_m0` | 4 | 2960975 / 272068480 | 2960998 / 272068501 | `icml27-topic-search@54f544ce54bca82be3cc8b3c72662b354c6d3e40` |

上述均用同一 boot_id；完整 UUID、argv、原收據、token 在暫停快照。全局最多四個 GPU 工作、每卡最多一個我們的工作。容量足夠可與一個既有 compute PID 共用，不要求完全空卡；兩次相隔五秒及取得 lease 後均須重查。不可多隊列或讓三四個人競爭同卡。

已完成：`fixed-embedding-core2`、`fixed-embedding-gme2`、`mcq-core2`、`native-siglip2-einops-v1`。GME2／SigLIP2 的 completed 目前只代表 producer 完整執行，沒有在本交接宣稱資格通過。

仍在排隊、暫停後不得自動啟動：

- fixed embedding：`core8`、`qwen8`、`vlm2vec2`。
- fixed MCMR：`core2`、`core8`、`gme2`、`qwen2`、`qwen8`、`vlm2vec2`。
- native paired：`clic_clips_l14`。
- native reranker：`qwen2`、`qwen8`。

舊 failed／cancelled 的歷史全部保留，不會因統整 repo 自動清空或重跑。`fixed-embedding-qwen2` 是既定 synthetic batch-invariance gate 未過，不可放寬門檻後自動重試；其他既知 API／依賴問題的修復另有新來源與新輸出。

## 資料與已接受設定

- 主要面板、公開評估資料、影像及已列必要權重均已落地盤點；早期 `dataset_download_audit` 的缺 COLA／Flickr／SugarCrepe++ 結論已過時，不能再當現況。
- 已同意：沿用 COCO train2017 dev；COLA 官方公開版只有評估資料，僅跨資料集固定測試，不能假造 train 或拿它選方法。
- COCO dev-v2：20,000 圖，正式 conjunction 1,089 題／dev-small 252；negation 237／small 197，只作診斷。與已曝光 test／PACO／Karpathy／Half-Truth 圖片聯集 31,570 張的交集為零，獨立 audit 已有。
- COLA：GQA 320 題／2,589 圖；PACO 400 題／7,921 圖。GQA released gallery 與論文 1,952 圖有已披露差異；MCQ 210 例。
- SugarCrepe++：4,757 例；SugarCrepe：7,511 例。Half-Truths 1,437 例（509＋454＋474）；FACTOR-Bench 1,695 例。
- 原 CORE COCO 5,000 圖／25,014 texts；Google SigLIP2 是 5,000／25,010。CORE HF Flickr captions 也與 raw Karpathy 有差異，不可混用。
- MCMR：3,997 queries、104,981 product IDs、103,176 張被引用影像；123,561 張有效影像已落地。保留 product IDs，不因共用圖去重；一個未用零位元 archive 已披露。原生 renderer 與未釋出的 CORE converter 不同，僅 difference-only，不能算嚴格 P0 資格。
- **新發現尚待補取的特定協定資產**：Triplet 論文明引 LAION CLIP_benchmark 的 COCO／Flickr annotations，目前只有 release metadata，實際兩個 annotations 尚未取得（約 4,116,303 與 398,044 bytes）。這不影響已完成 CORE／Google 資料，但 Triplet 原生資格不能直接代用現有 caption order。
- K=100 為主；50／200 為補充。自訂方法只用 dev／dev-small；固定官方基線可用測試集。

## 已完成準備、尚未執行的 P2

新 repaired MLLM 接線七檔已經 32 CPU 測試與獨立來源審查通過，凍結於 `icml27-topic-search@f69b876213f98d1aa53788a62ecb6c67a9f233e0`。只接受明列 `9943159` 修復後兩個家族，完整 P0 producer ref／spec／settings／source 表／三資料集 actual processor／runtime 均精確綁定，不修改舊 producer。

真正 CORE2 候選已建立：

- `/ssd4/cactus8603/icml/codex/retrieval_matrix_panel_runtime_next/p2-core2.review.v1.json`，SHA256 `33ca7c2b88da48027335e1973cdf83d302b1cab09671273199453187a7dfcd5a`。
- 狀態仍 `REVIEW_REQUIRED`／`reviewed=false`；fresh 真 P0 evidence 審查已完成 PASS，讀取日誌正常 `READ_COMPLETED`。
- [獨立候選審查報告](/ssd4/cactus8603/icml/codex/reports/2026-11-17-core2-panel-plan-independent-review-codex.md)，SHA256 `785d9017096bb3c62e67e41a452a2944579029ceb9cef054a1695d7168048b92`。
- **沒有 approved plan、沒有排程、沒有新 P2 模型效果。** 恢復後才可另建 approved，僅變 `state`／`reviewed`，保留原候選，來源與 plan 一起 commit 後才進下一個 additive 中央批次。
- P2 輸出規劃四資料集 conjunction-dev／negation-dev／cola-gqa／cola-paco；擬新輸出 `dl/retrieval-matrix-panel-runtime-next/runs/p2-v1-20261117/core2`，尚未執行。
- CLIP P2 adapter 也已凍結於 `icml27-topic-search@d5aa2cb592d0a48ebac260973302b0e68a3514a3`，但仍須真 SigLIP／paired P0 資格才能建正式計畫。

## 暫停時仍有的 CPU 工作

1. Root 在暫停前啟動的 GME2 原機械判讀 session `79347`，已在交接保存期間自行結束（exit0，`COMPLETE`），沒有新啟動分析。固定入口 `/ssd4/cactus8603/icml/codex/retrieval_matrix_analysis/run.sh mechanical`，job `fixed-embedding-gme2`，來源 `7b4855f5d1b7f6e6c1f3850b8a2a9e41b5ac82ed`。結果位置 `dl/retrieval-matrix-analysis/results/fixed-embedding-gme2/`；原 ReadSession 已由既有程式正常結束。Root 未另打開或摘取新報告中的模型數字。**接著仍須 fresh 獨立重算，尚無正式 GME2 資格。**
2. Native-analysis 原 36-test CPU 全套 session `79247`，host PID `3103330`／PPID `3103329`，交接時 30 項 ok，沒有全套結論。持久 log／完成時 receipt：`dl/retrieval-matrix-native-analysis/cpu-full-preflight.v1.{log,json}`。此為 source／synthetic 測試，沒有讀真 native effects。

其他代理均已結束本輪，沒有新下載或各自 GPU 隊列。剩餘既有 CPU 測試可自行結束，不把局部測試當最終 PASS；恢復時先核完成狀態，不直接重跑。

## 工作中草稿的保存點

### Native analysis

`/ssd4/cactus8603/icml/codex/retrieval_matrix_native_analysis/` 已有七個原生 producer jobs 的完整 prospective manifests／機械→獨立→原 Q.make/admit 管線、各家原始 CPU 環境與 array binary、私有讀取日誌、角色排除與 strict 逐題核對。仍待獨立 prelaunch review 與 approved freeze，**不得讀真模型效果**。

- Draft `plan.native-analysis.draft.v1.json`，SHA256 `6bf340b40b26a1525bd4cbdd14fddd766b332b405b34b9637b92ad45387e2fe3`，`REVIEW_REQUIRED`。
- [作者暫停交接與完整 source hash manifest](/ssd4/cactus8603/icml/codex/reports/2026-11-17-native-analysis-paused-handoff-codex.md)，SHA256 `23ff2c78922523a7aeb2c039f788e55d26701b6c0edfea9fd35bee0a5dcc9f9c`。
- 已知原 facade 17 tests PASS；前輪 contracts/workflow/pipeline 17 PASS；三類真 CPU import／forged-binary rejection PASS。最後全套尚待核收。
- 真數值審查需排除 root／matrix_runner／public_data／official_p0／p3_contract_review／native_full_fixtures 及 mechanical actor。fresh `independent_numeric_review` 未參與實作，可於使用者恢復後承接。
- qwen_native2/8、Qwen8 Instruct 尚缺兩個精確 targets，不 eligible；native-reranker scope 不能直接送進舊 P3 gate。既有 retrieval producer 沒完整 GPU calls／peak，不能造零成本。

### Triplet 原生 P0

`/ssd4/cactus8603/icml/codex/retrieval_matrix_triplet_native_p0/` 目前只有八個程式檔：contracts／build_plans／acquire_audit／metrics／preflight／qualification／runner／central_task。**未測、未完成 docs／PREREG／三份 draft，不能執行。**

- LAION source 固定 `a23028288ed914efb31c5ff32a1e702123cbd0ff`，六檔 122,049 bytes，在 `dl/retrieval-matrix-triplet-native-p0/sources`；這是投稿前時間代理，並非 Triplet 作者自釘 commit。
- 三個 variant 必須各自保留：CC12M Triplet、OpenCLIP finetune、作者 CC12M NegCLIP，不能換成原版 NegCLIP。
- 論文 Table5 與 Table13 相同數字的 I2T／T2I 表頭互換；registry 目前忠實抄 Table13，不可擅自重標為 conventional 方向。
- 論文引用 LAION，但實際 AMP／batch 命令未報；作者 requirements 是 open_clip 2.26.1／torch 2.4.0，既有 extra wrapper 硬驗 2.31.0。只能共用已核權重，不能把不同 wrapper 當作者原生協定。
- 待續：補 tests／README／PREREG→產三份 `SOURCE_REVIEW_REQUIRED`、`executable=false` 草稿→CPU audit→root 統一取得 annotations 並核 order→非作者審來源缺口→再決定明示 adaptation 與正式資格邊界。

## 使用者恢復後的接續順序

1. 先核既有 CPU 測試的完成收據、GME2 已完成機械判讀及讀取日誌完整性、四個 GPU worker 原收據；不因父程序停住而重送完成工作。
2. 確認 repo 統整後的正本位置與執行路徑。現有 frozen bytes、完整 SHA 與絕對路徑仍指中樞，不重寫歷史。
3. CORE2 P2：保留已審候選→另建 approved（只改兩欄）→freeze→CPU source preflight；P0 evidence 的每次重讀先註冊；P2 真效果另需獨立、事前凍結的分析計畫。
4. 新中央批次可用既有 immutable NEXT builder，保留原 34 個 task 字典與歷史；新 P2 ID／output 插入 MCMR 前。先核全部來源／plan／新 schedule freeze，登記 repo@commit，再以 root exact-parent-only handoff 或核准的恢復程序接手；保留四卡上限與原 leases。暫停期間不做。
5. GME2：原機械完成後 fresh 獨立重算→真 Q.make/admit；其他已完成 P0 同樣走原固定流程。
6. Native-analysis：核收最終 CPU suite→非作者 prelaunch source review→另建 approved freeze→SigLIP2／paired／native MCQ 原機械→fresh 數值重算→原生資格。沒有資格不接 P2。
7. 補 Triplet 三 variant；再補原版 NegCLIP、GME7、UniME-V2、CS-CLIP 各自原生 P0 協定。原 UniME Qwen2 公開 checkpoint 身分未建立，禁止用 V2／Phi／LLaVA 偷換；CS-CLIP 多個擾動是否真不同 benchmark 須先核，不能湊兩個資料集。
8. LCSE 仍要依原作者的 operator-pair calibration 協定實作，與既有 train-presence calibrated／paper-parameter transfer 版分列；不能把既有 AND adaptation 宣稱作者協定重現。
9. 完成 Claude 報告64全矩陣 P0→P1→P2→P3→P4：dev 選兩個 first stages／reranker 組合、K100 主50/200輔、各資料集分列及真 calls／成本、通過者才進 P4。仍有 P2一般檢索／MCMR／paired分支與新 P3 scope adapter 未接；目前工作遠未全部完成。

## Repo 統整所依賴的現況

中樞是 `icml27-topic-search`。既有 compositional-retrieval、`codex/retrieval*` 與後續 paper 工作目前仍以中樞為正本；Claude 已建立中樞 `@b6c17ad` 的唯讀 topic 鏡像，尚未 cutover。原 `HOLDS.md` 請求指出待執行中工作及 launcher 不再依賴中樞後由使用者定案；本次使用者正在統整，實際搬遷仍依其接續指示。

目前仍被依賴的內容包括 `codex/retrieval*`、`dl/` 的資料／權重／env／輸出、`hf_home`、原 schedule/runtime、中央 singleton／GPU lock 路徑。這份交接及 WIP 保存不變更它們、不做 git push、不處理無關 `runs/`。後續搬遷要保留 frozen 來源與必要相容路徑，再用新提交記錄新入口。

本次僅將交接、候選與原本未提交的 native-analysis／Triplet 程式保存為本機 WIP checkpoint；提交不代表 draft 已獨立通過或可執行，兩個工作目錄的待辦仍照上文保留。尚未建立任何新 approved plan。
