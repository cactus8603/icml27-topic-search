# [codex] P3 官方重排接線獨立預檢

[codex] 2026-11-16 22:00（台北）。審查者：`official_p0/p3_contract_review`，實作者：`official_p0`。結論：**程式與 CPU 合成驗收 PASS，當前無剩餘程式阻斷。**真實第一階段 P0/P2、重排 MCQ P0 與選模證據尚未齊備，因此沒有 production plan、沒有 P3 GPU 資格；本報告不把未執行的官方重現或模型效果當作通過。

範圍為 `codex/retrieval_matrix_rerank_p3/`、其 frozen backend 與 P2/native 資格介面。只讀程式及 frozen plan metadata，未讀真實 P0/P2/P3 效果，未執行 GPU，未修改接線程式、未 commit。

## 已核行為

- 僅接受兩個不同第一階段：一個 MLLM、一個 CLIP；development 選模收據綁各自 conjunction-dev report 及原 index-meta SHA，明確排除 test 選模。
- MLLM 和 CLIP 使用不同 P2 phase；CLIP 固定原生 qualification validator 及其 data/metrics/evidence 四個來源，完整 spec digest 不得更換。P2 provenance 要有完整 producing commit；plan 及全 source SHA 逐一對 producing Git blobs，不要求今日已演進的 P2 原碼等於舊版。
- 第一階段需 COMPLETE index-meta、完全相同 provenance／model／dataset／settings；score 前重驗完整圖片 SHA、原向量、order 及收據。全庫 order 必須是完整 permutation，不能傳入只存 top100 的 MCMR cache。
- 入選固定原 full order 前 200；不讀 label/ignore/hard_list 篩選或補候選。以同一份原分數計 K50/100/200，K100 為主效果；同分沿 embedding 原序，prefix 外 tail 完全保留。ignore 只在 CPU 評分處理。
- CORE 首批重排資格保留三個 benchmark、8 subsets、15,912 rows／58,891 pairs；核原 FP32 arrays/state、分母與 subset macro，以及完整 model spec、MCQ source freeze 和非產生者獨立重算收據。至少兩個不同 benchmark 在 1pp 內，第三項失敗仍保留。
- exact instruction、released backend source、原 runtime、完整 reranker 集合再次於 execution gate 綁定；實際 pair batch 4，storage chunk 64。原 content list／scalar 無重縮放，以 float64 容器無損保存 Python 原 scalar。
- 每 4 對先原子保存 immutable raw transaction；array/state 每 64 對提交。resume 先驗原 raw 的 pair IDs、identity、backend SHA 和 score/content 一致性，已保存 raw 不重跑模型。calls journal 先記 STARTED，再記完成／失敗及 pair 次數、時間與 peak；invocation 也記起訖。硬殺造成的未知時間／peak明示，不能當零成本。
- GPU 入口要求 root 中央 dispatcher 的 owner/task/可見設備與恰好一張 GPU。只有已資格齊全、獨立審查、commit 及乾淨來源的正式 plan 才能執行 synthetic GPU smoke；CPU 分析分開。

## 發現與獨立驗收

初稿在 storage64 的 raw 寫入與 checkpoint state 提交之間中斷，resume 會重新推論，遇數值不同即因 immutable raw 衝突而無法恢復。已改為上述每 4 對 raw transaction，沒有修改 frozen checkpoint/backend。

最後修版 **14 CPU tests PASS**，涵蓋 label-free admission、非法／partial order、stable ties/tail、batch4/storage64、完整 MCQ 分母與非產生者 gate、raw/state 中斷恢復、失敗呼叫留存、smoke 身分／數值、producing Git blobs 及 CLIP 原生分流。

另以獨立 synthetic backend 重跑初稿同一中斷點，故意讓每次 model call 產不同 scalar：首個 state 未提交時已有 64 對 raw；resume 無重算這 64 對，整批恰好 **50 次 batch4／200 distinct pairs**，前 64 原 scalar 精確保留。再次讀 COMPLETE cache 為 **0 次新模型呼叫**。CUDA 未初始化，fixture 已移除。

native 資格上游在審查期間同步硬化；已重新以 current bytes 和實際 native CPU runtime 核 producer／validator schema：9 項完整 module dict 與固定依賴一致（含 distribution_version），本審查已實跑 native 12 CPU tests PASS。其後 native 最終硬化另由 root 獨立審查。先前依舊節錄提出的 schema 阻斷已撤回，沒有尚待修正的上游 runtime blocker。

所有成本只可稱每題實際 200 對的共享 cache 成本。K50/100 是配對 prefix 效果，沒有獨立服務延遲測量。真實 P0 重現、第一階段選定、正式 P2 index 與 production plan 仍須後續完整證據及獨立審查，不由本次 CPU PASS 推定。
