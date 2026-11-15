# [codex] 檢索成本獨立審查

2026-11-16（台北）。審查對象為 `benchmark-platform-torch214-20261116` 的完整成本報告及凍結來源，並非方法 AP／盲審結果。本審查未執行 GPU、未重啟作業、未改方法、環境或原實驗程式。

**結論：可報告本機同步元件耗時與批次吞吐；同為 50 calls 的所選政策和 CORE50 成本相近，本輪不足以主張可靠加速。沒有量得 fresh-query 端到端延遲、冷啟動延遲或 peak VRAM。**

## 完整性與範圍

- 完整 `report.json` 首次讀取：**2026-11-16 06:03:37**。1694/1694 個預定項目齊全：16 次 CPU joint search、98 政策 × 8 queries × 2 repetitions 的 1568 次 admission、96 次 fresh text encoding、14 次 reranking。共實算 **10,400 pairs**。
- report 的 identity、measurements、load events 與 checkpoint 逐項相同；progress 為 complete。每項時間有限且非負；每臂 pairs、batch4、8-query 順序與固定 hash 排程一致。沒有把其他政策的探索成本當作所選政策的線上成本。
- 固定 query indices 為 `[134, 6, 82, 96, 111, 31, 12, 61]`；每個只重複兩次。這是同一小組 query 的描述性成本，沒有估計部署工作負載的獨立信賴區間。
- 成本政策與 `selected-policy.json` 的 LOCKED 紀錄一致：`m1/top200/fuse_logsigmoid_cs_t1_w0.75`。原 published reproduction 保留 MISMATCH；平台比較仍是 development，非正式證據。
- 執行來源 commit `1f9caa2b10b2864689592393409decd4092c9cd7`；已驗 `benchmark.py` bytes 與該 commit 相同。硬體為 RTX 3090，PyTorch 2.14.0+cu126、CUDA 12.6、cuDNN 91002；OMP／OpenBLAS／MKL 各 1 thread，Torch interop 設定為 48。

## 重排元件：完整機械表

每個 repetition 均把 8 queries 全域以 batch4 打包，可跨 query 邊界，並包含 RGB 解碼、處理器與模型呼叫及 CUDA 同步等待。下列「攤提秒/query」為 `(rep0 秒 + rep1 秒) / 16`，**不是單查詢延遲**；pairs/s 為兩次合併 pairs 除以合併秒數。

| 臂 | 每次 pairs | rep0 秒 | rep1 秒 | 批次攤提秒/query | pairs/s |
| --- | ---: | ---: | ---: | ---: | ---: |
| 固定 25 calls | 200 | 16.066 | 16.638 | 2.044 | 12.231 |
| 固定 50 calls／CORE50 | 400 | 33.790 | 35.214 | 4.313 | 11.594 |
| 固定 75 calls | 600 | 53.758 | 50.658 | 6.526 | 11.492 |
| 固定 100 calls | 800 | 67.707 | 68.326 | 8.502 | 11.762 |
| 固定 150 calls | 1200 | 99.955 | 101.807 | 12.610 | 11.895 |
| 固定 200 calls | 1600 | 134.596 | 133.919 | 16.782 | 11.917 |
| 鎖定政策 50 calls | 400 | 32.968 | 35.144 | 4.257 | 11.745 |

所選政策／CORE50 的此元件時間比為 **0.9871**，差約 **−1.29%**。兩次 repetition 的時間範圍重疊，query 順序及圖像／文字 padding 也不同；可說同預算下量到的成本接近，不能由這兩次測量宣稱穩定加速。較大預算的比較只支持這組 queries 上的 reranking 元件成本，不是完整系統速度。

## CPU、文字編碼與模型載入

下表皆為 16 次（8 queries × 2）的中位數；範圍是觀測最小至最大，**不是 95% CI**。

| CPU 項目 | 中位數 ms | 觀測範圍 ms | 所選政策是否需要 |
| --- | ---: | ---: | --- |
| CORE full dot + 全圖庫 z-score | 10.119 | 8.651–12.895 | 需要 full 搜尋；z-score 是綁在同項的額外保守成本 |
| CORE 穩定全圖庫排序 | 0.798 | 0.614–1.170 | 需要 |
| CS-CLIP 兩條件 dot + 全圖庫 z-score | 5.994 | 5.350–7.357 | 需要 |
| 所選政策評分、融合與選 50 | 0.534 | 0.489–0.782 | 需要 |
| CORE 基準選 50 helper | 0.185 | 0.158–0.325 | 基準對照；非額外加到所選政策 |

CPU joint-search 中位數為 **87.488 ms**，同時量兩個 backbone 的 full、兩條件及 LCSE mean-dot 特徵；所選政策不需要 CORE condition、CS full 或 LCSE 的額外搜尋。不能把 87.488 ms 與必要子項再相加，也不能把整份 joint-search 都算成單一政策成本。上述子項及其各自中位數不是一次串接執行的端到端實測值。

| Fresh text encoding 元件 | 每次實際文字數 | 中位數 ms | 觀測範圍 ms |
| --- | ---: | ---: | ---: |
| CORE full query | 1 | 46.365 | 42.754–59.952 |
| CORE two conditions | 2 | 53.576 | 42.898–72.187 |
| CORE four templates／condition | 8 | 57.000 | 47.444–162.200 |
| CS-CLIP full query | 1 | 11.099 | 9.098–15.198 |
| CS-CLIP two conditions | 2 | 12.472 | 9.784–13.772 |
| CS-CLIP four templates／condition | 8 | 12.006 | 10.541–21.342 |

`batch_size=32` 是上限，並非每次處理 32 queries。fresh encoder 輸出沒有接回後續 admission；admission 使用的是固定 cached vectors。四模板計時沒有顯式包含 8 向量形成 2 個均值向量的後處理，不能將其說成完整 fresh LCSE 查詢成本。所選政策使用 CS 單模板條件分數，不需四模板這一路徑；條件詞向量可快取的收益限於此固定條件表的設定。

| 載入／前置元件 | 秒 | 另記 warmup 秒 |
| --- | ---: | ---: |
| 當次 preflight 與輸入載入 | 9.091 | — |
| CORE encoder 新實例 | 10.961 | 0.745 |
| CS-CLIP 新實例 | 4.355 | 0.089 |
| CORE reranker 新實例 | 4.559 | 0.734 |

模型載入含當次建立實例及裝置搬移，檔案系統 cache 不受控，**不等於磁碟冷啟動**。這些 load events 不能不加說明就按每個 query 收取或完全忽略；是否常駐、切換模型及攤提方式取決於部署。

## 儲存與未量測部分

| 報告的原始 array payload | bytes | MiB |
| --- | ---: | ---: |
| CORE image 向量 | 67,100,672 | 63.992 |
| CS-CLIP image 向量 | 16,775,168 | 15.998 |
| 兩 backbone 的 image 向量 | 83,875,840 | 79.990 |
| cached full-query 向量 | 1,392,640 | 1.328 |
| cached single-condition 向量 | 778,240 | 0.742 |
| 六個 cache arrays 合計 | 86,046,720 | 82.061 |

`indexed_storage_bytes` 是上述 arrays 的 **RAM payload nbytes**，包含 query cache，並非完整系統儲存。相對只用 CORE image index，所選政策另需 CS-CLIP image 向量的 15.998 MiB，模型權重另計。另以 file stat／SHA-only 核對：`vectors.npz` 真實檔案為 86,048,320 bytes（82.062 MiB）；本次一併載入、但未列入該欄的 `lcse_vectors.npz` 為 3,903,682 bytes，校準 JSON 為 13,857 bytes。所選政策不需這兩項 LCSE 額外產物；全部數字均未包含模型權重、原始 JPEG、cross lookup、其他 metadata、重複 checkpoints 或完整建庫成本。

benchmark 雖 reset peak-memory stats，但沒有輸出 peak VRAM，不能據此給峰值數字。它也沒有量得一次串接的 fresh encoding → 搜尋／admission → reranking → 最終排序及回傳流程；影像編碼／建索引成本不在 online 元件內。不能直接外推 H200／叢集、MCMR 商品長圖文、多使用者負載或模型常駐部署的延遲。

## 額外的候選忠實性檢查

針對「每 query 的兩條件矩陣乘法」與 analysis「整張條件表乘法」可能有浮點邊界差，06:01:13 執行一次獲准的 CPU-only 核對。使用同一平台六個 vectors、僅解析 `panel.json` 最前面的 `condition_index`，沿原 analysis features／policy／admit 產生所選政策候選；**8/8 queries 的有序 50 indices SHA，與 16/16 個成本 checkpoint admissions 及成本 helper 重算全部相同**，集合差皆為 0。此證據限固定 8 queries；沒有讀取或計算 AP／relevance、沒有改浮點規則。

`cross.npz` 只讀 NPY headers：`core_order` 為 `(136,8191), int32`，`cross` 為 `(136,200), float64`，沒有讀 cross scores。沒有開啟 blind/private、sealed 或本輪方法 AP 報告。

## 建議總報告使用的說法

> 在固定 8 條開發查詢、兩次重複及跨查詢 batch4 下，所選 50-call 政策的含解碼重排元件為批次攤提 4.257 秒/query，CORE50 為 4.313 秒/query，成本相近。CPU 政策選候選中位數為 0.534 ms；搜尋、fresh 文字編碼、模型載入及儲存另列。這是本機元件成本測量，未量得端到端延遲或可靠加速。

來源定位：`benchmark.py:90` 同步計時；`:137` joint CPU features；`:239` 全域 batch4／解碼；`:406` CPU policy admission；`:427` fresh encoding／load；`:476` 最終報告及儲存／邊界。

## 讀取與可追溯紀錄

- 05:52:10 起讀成本 checkpoint；05:53:57 讀 queue 與 selection lock 元資料；06:01:13 完成獲准的 label-free 候選核對；06:03:37 首次讀完整成本 report。稍後只對輸入檔做 stat／SHA 核對。
- 成本報告：`dl/retrieval3090/runs/benchmark-platform-torch214-20261116/report.json`；SHA `d242f95e6d80ae3165ef24cc9822588365c58467c3b4b08c30056715203f934a`。
- 最終 checkpoint SHA：`c0847fdf0ff6924870b25077c36fcd96671ca5fb023d5723dac7654d89e6979b`。
- 凍結 `benchmark.py` SHA：`2a53579a02320e4e185b9d0e8b9b225998bc63d8d601bd2c0339327063e7c75c`。
- queue：`dl/retrieval3090/queues/benchmark-platform-torch214-20261116.json`；selection：`dl/retrieval3090/runs/analysis-platform-torch214-20261116/selected-policy.json`（僅 lock／policy／來源欄位）。
- label-free 向量來源：`dl/retrieval3090/runs/platform-torch214-20261116/vectors.npz`；另只解析同目錄 panel 的 condition-index 前綴與 cross headers。補充檔僅 stat／SHA：`dl/retrieval3090/runs/supplement-torch214-20261116/{lcse_vectors.npz,lcse_calibration.json}`。
