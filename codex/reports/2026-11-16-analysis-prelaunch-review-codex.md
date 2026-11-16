# [codex] Retrieval matrix analysis 獨立 prelaunch review

[codex] 2026-11-16 23:34（台北）  
Reviewer：`/root/independent_numeric_review`

結論：**PASS，僅限下述候選版本的 CPU 分析流程凍結前審查**。本輪未讀取任何真模型分數、arrays、metrics、reports、aggregate 或 verdict，未執行真 mechanical／independent launcher，未操作 GPU／排程。這不是模型 P0 資格判定，也不是使用者定案。

我沒有實作本 repo 的模型、評估、分析程式或這次修復；僅讀 source、執行合成測試、回報阻擋並複核實作者的修復。我沒有修改分析實作。

## 審查版本

候選計畫：`codex/retrieval_matrix_analysis/plan.analysis.review.v1.json`  
SHA256：`94c468ccb2b7d5f5916925af91acb786af4a6281a1df6963488ff9190d386ce3`

候選仍為 `state=REVIEW_REQUIRED`、`reviewed=false`。本審查允許 root 在其餘內容不變下，將這兩欄改為 `FROZEN`、`true`，再 commit 凍結；任何來源、角色、計畫內容或判準更動均需重新審查。真結果讀取只能在凍結後另外獲得明確指派才開始。

審查當下的來源 SHA256：

| `codex/retrieval_matrix_analysis/` 檔案 | SHA256 |
|---|---|
| `build_plan.py` | `79d46776d8a035c9ef999daa82687c682ccdef426dc59b5c3f7ae23fb8cf37a4` |
| `contracts.py` | `16baf9710417a76557ee334e8f2020228763119f7ae54c79aef17ff1ca996c8d` |
| `launcher.py` | `c836e65773a0d8f650d01a6579518c052ff91ef19e7376f17052cc4d8bf99bcd` |
| `reading.py` | `49777bf1a59a27da26a2b1fdab00b48657a103d041d947e07fcf9e71e186a509` |
| `recompute.py` | `09118fdc93cfd07fc179ffbae5900237c3135fa8d9abf3651f987aa2ba73f857` |
| `test_analysis.py` | `aced85e3b83cedcbe60663b2724d22f54625a9401cc8316b76d2275691b88aa4` |
| `run.sh` | `dc7767742b652ca2ba01a70dce545e43bee2070f1119218cc1586c0c22d21962` |
| `PREREG.md`（尚無讀取事件） | `ec398c21da3d58c199ab53334118b477c963ac3501ae74119ca4b02c851573c5` |
| `README.md` | `a911688cc524b7b2b10fafaddac78607e12fcb150a67f6000617b42174721e48` |

## 獨立執行與核查

以 `dl/retrieval-matrix-rerank/env/bin/python -B`、明確空白 `CUDA_VISIBLE_DEVICES`、`PYTHONDONTWRITEBYTECODE=1` 執行：

```text
python -B -m unittest -v codex.retrieval_matrix_analysis.test_analysis
Ran 26 tests in 22.252s
OK
```

另已完成所有分析 `.py` 的 AST parse 與 `bash -n codex/retrieval_matrix_analysis/run.sh`。

我直接對最終候選呼叫 `C.validate_plan` 與 `C.independent_environment`，並攔截 `C.read` 限定僅能讀 `codex/` 下的來源 JSON：共 15 次 source JSON 讀取，14 個 job、來源 SHA、原計畫／commit／model／output 對映，以及實際匯入的 NumPy 路徑、版本、distribution version 和 array binary SHA 均通過。沒有讀取真 runtime 結果 JSON。正式 `PREREG.md` marker 後的 tail 為空，事件數為 0。

26 項測試包含實際呼叫既有 `panel.qualification.make/admit` 與 `P3.qualify_reranker` 的**合成**橋接；不是只比較自訂 fixture 的布林 PASS。原生 IR 的額外合成測試使用凍結環境內真正的 `pytrec_eval`，覆蓋大量同分與超過 10 個 relevant documents。

## 原先阻擋與修復複核

初版候選 `155cc3defaa85a4456df73dc61ef7a629573d2e333c28648410c715cb5cd77c7` 的 22 項測試雖通過，但讀取 log 僅驗內部 hash chain。我以獨立 temporary PREREG、mocked Git bytes 重現：寫入 START／COMPLETED 後刪除整段 tail，`source_gate` 仍通過。這不符合預先登記的不可刪改讀取歷史要求，故當時阻擋凍結。

實作者修復後，我逐行核對新的外部 append-only journal：GENESIS 綁 PREREG path、body、plan SHA 與 source commit；PREPARE 綁 exact event 及前後完整 bytes／SHA／事件數／head，先 durable 寫入並 fsync，再 append／fsync PREREG，最後 durable COMMIT；新 journal 與目錄項亦 fsync。pending、部分寫入、缺少既有 anchor 或雙邊不一致均拒絕，沒有自動重設或採納舊 prefix 的分支。

我另外重跑自己原先的重現程式，全部使用 `hf_home/tmp` 內的專屬 anchor，確認真正的 `source_gate` 現已拒絕：

- 刪除全部讀取事件。
- 刪除最後一個完成事件。
- 保留完整 PREREG，單獨將外部 journal rollback 至較早版本。

恢復完整有效合成歷史後 `source_gate` 通過，保留兩筆事件。新版 suite 另通過 PREPARE 後／log append 後 crash 的 fail-closed 測試、讀取 body 不會進入的測試、缺少 anchor 與 directory fsync 順序測試。此阻擋已解除。

## 數值、來源與角色邊界

- 14 個 job 限定為 2 個原 CORE MCQ、6 個修復後 embedding、6 個修復後 MCMR。MCQ 固定原 commit `25f7bd515875a0c698202d0047d7b7952fd6846b`，修復批固定 `99431594893bba94337b3befd10a7a483ba7374f`。完整 plan／model spec、settings、source SHA、輸出路徑與原 central-v3 schedule 逐 job 對映；原失敗版本不混入。
- MCQ 保留 15,912 題、58,891 pairs、8 subsets、3 benchmarks。COLA 為原 text-choice 比較方向；SCPP 為 5 subsets 等權 macro；NegBench 為 COCO／VOC 等權 macro；所有 ties fail。不以 pooled row-weighted macro、去重或換資料集改寫分母。至少 2 個 distinct benchmarks 在固定絕對誤差 .01 內的規則未變，3 個 benchmark 都保留。
- Embedding 保留 COCO／NegBench／Flickr 以及空 caption。只用原 BF16 GPU score caches 與原 `topk` admission 重算官方欄位，不用 secondary FP32 full-gallery ranking 資格化模型。checkpoint identity、原 items／batch_size fingerprint、每個原 batch bytes SHA、完整 array/state receipt 均重算／核對；不把新的 CPU dot product 當作原 GPU inference。
- IR 同分採 score descending、document ID 反字典序，I2T recall 使用 relevant-caption fraction。AP@10 以全部 relevant documents 為分母。兩方向的 7 個原 IR 欄位均重算。MCMR MRR 先保留原 native first10 admission，再按 evaluator 的 score／document ID 排序。
- MCMR 固定 104,981 個 distinct product IDs 與 3,997 queries，保留各 product 自己的完整 metadata，不按重複 image 合併。官方 MMEB renderer 與未公開 CORE converter 的差異明示；receipt scope 固定 difference-only，`eligible_for_P0_qualification=false`，合成 panel gate 亦拒絕將其當 P0。
- Reviewer 排除 producing implementations、analysis authors、central producer actor 與 mechanical actor，並須明確接受獨立角色。共享 producer/backend 作者 `/root`、`/root/matrix_runner` 在 embedding 與 MCMR 排除列表內；MCMR 額外排除 `/root/public_data`；Qwen2 兩類額外排除 runtime-fix 作者 `/root/official_p0/p3_contract_review`；分析作者 `/root/official_p0` 亦被排除。角色／ref／path 被改動的合成拒絕測試通過。
- 流程保持 original mechanical → 非實作者獨立重算 → 使用者定案。original mechanical report 不可覆寫；部分失敗保留並拒絕默默重試。獨立重算前必須有同一 plan／commit 的成功 mechanical receipt 與已完成 reading event。數值 agreement tolerance 固定 `1e-12`；agreement 僅代表重算相符，最終使用者決策仍為 `PENDING`。

以上是 source 與合成正確性結論；尚未對任何真模型輸出做數值審查。
