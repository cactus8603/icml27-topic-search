# [codex] Native reranker MCQ P0 啟動前獨立審查

[codex] 2026-11-16 22:44（台北）。審查者：`/root/official_p0/p3_contract_review`，與實作者分開。本次只讀 source、官方來源與 CPU 技術收據，並執行合成 CPU tests；沒有載入正式權重、讀取真實模型效果、執行 GPU 或排程作業。

結論：目前程式與 CPU 技術檢查未見未解阻斷；這不是模型數值資格 PASS。草稿仍為 `REVIEW_REQUIRED`、`reviewed=false`、六個 `verified_protocol=false`，須另經協定核准、Git 凍結及中央指派。Qwen official-native 2B／8B 與通用 True/False 適配仍缺少兩個 exact MCQ targets，現有 gate 在模型初始化前拒絕其 P0 GPU 執行，不能借用 Swift 或 MCMR 異協定數字。

審查固定版本：[plan.native-mcq.review.v1.json](/ssd4/cactus8603/icml/codex/retrieval_matrix_rerank_native_p0/plan.native-mcq.review.v1.json)，SHA256 `40ad9abd8c0a7a19d2ac018a38c6bf4187dcfa69ee182c03c8c8b1dcd9a84c29`。已獨立執行 `runner.validate`、核對全部 19 個 source hashes，並實際 import 全 9 個模組，逐一比對 path、version 與 source SHA；CUDA 未初始化。CPU processor/template [probe](/ssd4/cactus8603/icml/dl/retrieval-matrix-rerank-native-p0/cpu-probe.json) SHA256 `8ddf9a40efc74058e9597ac974bca7e0da83862fa6883ca1c6ced1a18e6a51af`，與計畫精確綁定，含 Jina 實際 processor class/config、image grid 及 reward token100。

已核對 Swift 的 pair batch32、Jina 每題 COLA2+2／SCPP3／Neg4，以及 native／pointwise 逐 pair；storage32題保留原呼叫邊界。Jina COLA 為 image-choice，CORE／Qwen 為 text-choice，另一軸只能作 diagnostic。三個完整 benchmark 與 strict tie-fail／宏平均規則、15,912題／58,891pairs／9,425圖均固定；Qwen2／8 Swift 與 Jina 具來源 targets，仍無本機數值 PASS。歷史 checkpoint／Swift build 無法完全追溯的限制已明示。

原 raw scalar／content／branch logprob 保留，FP64 array 僅為容器；raw transaction 先保存，checkpoint 後提交，中斷恢復不重跑已存呼叫。先前指出的資料來源可重綁與 array state 證據不足已修復：資料引用及分母與 protocols 精確相同，完整 state fingerprint 綁定 producer identity／items／batch32，逐段核 start／end／payload hash。獨立資格審查重開原 arrays/raw、核完整 model／variant／source freeze 與實際 NumPy path/version/SHA，另行向量化重算；數值符合先記 `NUMERIC_MATCH`，非產生者審查後才可形成資格。失敗／未完成呼叫的 forward 預算明列為上界，沒有宣稱已完成。

本審查者於最終 source 版本獨立重跑 **14 項 CPU tests 全過**（11.381秒），涵蓋完整合成分母、原批次、raw 中斷恢復、state 身分與邊界、缺 targets、Qwen silent-null 拒絕及 True／False 原兩次 forward。另以 pinned Jina remote class 執行 12,369參數隨機 BF16 CPU text forward，Transformers4.57.3 下 `mm_token_type_ids=None` 與明傳零值兩路相容且逐 bit 相同；未載入正式權重、CUDA false。

實作者的 source-stable [CPU preflight](/ssd4/cactus8603/icml/dl/retrieval-matrix-rerank-native-p0/cpu-preflight.json) SHA256 `28f24651b08112d68ff3b7312eacc91cccb0a4448dfadcdb7111659f53ef8e0d` 已讀並核對計畫／source hashes；該收據記錄 14tests、AST／shell syntax、全部9,425圖 bytes／SHA／decode PASS，以及六 variants 原 pairs／order 相同。完整逐圖驗證為實作者執行，本審查者沒有另重跑逐圖掃描；沒有將其視為模型重現或 GPU 授權。
