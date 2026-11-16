# [codex] P3 兩個第一階段與官方重排接線

本模組只準備CPU可審程式，未執行GPU或讀P3效果。使用凍結 `codex/retrieval_matrix_rerank/backends.py`，不修改MCQ來源、模型、raw score或prompt。首批CORE2/8；Qwen／Jina／通用MLLM的MCQ與P3適配須另開正式批。

`contracts.py` 固定完整P2 order→top200 pairs→共用prefix50/100/200的接線，候選介面不接受labels或ignore。`runner.py` 分smoke、score、analyze；每pair原query／image SHA／ID、raw response及原scalar都保留。storage64內實際backend batch4，每4對先原子保存raw transaction；中斷後先驗raw，直接恢復array，不重跑已保存的模型呼叫。cache scalar用float64容器無損保存原值。calls.jsonl逐呼叫先記STARTED，再記完成／失敗、原pair數、時間與GPU peak；invocations.json亦先登開始，保留中斷／resume。硬殺時尚未完成的時間與peak可能未知，會明示。實際每題200 distinct pairs；K100是主效果，不能稱共用cache時間為K100獨立服務延遲。

第一階段一個MLLM、一個CLIP。每個reference有 `family`、P2 `plan:{path,sha256}`、`model`、以及 `indexes:{dataset:{meta:{path,sha256},identity:{path,sha256}}}`。只接受已qualified的P2完整index，另有selection收據：`state=FROZEN,scope=development-only,test_results_used=false,criterion`，`first_stages`逐模型綁P2 plan SHA/model/family，`development_reports`各綁真正conjunction-dev報告及同一index meta SHA。MLLM只接受phase=P2-qualified-fixed-baselines並重用P2完整P0資格驗證；CLIP只接受phase=P2-qualified-fixed-clip-baselines，兩資料集官方資格需提供另行獨立審查的 `qualification_adapter:{path,sha256}` （固定codex/retrieval_matrix_native_p0/qualification.py）Python檔，公開介面 `admit(receipt_entry,model_spec)` 回傳PASS及完整model_spec_sha256；該檔與其data/metrics/evidence依賴亦須加入P3 sourcefreeze。P2 producing commit須為40字元，plan bytes與完整source SHA必須等於該commit的Git blobs，index-meta須COMPLETE並精確綁provenance；推論前重驗全部image SHA。沒有這個實際validator／收據時builder拒絕，不能用假CLIP或第二個MLLM代替。

`reranker_qualification`每個model有 `model`、`plan`、`complete`、`report`、`review`，後四者均為實際檔案path/SHA。MCQ review scope固定 `official-reranker-mcq`，須有不同producer/reviewer、independent_recomputation及audited_arrays，綁model/plan/complete/report hashes。三個benchmark與8 subsets完整列出，15912 rows／58891 pairs完整arrays/state逐SHA核驗，至少兩個benchmark在1pp內，完整backend spec不能改。CORE原MCQ report雖無獨立identity欄，會由complete.identity與原plan、report hash及非產生者收據共同重綁。

輸入JSON頂層為 `first_stages,selection,reranker_qualification`；builder只讀真實證據，不建立placeholder或假資格plan。實際P0與P2證據尚未齊備，因此本次沒有產生production plan。

```bash
CUDA_VISIBLE_DEVICES='' dl/retrieval-matrix-rerank/env/bin/python -B -m unittest discover \
  -s codex/retrieval_matrix_rerank_p3 -p 'test_*.py'
dl/retrieval-matrix-rerank/env/bin/python -B codex/retrieval_matrix_rerank_p3/build_plan.py \
  --inputs REAL_EVIDENCE.json --out REVIEWABLE_PLAN.json
```

計畫初始 `REVIEW_REQUIRED`，root獨立核後設FROZEN/reviewed=true並commit；全40字元commit、乾淨工作樹與所有資格檢查通過，且由root中央排程賦予owner/task與恰一張GPU後，才可用 `run.sh COMMIT PLAN MODEL FIRST_STAGE DATASET OUT`。輸出限定 `dl/retrieval-matrix-rerank-p3/runs/`。GPU只產raw score，CPU分析另明確執行。MCMR native只存top100，不符合這個完整order/K200介面，不能誤接本批。
