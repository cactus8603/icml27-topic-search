# [codex] Central retrieval dispatcher

此協定只改排程，不改任何模型參數、模板、權重、資料、輸出路徑或原 frozen launcher command。先由 root 停止確認 child=None 的八個舊等待 queue，逐筆登記；本程式不會取消舊 queue 或其他人的程序。MCMR 可在首次執行前加入同一份 tasks，仍共用所有限制。

- 一個 singleton scheduler：`dl/retrieval3090/runs/.gpu-locks/central-dispatch.lock`。每GPU沿用同目錄 `GPU-UUID.lock`，同時最多一個我們的 job。
- 全局同時最多兩個 live launcher，包括 starting/uncertain。所有子代理停止自行送GPU。
- 2B 的 core2/qwen2/gme2/vlm2vec2/MCQ-core2 最低空間8192MiB；8B最低20480MiB。沒有 utilization 門檻，也不要求空卡。
- 加入我們之前，GPU最多一個不同的既有 compute PID；已有兩個或更多便不進入。首個合格 snapshot 後至少五秒，再次合格才嘗試鎖；取得鎖後再查一次空間與 PID 數。
- 使用 `resources.py` 的 FileLock/query_gpus/atomic_json。Worker 和原 launcher 繼承GPU鎖FD；central意外結束時仍執行的job不釋放lease。
- state/events保存launch、exit、失敗與原命令。失敗、取消或不確定的attempt均不自動重送。worker/child以boot-id、PID和start-ticks辨識，避免PID重用。重啟依現有attempt監看，不能把仍執行的job當pending。
- 只讀 non-metric COMPLETE receipts確認先前是否已完成，並核 model/freeze_commit/plan SHA；不打開研究判準結果。不重複完整輸出，部分輸出交原producer既有續算處理。
- root先短claim共享log，預登記central tasks及scheduler PID後commit/release。實際啟動既有task時，以單次O_APPEND新增`[codex] ... start ...`；平時細節寫自身append-only journal，不長期持有LOCKS。
- SIGTERM/SIGINT只停止該dispatcher帶身分token的worker，worker只通知自己啟動的launcher process group；不根據nvidia PID清單殺任何程序。非正常消失保留failed/uncertain，不自動重試。

啟動前root須把schedule設FROZEN，獨立review、commit全部本目錄source/protocol/schedule。CLI還會逐檔核目前執行bytes與該commit，並核每個原plan/launcher SHA。模型producer自己的原freeze gate仍有效。

```bash
python3 -B codex/retrieval_matrix_dispatch/build_schedule.py
python3 -B -m unittest codex.retrieval_matrix_dispatch.test_dispatch -v
# root完成來源freeze並確認舊queue都退休後：
python3 -B codex/retrieval_matrix_dispatch/dispatch.py run \
  --schedule codex/retrieval_matrix_dispatch/schedule.json \
  --freeze-commit FULL_DISPATCH_COMMIT \
  --runtime dl/retrieval-matrix-dispatch/central-v1
```

CPU測試只使用合成GPU snapshots、假launcher與獨立臨時檔，不查真GPU，不寫共享log，不啟動模型。
