# [codex] Immutable additive central batches and exact-parent-only handoff

本模組是下一批單一中央排程的準備；開發只用 CPU 合成資料與假 Host。不得由子代理執行 inspect 的 host 外部升權、execute、recover、GPU 或送件。只有 root 在 source／PREREG／schedule 已獨立審查並 commit、shared log 短 claim、host exact 身分與 lease 證明俱全時，才可明確執行。此目錄尚未凍結前不產正式 schedule 或操作現行程序。

builder 只接受 root 明列的已凍結 plan/launcher/full commit/output/completion/minimum memory；逐檔 git-show 核 bytes。舊 schedule 每個 task dictionary 完全不變且相對順序保留，不取消、改參數或重跑舊工作。新任務使用全新 ID／output，預設插在第一個 MCMR 前，可在凍結前明列另一原 task anchor。舊 failed/cancelled/completed/uncertain/attempt/token/原 receipt 歷史全保留，不以新批次清空。

使用者於本批凍結／執行前明示「最多改成4張卡工作」。新schedule明設max_live=4，next Dispatcher.tick明確複製原admission流程，只把兩處global cap2判斷改為真實live_count對schedule.max_live；不是縮放計數或多個queues。舊base schema只核normalize副本max_live=2，真正執行與state仍是4。原live v3與base來源不改。

其餘容量沿用 base dispatcher：每卡1個我們的 job、原8GiB與20GiB類別、既有 compute PID 最多1、連續兩次間隔5秒與取 lease 後重查、best-fit。原 inherited12GiB task 必須字典完全不變；新12GiB只有固定 google/siglip2-so400m-patch14-384@e8e487298228002f3d8a82e0cd5c8ea9c567f57f 的原完整 FP32 image4/text32 payload，額外將新 plan SHA、model spec SHA及 task digest 全綁入例外表。任何 OOM／失敗保留，不改 batch 或自動 retry。

private base dispatcher 保留 frozen worker、容量確認、lease FD、STARTING 在 Popen 前 atomic 保存與 Popen 後失敗窗口。只重用 frozen v3 seed/reconcile/stop_own；不修改 canonical base 或 v3 的 validate/global。新 controller／worker 都核本模組、v3 與 base source freeze。adopted 與新 workers 同計 cap4，原卡持續占位；收據核原 token/task/schedule，multi-handoff 保留最初 runtime/schedule。停止只處理本代 exact Popen workers，不 signal 採納者。

完成檢查只讀 non-metric producer receipts。一般模型用 COMPLETE+identity.model/ref/plan；native reranker 用 COMPLETE+identity.variant/variant_spec/ref/plan+15912例/58891pairs/9425圖。只允許已獨立核 protocol 的 qwen2、qwen8、jina_m0，至少兩個 verified published targets；不讀 metrics 或造資格。

handoff 配置明列 old/new schedule SHA、source ref、runtime、controller、python 與 expected PID/start/boot token。舊 parent exact cmd/token 核對後只 SIGSTOP 該 parent；每次 signal 都先取得 pidfd 再重核 token/cmdline，用 pidfd_send_signal 防數字 PID 再利用，沒有 os.kill fallback。穩定 state 必須有相同 scheduler token，LIVE必須 running，依舊schedule凍結上限核原v3最多2個、新next最多4個，GPU各異；不以新cap4放寬舊v3的cap2快照。逐一核原 receipt/config/task/token、最初 source schedule/ref、worker/child argv、worker FD 與同一 shared GPU lock inode及 flock。只有上述再次通過才 SIGKILL 已停的 exact parent，從不 signal worker/model/group或掃 GPU 程序。retire 前失敗 SIGCONT 同一原 parent。

已驗身分的 terminal receipt 若與 snapshot 競速，只在原 model child 已死時保留終態；不把原工作重送。retire 後若仍有不明 running/terminal race、lease消失或 singleton 不空，留下完整 previous state/migration/source/error，停止等待 root。明確 recover 僅可在原 parent 已退休、沒有任何新 Popen 嘗試與新 state 時重核原 receipts再啟動中央，不能重跑任何 producer。replacement Popen 一旦嘗試即保存 PID/token/argv；不得自動再 spawn。新 singleton controller 自己 reconcile 舊工作，才依同 cap4派新任務。

所有 parent hold/retire/release、新中央 submit 以 [codex] 台北時間 O_APPEND 登記；保留原 runtime/events與每份 receipt snapshot。root 需先 short claim shared log，並在執行前逐一登記新增 task 的 ID/ref/output；不能以長 lock 阻擋其他代理。handoff 不是 scheduler 自動功能，也不是 failed producer 的 retry。

CPU 測試為 `PYTHONDONTWRITEBYTECODE=1 python3 -B -m unittest codex.retrieval_matrix_dispatch_next.test_next`。合成 receipts/commands/processes 位於 hf_home/tmp，唯有 FD/flock 測試使用測試程序自己的 CPU FD 與私有暫存 lock；不接觸實際 GPU locks或信號。

root review 後可先複製及更新 additions.candidate.json，填入所有已獨立核准的新 runtime refs（原 paired54f 因缺 einops 已明確排除）。builder 介面：`python3 -B codex/retrieval_matrix_dispatch_next/build_schedule.py --old-schedule OLD --old-freeze FULL_REF --old-runtime OLD_RUNTIME --additions EXPLICIT_JSON --out NEW_JSON [--before OLD_TASK_ID]`。輸出仍須 root review/commit 才能 preflight/run。handoff.example.json 的 pid0、新 schedule SHA及ref均為無效 placeholder；必須先換成 host 真證據，不能直接執行。預設 handoff `inspect --config FILE` 只做來源核驗；execute/recover 另需 root 的明確 parent-only CLI flag。本次沒有生成正式 schedule 或執行這些操作。
