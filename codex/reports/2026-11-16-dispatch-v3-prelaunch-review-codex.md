# [codex] Central v3 adoption：獨立 prelaunch review

狀態：此次 source bytes 的 CPU／靜態 prelaunch 審查通過；真正移交仍由 root 完成 host 證據與 frozen source gate。審查者為非實作者 Codex matrix_runner；只讀 controller／builder／worker gate 與 CPU 合成測試，未啟動 GPU、未讀模型效果、未操作或 signal 任何現行程序。

已獨立重跑 `PYTHONDONTWRITEBYTECODE=1 python3 -B -m unittest codex.retrieval_matrix_dispatch_v3.test_controller`：10 tests PASS，0.171 秒。含最初 worker receipt 的 schedule/runtime 在多次 handoff 後保留、失敗與 terminal 歷史不重設、只退役已知 pending、採納 job 與本代 job 同算 cap2，以及採納 worker 原 GPU 仍被占用。

adoption 逐 SHA 核原 schedule，保留每個原 task 的 argv、freeze、output與狀態歷史。採納 active row 仍指向原 receipt／token／原 schedule SHA，不會改成新 schedule 身分。reconcile 拒絕錯誤 token/task/schedule；既有 uncertain 仍占位，worker與child都消失而缺 terminal 證據時 failed 且不重試。新 worker 先核 v3 source gate，再進 base worker gate驗同一實體 GPU lock FD，保留兩次容量查詢及拿鎖後重查。來源 controller source／PREREG 與原 dispatcher 都需 frozen Git bytes相符。

初讀發現的 SIGTERM 範圍問題已修正並複核：v3 自訂 `stop_own` 先 reconcile，再只迭代本次 controller 的 `self.children`；必須仍為 LIVE、worker PID 等於實際 Popen PID，且 base `alive` 核對原 start/boot 身分才 signal。採納或 controller 重啟前的 worker 沒有本代 Popen handle，因此不進停止集合。新增測試證明採納 PID100 不被 signal，本代 PID200 才收到 SIGTERM，收據變成 PID201 時拒絕 signal。此停止路徑未掃 GPU PID 或其他人的程序。

父 controller-only移交仍須依 PREREG，以 host exact PID/start/boot/cmdline 與原 worker/child token及GPU lease證明進行；不能以一般 SIGTERM 走舊 controller 的 stop_own。此處沒有代替 root 執行任何移交，也沒有讀 GPU 作業主結果。

本次審查綁定以下 SHA256；若此四檔再修改，須重新檢查 freeze table 與相關測試。

| 檔案 | SHA256 |
|---|---|
| controller.py | e73c69f279a5addb650d43163e30d240cd06936e5a8fb2694a135f0f390dc8ae |
| build_schedule.py | 9ab559f84700f7a431f87c775c44bbc7a8b722bd975cc0a3a15acb20670ad324 |
| test_controller.py | 7c61e5234d345881fe7feca7524d1034eae8c5dcd44c570b3a1adaed6871a395 |
| PREREG.md | c1219ce1e2bd26129a8df96b4aa56cc7ed14de1de0aeb86221a32b23a69a598a |
