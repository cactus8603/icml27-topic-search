# [codex] 使用者四卡上限：root 獨立 prelaunch review

2026-11-16 23:29（台北）。使用者明示「最多改成4張卡工作」。此為來源與 CPU 審查，尚未執行 handoff，不讀主判準 arrays、metrics 或判定報告。

root 非 next dispatcher 實作者，核對原 cd02391 與新五檔 diff、完整 controller/build_schedule/handoff 路徑及測試。新 schedule 必須 max_live=4，實際 live_count 不縮放；adopted/running/starting/uncertain 均占位，每 GPU 至多一個本方工作。原 v3/base 來源未修改。舊 v3 handoff 仍按其原 max_live=2 驗證，不因新上限放寬原快照；新 next 可保留四個不同 GPU 的既有工作。

root 重新執行 26 next、10 v3、10 base 共 46 CPU synthetic tests，全部 PASS（1.579 秒）。獨立 AST 比較亦 PASS：移除說明、新 limit 賦值，把兩處 limit 還原為 2 並還原私有 B global 限定後，tick 與 frozen base 完全相同。容量兩次間隔五秒、最多一個既有 compute PID、lease 後重查、STARTING 在 Popen 前持久化、spawn 後不確定時保鎖流程均保留；達四不查詢／啟動第五個。

完整 handoff 只操作配置所指 exact parent，pidfd 加 PID/start_ticks/boot_id/argv 再驗證；原 worker/model 不收 signal。原 receipt、config、最初 source、每個 worker FD 的同 inode 與 flock 均需通過。退休前失敗恢復原 parent；退休後不自動重複 replacement spawn。正式新 schedule、七個 explicit additions 與配置尚須另凍結和 preflight；本報告不把未執行的切換寫成成功。

允許將已審 source/PREREG 凍結，接著建立及核對含原 27 tasks 與七個已凍結修復／原生任務的單一新 schedule。舊 task dictionary、相對順序與失敗歷史保留。SigLIP2 的 12GiB 例外只適用完整原 FP32 model spec；其他工作保留 8/20GiB。新排程不改 producer 科學設定、不自動 retry failed 工作，不以此賦予任何 P0 資格。

來源 SHA256 同實作者交付表：controller f55362906d39c11a7dd6d91a938a5a7533cda1f2f1fb30143bcb2c1982d5a044；builder bd59276fcb996c171b0a81601f3fdfc1c7f1fb93955b72372e474b3d593cbff9；handoff 5ea076fb1bd82a7393e2e02831cdb81eb25053bc749e9f3031a726f2b35dedbf；tests 8234b4a34396456c0ef2c356918e45e0ec36672e583d33386071d1e5d75cb8f5；PREREG aa284731afa9e89a306fcb8c7c0c0b83554f3f5ceb6604f50caf887a4ce8fc55。
