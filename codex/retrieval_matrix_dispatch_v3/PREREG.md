# [codex] Central v3: retain live workers during reviewed technical repairs

2026-11-16。原中央v2已實際執行；CORE2重排MCQ正常推論，部分embedding/MCMR在合成smoke遇bare-decorator API或缺einops而停止，沒有該批benchmark結果。v3保留每個原作業的命令、ref、output、所有terminal歷史，不清空失敗狀態、不自動重試。另加已獨立審查且commit的技術修復作業，使用新ID、新plan/ref、新輸出。

舊controller只在host以exact PID/start/boot/cmdline核對後短暫停止其父程序。快照保存所有state／attempt／worker與child token；必須證明原running worker實際存活且持有原GPU lease，才能只退休舊中央父程序。不得signal模型worker或child，不因移交重新啟動它們。新中央重核舊schedule SHA、原argv/輸出/收據身分，接收其動態完成收據並將它們計入global cap2、占用原卡位，GPU鎖持續由原worker/child繼承。若snapshot不明、known-bug pending項已啟動或lease無法證明，恢復舊中央並延後移交。

原未啟動而確定受同一decorator錯誤影響的CORE/Qwen8 embedding/MCMR pending項可在移交時退役；已started一律不能退役。原failed、completed、cancelled、uncertain狀態保留；不把uncertain當可重跑。接手收據若token/task/schedule不符直接拒絕；原worker和child消失但無terminal收據則保留failed，不自動重試。

仍為同時最多2個我們的job、每卡1個，加入前既有compute PID最多1個，5秒兩次記憶體確認、拿鎖後再查，不要求utilization為0。既有8GiB/20GiB類別保留。SigLIP2獨立新增12GiB資源門檻：完整FP32權重約4.55GB，固定384/image4/text32，留足餘量；原20GiB只是未實測前過度保守的排程值，沒有改模型、資料、batch或precision。GPU smoke保存實際峰值，失敗不得降低設定／自動重試。

controller與worker均核本目錄所有source/PREREG SHA及frozen Git bytes；原dispatch來源不變。新worker仍使用原lease、argv、checkpoint與非指標completion收據。只由root單一中央派工；所有送件／退役／接手依COORDINATION登記。CPU controller測試只用合成收據與假process/query，不開CUDA或benchmark主結果。

中央收到停止訊號時，只能停止本次controller instance實際spawn、仍有Popen handle且PID/start/boot token相符的worker；採納的worker或重啟前留下的worker不在此集合，不能被此停止路徑signal。父程序移交獨立核對並退休exact中央parent，不能以停止worker流程代替。
