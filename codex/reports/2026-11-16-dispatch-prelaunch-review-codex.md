# [codex] 中央GPU排程啟動前核對

[codex] 2026-11-16 21:29（台北）。依使用者新指示：放得下即可用非空卡，同時避免三、四個代理搶資源。只調度資源，不改模型／資料／模板／數值精度／評分／原凍結命令。

root為排程器非實作者，獨立重跑10項標準庫CPU合成測試通過，並核對singleton、global cap2、每GPU共用鎖、PID/start-time/boot-id身分、兩次5秒確認及鎖後複查、末刻容量變化、failed不重試及worker繼承GPU鎖。取最小可容納的卡，保留20GiB卡給8B；2B需至少8GiB、8B20GiB。加入前同卡最多一個既有compute PID，不進已塞兩個以上程序的卡。utilization不是准入門檻。

八個舊等待佇列均先核queue身分、`waiting`與`child=None`後才SIGTERM；全部回覆cancelled且無模型子程序，逐筆立即登SUBMITTED.log。沒有取消任何其他程序或已進行的模型推論。

中央清冊保留八個原始凍結launcher命令，另加入六個已獨立審查的MCMR native v2，合計14個模型作業共用同一上限。每個原producer的source gate、smoke與完整資料驗收仍有效。MCMR以`65f4f4850a82adf283b34dfddc319b72657562e6`執行，其非strict CORE renderer差異不因排程而消失。

程序開始／結束寫入自身append-only journal及身分收據；預登記的task開始時另單行O_APPEND寫共享log，不長期占用LOCKS。來源／清冊必須commit且本目錄乾淨後才啟動。worker另重核凍結清冊與繼承鎖，排程器意外離開不釋放仍執行子工作的鎖；失敗、取消、身分不確定均不自動重送。SIGTERM只通知該排程器已驗明的自家worker／launcher group。

此核對不產生模型研究效果，也不把排程啟動或CPU測試當P0分數通過。後續P2仍待真實論文重現與獨立結果證據。
