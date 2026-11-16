# [codex] next 中央排程 cap4 更新：CPU 交付

依使用者最新明示「最多改成4張卡工作」，root授權在尚未live的next來源做新freeze。本次只改 `codex/retrieval_matrix_dispatch_next/` 五檔，未改原live v3/base，也未建立正式schedule、讀live state或效果、查GPU、送件、啟動或signal任何真程序。

new schedule嚴格要求max_live=4；builder亦明設4，不繼承舊v3的2。private validate只把深複製的schema-check副本改2以復用舊base驗證；實際schedule/state/live_count從未縮放。next Dispatcher.tick明確複製原admission method，僅兩處global上限改為實際live_count對schedule.max_live；不以多個queues或計數技巧增加上限。

5秒兩次容量確認、取lease後重查、既有compute PID最多1、8/12/20GiB已核門檻、best-fit、每卡1個我們的工作、完成skip、原failed不retry、STARTING在Popen前atomic保存及Popen後uncertain保鎖分支均保留。新tick與frozen base tick經AST比較：僅移除說明docstring、新limit賦值，把limit還原2及B私有global限定還原後，AST完全相同（True）；沒有其他control-flow變更。

handoff capture依原schedule的凍結max_live核其原上限：原v3仍最多2，新next可最多4，各GPU必須不同。不能因新上限4而放寬舊v3快照。future next→next可保留4個live與多次handoff最早origin/token/receipt；exact parent pidfd、完整worker/config/child/FD/flock證明及retire前恢復／retire後failclosed規則不變。仍僅一個singleton controller，從不signal或重跑原模型工作。

26個next CPU synthetic tests加frozen v3/base各10個，共46 tests PASS，1.874秒：

```bash
PYTHONDONTWRITEBYTECODE=1 python3 -B -m unittest \
codex.retrieval_matrix_dispatch_next.test_next \
codex.retrieval_matrix_dispatch_v3.test_controller \
codex.retrieval_matrix_dispatch.test_dispatch
```

新增／更新驗證包括：兩個adopted加一個新running計為3，第一個容量sample不能啟動，5秒後可起第4；達4後不再query或起第5，uncertain亦持續占位。低於cap4仍拒絕occupied adopted GPU。builder從明確max2來源產max4。future cap4 parent可採納4live並保留最早MCQ origin；原v3 cap2來源若出現4live則retire前拒絕並恢復原parent。舊失敗歷史與所有taskdict不變、canonical validate無污染。全部spawn/signal/query均是假Host或fake Popen；FileLock僅測試私有暫存路徑。

| 更新檔案 | SHA256 |
|---|---|
| controller.py | f55362906d39c11a7dd6d91a938a5a7533cda1f2f1fb30143bcb2c1982d5a044 |
| build_schedule.py | bd59276fcb996c171b0a81601f3fdfc1c7f1fb93955b72372e474b3d593cbff9 |
| handoff.py | 5ea076fb1bd82a7393e2e02831cdb81eb25053bc749e9f3031a726f2b35dedbf |
| test_next.py | 8234b4a34396456c0ef2c356918e45e0ec36672e583d33386071d1e5d75cb8f5 |
| PREREG.md | aa284731afa9e89a306fcb8c7c0c0b83554f3f5ceb6604f50caf887a4ce8fc55 |

原cd02391的cap2交付報告保留為歷史，不以本報告改寫既有來源或主結果。additions候選與handoff模板未改，正式new schedule/additions/config仍待root在外層審查／凍結。已停筆交root獨立review；這份實作者準備報告不自動授權實際handoff或GPU工作。
