# [codex] next handoff：固定 Qwen2 wrapper exec 的窄修交付

root回報原v4移交在退休舊parent之前，因Qwen2的已凍結central_task.sh合法exec成run.sh而中止，並已恢復原parent。本次僅依root提供的技術失敗原因與已凍結來源修補next handoff；未讀live state、原失敗runtime、模型arrays或效果，未操作GPU、程序、schedule或送件。沒有修改原v3/base、原模型來源或既有失敗記錄。

一般child command仍逐元素完全相同。新增例外只允許99431594893bba94337b3befd10a7a483ba7374f四個原task：fixed-embedding-gme2、fixed-embedding-vlm2vec2、fixed-mcmr-gme2、fixed-mcmr-vlm2vec2。task ID、canonical launcher/plan/output/cwd、full ref、model、原完整command和兩個來源SHA均須與明列常數一致。再核wrapper、run.sh、plan三檔的實際SHA與同一原ref Git bytes，plan須FROZEN/reviewed，phase對應runtime_fix.mode且所有完整model spec digests相同。

唯一導出的實際argv是 `bash CANONICAL_RUN_SH MODE ORIGINAL_REF ORIGINAL_PLAN ORIGINAL_MODEL ORIGINAL_OUTPUT`，其中MODE僅由該原plan明列的embedding或mcmr決定。不解析任意bash/exec，不接受suffix、別名、額外參數或其他ref。原Host.command的PID/start/boot檢查與原receipt/config/token、worker command、FD/inode/flock檢查保持原樣。每筆migration的worker proof同留dispatch_command、實際child_command與launcher_expansion來源、phase/mode/spec SHA；不以轉接後argv覆寫原task或receipt。

wrapper SHA：d603ddec31f8bc95423ccf8505ca6a8325e32e333f671fd5e332a9db68b42abb。run.sh SHA：ba7089ca60866ef7e94589333524934d333a94135100f5a42be25ae1faaeae9d。embedding plan SHA：07a9b6859e17d12e99a00439327381655ec22b1369e2d27a9fa4a5f8a5ea3053。MCMR plan SHA：2e525b5698782f2abd89ded3042e3416886a3c666b4d470c77bf888fc8dbb9a0。

33個next synthetic tests加凍結v3/base各10個，共53 tests PASS，2.243秒：

```bash
PYTHONDONTWRITEBYTECODE=1 python3 -B -m unittest \
codex.retrieval_matrix_dispatch_next.test_next \
codex.retrieval_matrix_dispatch_v3.test_controller \
codex.retrieval_matrix_dispatch.test_dispatch -v
```

新增測試核四個合法組合及逐檔Git證明、普通exact command不擴張、錯mode/model/ref/plan/output/run path/額外argv拒絕、原task身分變更拒絕、run.sh固定Git bytes不同拒絕、phase/mode/FROZEN/reviewed/完整model spec改動拒絕；整合fake handoff確認同child PID/start/boot與新舊argv證據保留。所有signal/spawn均是假Host或mock；Git只讀已凍結來源，FD/flock沿用私有CPU暫存測試。第一次合併命令曾誤寫不存在的test_v3模組，正確test_controller命令如上已全數通過。

AST語法檢查通過。controller.py與build_schedule.py逐bytes等於原83cd20a，容量、cap4、原task dictionary與排程行為未改。

| 檔案 | SHA256 |
|---|---|
| handoff.py | ab665b7d5df89a26955bc52ae8d8e2b5701a19e7294840f1804500c5c78bdeb4 |
| test_next.py | 76cd218d8a544cd27e6d6ba0f8c2e553b9eb658cd5e9c334a9d5a9330201cb27 |
| PREREG.md | 4b72015417ebe9af543a4e93bec6214502fc4e2af9204863dba3cb047c21f492 |

實作者準備交付，已停筆，待root獨立審查、新source/schedule/config/runtime凍結才可操作新移交。此報告不授權執行，也不把原失敗v4 runtime當可重試runtime。
