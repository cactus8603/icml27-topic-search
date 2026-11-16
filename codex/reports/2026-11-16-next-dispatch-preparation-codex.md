# [codex] 下一批中央排程與 parent-only handoff：CPU 交付

實作在 `codex/retrieval_matrix_dispatch_next/`；本報告是實作者交付，不代替 root 獨立 prelaunch review。所有 frozen base/v3/模型模組均未修改。沒有讀 live runtime 或模型效果、沒有 GPU／nvidia query／真程序 signal／排程／正式 schedule build／handoff，也沒有 commit。

新 builder 只接受 root 明列的 frozen launcher/plan/full ref/output/completion/memory，逐檔 git-show 核 bytes。舊27 task dictionaries 與相對順序完全相同；不能退役任何舊列或重設 failed/uncertain。新增列有獨立 ID/output/digest，預設插在第一個 MCMR 前，亦可 explicit `--before` 指定原 task anchor。最終 additions input 本身亦須在新 schedule freeze commit 中。

controller 私有載入原 base dispatcher，保留原兩次5秒容量確認、拿卡鎖後重查、best-fit、STARTING-before-Popen與 Popen 後保鎖窗口。只重用 v3 seed/reconcile/stop_own，不修改 canonical B.validate或v3 global；原 worker最早 runtime/schedule/token/receipt、所有 terminal history 維持。採納與新工作同算 global2/perGPU1；failed不重跑，uncertain占位。停止只碰本代 exact Popen worker，採納者不 signal。

原8GiB/20GiB類別不變。inherited12GiB列亦不得改字典；新12GiB例外要求整份 model_spec 完全等於 frozen 原 SigLIP2 spec，並綁新 plan/model/task SHA。native rerank completion adapter只讀 scores/complete.json，核 exact variant/variant_spec/plan/ref與15912例、58891pairs、9425圖；不讀metrics。僅 qwen2/qwen8/jina_m0 已核 protocol且至少兩 published targets可用，前兩20GiB、Jina8GiB。

safe handoff 用明確 old/new schedule SHA/ref/runtime/controller/python與host-proven parent PID/start/boot配置。只對 exact parent做SIGSTOP→穩定snapshot／兩原 worker/child/config/receipt／最初source freeze／FD對GPUlock inode＋flock證明→再次核→SIGKILL已停parent；不signal任何worker/model/group。每個 parent signal 都先 pidfd_open，再核token/cmd，以 pidfd_send_signal排除數字PID再利用；沒有 os.kill fallback。此 Linux/system Python3.10 提供兩API，確認僅用 hasattr，沒有真的開 pidfd或送訊號。

retire前任何失敗恢復原parent；原工作若競速完成，只在terminal receipt身分一致且原model child已死時保留終態。retire後模糊race／lease失效／singleton占用就留下previous-state/migration/error等待root，原worker繼續。explicit recover只允許舊parent已退、沒有新state及任何replacement Popen嘗試時重核原證據再起中央；Popen前先保存attempt marker，任何不確定啟動不自動再spawn。所有真操作未執行。

## CPU 驗證

21個新測試與 frozen v3/base各10個，共41 tests PASS，1.278秒：

```bash
PYTHONDONTWRITEBYTECODE=1 python3 -B -m unittest \
codex.retrieval_matrix_dispatch_next.test_next \
codex.retrieval_matrix_dispatch_v3.test_controller \
codex.retrieval_matrix_dispatch.test_dispatch
```

涵蓋原27 task source preservation、兩live跨runtime origin、cap2／perGPU1、failed不retry、任何舊dict/order變動拒絕、SigLIP完整spec+12GiB SHA例外、native rerank exact completion、私有import無污染與own-stop-only、parent/child/receipt錯身分拒絕、ambiguous starting拒絕、retire前SIGCONT、正常terminal競速保留、retire後failclosed與explicit recovery、Popen不確定不得再啟，以及mock pidfd signal必須重核cmd。唯一真FD/flock測試使用測試程序自己的CPU FD與hf_home/tmp私有lock，不接觸shared GPU locks。其餘 process/signal/spawn/query全部合成。

## 候選與限制

`additions.candidate.json` 只有四個已凍結明確請求：新 SigLIP2 einops修復 f3f787eeb193aad8dcc6f2ee43e9609ddf5b47eb，以及54f544ce54bca82be3cc8b3c72662b354c6d3e40的三個 approved native rerank variants。已逐檔查Git bytes、SigLIP完整spec與rerank protocol。原54f paired3因GPU條件FlashAttention缺einops而明確排除，待新paired_runtime_fix ref，沒有把舊ref放進可執行清單。

handoff.example.json 的pid0／新schedule SHA／新ref是故意無效的placeholder，不能執行。候選和範例只供review，沒有正式schedule或runtime。root將最終 additions/schedule/config放dispatch_batches外層，完成獨立審查／source freeze／短claim逐筆登記後，才由唯一中央接手。

## 本次 source SHA256

| 檔案 | SHA256 |
|---|---|
| controller.py | 20de620e4e4dd9c70b25bee2c37f039d9301b991163351f749eaa161dc8ae325 |
| build_schedule.py | adf341ad540d404f2a64e10caa1aa676fd348a403419e25a8812b19ffefce0b8 |
| handoff.py | e1795d0ff2511f39aa9fb95fd93298fd273ae5f82584826692322226fbaed56f |
| test_next.py | e596023c6427fb71194c0d663cafd67190257eef12e039978990ed606986de40 |
| PREREG.md | db143587669d6ff195f0e724207076688301cd698249c8f0fcd5c58ed11b4d47 |
| additions.candidate.json | 23a7708b861b2496ec190b6f8325591ff3154cb9f8bd8d5111c5e6e7ef1336f6 |
| handoff.example.json | c8c0d24daafe1ae02f34505a146d7283b2bb3ca39ddd190e44d29bd55fa8bdfb |

停筆交 root；source若再變更須重核相應SHA與測試，不因本報告自動授權任何GPU／程序操作。
