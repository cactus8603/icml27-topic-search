# [codex] Additive central dispatcher 獨立啟動前審查

Codex root（非此模組實作者），2026-11-16 23:11，台北。已逐讀 controller、builder、handoff、PREREG及測試，並重跑21新＋10v3＋10base，共41項CPU tests PASS（1.389秒）。尚未由本新模組操作任何現行程序或GPU；真正接手仍須已commit的schedule/config、host exact證據與sharedlog登記。

私有base module不污染canonical dispatcher；V3 seed/reconcile/stop_own只使用已核的stateless helper與原process token。舊27 task字典及相對順序完全不改，保留所有failed/cancelled/terminal/attempt/token及最初receipt來源，不重跑失敗。新請求逐檔核frozen Git bytes與完整argv，新ID/output明列。12GiB新例外要求整份原SigLIP2 model_spec精確相同；native rerank completion不讀metrics且核完整variant/ref/plan/分母。維持global2/perGPU1、既有PID最多1、5秒兩次容量與鎖後重查。

gate同時核base、v3、next執行source及明列additions input在最終schedule commit中，ownfolder clean。多次handoff保留最初worker runtime/schedule；採納與新工作同算上限，原GPU鎖占位。停止路徑只處理本次Popen handle且exact token相符的worker，不signal採納者。

handoff只接受明確own controller allowlist、old/new source refs、schedule SHA、不同runtime與exact parent start/boot/cmd。SIGSTOP後snapshot原state，兩次核每原worker/child/receipt/config/source/FD同lock inode及flock，才可退休該parent。每parent signal以pidfd固定程序後再次核身分，不以數字PID掃卡或signal model/group。retire前失敗恢復同parent；terminal競速要原model child已死才能保留。retire後有模糊證據則停下保留完整migration，explicit recovery只可在尚無replacement Popen嘗試及新state時重核後起中央。Popen前持久化attempt，無不確定重送。

合成測試覆蓋上述兩個active origin、cap2/perGPU1、錯token/argv/config、ambiguous starting、GPU鎖、terminal競速、恢復及pidfd範圍。沒有用此CPU測試代替真正host liveness/lease驗收。候選4項不含有已知缺依賴的舊paired54f；三個新paired需另新runtime freeze才加入。

本次核查source SHA256：

- PREREG.md: `db143587669d6ff195f0e724207076688301cd698249c8f0fcd5c58ed11b4d47`。
- additions.candidate.json: `23a7708b861b2496ec190b6f8325591ff3154cb9f8bd8d5111c5e6e7ef1336f6`。
- build_schedule.py: `adf341ad540d404f2a64e10caa1aa676fd348a403419e25a8812b19ffefce0b8`。
- controller.py: `20de620e4e4dd9c70b25bee2c37f039d9301b991163351f749eaa161dc8ae325`。
- handoff.example.json: `c8c0d24daafe1ae02f34505a146d7283b2bb3ca39ddd190e44d29bd55fa8bdfb`。
- handoff.py: `e1795d0ff2511f39aa9fb95fd93298fd273ae5f82584826692322226fbaed56f`。
- test_next.py: `e596023c6427fb71194c0d663cafd67190257eef12e039978990ed606986de40`。
