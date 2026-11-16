# [codex] Central batch v2

2026-11-16 21:57（台北）。此批沿用已凍結中央排程與共用 singleton／每卡鎖。保留原14項作業的全部命令、模型、原freeze commit、plan SHA、輸出路徑、相對順序；只在MCMR前新增 native SigLIP2 的 COCO/Flickr P0。完整 FP32、影像4／文字32，尚未有 GPU 峰值實測，故事前採保守20GiB門檻。CPU audit 或檔案齊全均不算 P0 通過。

全域最多2個我們的工作，每張卡最多1個；加入前既有 compute PID 最多1個。記憶體足夠即允許共用，不要求 GPU idle 或 utilization 歸零。不碰他人程序、不自動重試或降設定。

新 schedule 必須先 commit，才可移交。在 host 環境以 boot ID／PID start ticks／完整 cmdline 確認舊中央身分，先短暫停止該中央程序，再重讀 atomic state、所有 attempt 與 owned worker 證據。只有14項全 pending、零attempt、無模型worker時，才退休舊中央並等待 singleton 釋放。任一項已 starting／running／不明，一律恢復舊中央、延後移交。保存原 runtime/state/events 與移交快照，舊任務不清空、不遺忘已存在的 attempt，不停止正在執行的模型。

此批 source freeze 不改既有 dispatch folder，不影響它原本的worker來源檢查。新增模型仍需自己已 commit 的 PREREG、plan、code 與獨立事前檢查。移交及送件依 COORDINATION 登記。沒有研究結果讀取；分析與資格判讀另外執行，非 GPU launcher 的一部分。
