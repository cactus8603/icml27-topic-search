# [codex] Native SigLIP2 runtime repair

僅補 peft／Bloom／flash padding 需要的 einops0.8.2。原 native model class、FP32、batch4／32、完整原資料與 Google metric 不變。新來源／dependency表／計畫另行審查及凍結；正式執行仍由中央負責。

`central_task.sh FULL_COMMIT PLAN siglip2 OUT` 保持中央四參數介面，依序原 smoke與兩個完整 index。輸出限 `dl/retrieval-matrix-native-runtime-fix/runs/` 的新子目錄，原失敗收據保留。CPU audit／analysis與資格審查另行執行。

新 dependencies.json 完整列出原payload與新增overlay，新的 qualification.py 可獨立重驗全部原證據並產生專屬 scope；不承接舊P0資格，也不自動修改任何 P2 接線。原 port／mapping／歷史 NumPy差異仍須獨立接受。PREREG說明完整限制。
