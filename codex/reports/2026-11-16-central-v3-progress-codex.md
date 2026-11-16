# [codex] 中央 v3 技術修復與不中斷接手紀錄

2026-11-16 22:46，台北。本記錄只含操作證據與技術狀態，未讀主判準結果。

依使用者「放得下即可，不必空卡，但不要三四人搶卡」指示，唯一中央仍維持 global cap2、perGPU1、加入前最多1個既有 compute PID、兩次5秒記憶體確認與持鎖後重查，不限制 utilization。所有子代理只做 CPU 準備與審查。

## 凍結與審查

原 CORE/Qwen3 decorator API 修復及 Qwen2 isolated einops overlay 於 `99431594893bba94337b3befd10a7a483ba7374f` 凍結；原模型、資料、settings、precision、batch、processor、DoRA 與 metrics 不變。兩份獨立 prelaunch review 已提交。中央27項清單與 v3 adoption controller 於 `665a5743c6c1308ae84d25dd450a625cb98fa3b4` 凍結；獨立10個 CPU adoption/cap2/receipt/停止範圍測試通過，host preflight source_gate PASS、legacy_live=[]。

## 主機移交證據

舊中央 PID2557793/start271448952/boot f34b259e-fe82-4dd1-ad7e-82b9e76bf226 先核對完整 argv，再只停止該父程序。原 MCQ CORE2 worker2563964/child2563981 仍存活，完整 argv、原 token/config/receipt/schedule SHA 相符；worker fd4 對同一 GPU-2719162b-ac91-96cf-5684-57f34e511ff6.lock inode，獨立 flock 證明鎖仍持有。快照、原 receipt/config 與原 state 均保留後，只退休這個已停止且精確核對的中央 PID，沒有 signal worker、child 或 process group。

新中央 PID2681588/start271663971 接手原 MCQ worker及其原卡位鎖，仍把它算入cap2；未重跑或複製分數。原8個技術失敗保留 failed；4個已知同 API 錯誤且從未啟動的8B embedding/MCMR 項保留 cancelled/retired_without_start，12個新修復作業有新ID/ref/plan/輸出。移交腳本由非實作者只讀核查後才執行；runtime metadata可完整追溯。

Migration receipt：`dl/retrieval-matrix-dispatch/central-v3/migration.json` SHA256 `2e81f55fa91801a8cbeadbbd65cd629413250c64c5f0c67654aed5daf8c08770`。新schedule SHA256 `48f2794f6a5a771ed08c07496b2ed54435af33c7b16442e577d8fa61a6f8519c`。

## 當下技術狀態

原 CORE2 MCQ 持續正式推論；fixed-embedding-core2 已通過真正固定完整權重 GPU synthetic smoke，接著執行三個原 P0 index。當下只有此兩個GPU作業由中央管理。SigLIP2 於 runtime imports 的 PEFT/Bloom/FlashAttention 鏈缺 einops，在 forward 前失敗；紀錄保存，另新目錄修復並審查，不改原 producer 或自動重試。Qwen2 修復目前仍 pending，不把 CPU tiny probes 視為真 GPU smoke。

所有主結果仍未讀。後續由另凍結 analysis PREREG+launcher執行機械判讀、獨立重算，再供使用者定案；producer PREREG bytes保持原凍結版本。P0尚未宣告資格通過，P2/P3仍需真資格閘門。
