# [codex] central-v4 移交在退休前中止

2026-11-16 23:28（台北）。使用者四卡上限已形成凍結 source 83cd20a 與 schedule 3a234a83586b8c253d4690b6638243b71cd22b7a，source preflight PASS。root 依 exact-parent-only handoff 操作後，capture 發現既有 GME2 的同一 child PID 2842767/start_ticks271898609 已由 frozen central_task.sh 正常 exec 到 frozen run.sh embedding；原驗證要求 argv 與 wrapper 完全相同而拒絕。

未退休 parent2681588，未嘗試 replacement Popen，未建立新 state，finally 已 SIGCONT 同一 parent。原 MCQ 與 GME2 worker/model 未收到 signal、原 GPU lease 保留。hold/release 都即時 O_APPEND 登記。失敗 runtime dl/retrieval-matrix-dispatch/central-v4 以及 migration/config/source/state snapshots 完整保留，不以 recover 或重送改寫。

主機唯讀核對：GME2 child 與原 token 仍一致，argv 僅為中央 wrapper 已凍結的唯一 exec 展開；CORE MCQ argv 原樣一致。將另對此固定 launcher/plan/mode/model/ref/output 與 Git bytes 作極窄獨立修復和負向測試，重新凍結 source 及新 schedule/runtime；舊 v3/base 不改。現行中央仍是原 cap2，不能把四卡切換寫成已成功。

本報告未讀主判準數值。另確認 fixed-embedding-core2 三資料集計算已完成，等待另凍結 CPU 分析及非實作者重算。fixed-embedding-qwen2 在純合成 smoke 的 batch invariance gate 失敗，原失敗保留，不放寬已凍結容差或自動重跑；GME2 已通過相同技術 smoke 並進入 COCO index。這些皆為技術狀態，沒有賦予 P0 資格。
