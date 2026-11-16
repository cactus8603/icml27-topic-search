# [codex] Qwen2 frozen wrapper exec 獨立 prelaunch review

2026-11-16 23:38（台北）。root 核對 handoff.py、test_next.py、PREREG 的完整 diff 與兩個原 plan、central_task.sh、run.sh 在 99431594893bba94337b3befd10a7a483ba7374f 的 Git bytes；controller 與 builder 仍等於 83cd20a。此處沒有修改實作，沒有讀取主判準模型數值。

例外明列四個原 task ID、固定 ref、兩個原 plan SHA、phase/runtime_fix.mode、兩個完整 model specs digest、wrapper/run SHA，以及 cwd/launcher/model/output/完整 argv。只容許 frozen central_task.sh 的字面 exec 到同目錄 run.sh；所有其他 argv 仍精確比較，沒有通用 shell 解析、suffix 或路徑別名。capture 仍核原 child PID/start_ticks/boot_id、receipt/config、原 source、worker FD 與 GPU flock。migration 同留原 dispatch_command、實際 child_command 與展開來源證據。

root 重新執行 33 next、10 v3、10 base 共 53 項 CPU synthetic tests。四個合法展開、每欄負向、run Git bytes 變動、完整模型或 phase/mode 變動、fake 完整 handoff、cap4/perGPU1 與原失敗歷史皆涵蓋。測試通過後可另凍結並建立新 schedule/runtime；之前 aborted central-v4 完整保留，不能重寫或自動 recover。原 v3/base 及模型科學設定不變。

審查版本 handoff SHA256 ab665b7d5df89a26955bc52ae8d8e2b5701a19e7294840f1804500c5c78bdeb4、tests 76cd218d8a544cd27e6d6ba0f8c2e553b9eb658cd5e9c334a9d5a9330201cb27、PREREG 4b72015417ebe9af543a4e93bec6214502fc4e2af9204863dba3cb047c21f492。

root 完整53 tests PASS（2.288秒）。source freeze deaf213 後另建立 central-v4b：與原 v4 相比唯有已審三檔 controller_sources hashes 更新；34 task dictionary／順序、七個新 output、所有 scientific plans、adoption origin、max_live=4 均完全相同。新 schedule SHA256 1638788fee9d62abafe9cf3bc6f4ba9e7b2a17b255885451701c1102de02ae8c，C.validate PASS，須另 commit 後才移交；原 aborted v4 保留。
