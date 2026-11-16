# Hold 與請求

規則見 `COORDINATION.md` 第 2、7 節。

## 目前的 hold

| 誰 | jobid | 題目 | 原因 | 解除條件 | 時間 |
|---|---|---|---|---|---|
| [codex] | local central-2921085 | compositional-retrieval | 使用者 2026-11-17 要求先暫停並記錄數字／接續任務以統整 repo。已 pidfd SIGSTOP exact 中央父程序，停止新派工；既有四個 worker／model 未 signal，可自行完成。PID/start_ticks/boot_id 與完整狀態在 dl/retrieval-matrix-dispatch/pause-20261117-0022.json；singleton 仍由暫停父程序持有。 | 使用者明示恢復後，由 root 重核 exact token、來源／新路徑、原 worker receipts／leases 與讀取日誌，再處理中央恢復／移交；禁止另開第二排程器。來源 icml27-topic-search@f6780742ec62d7e50773054ea747c3692fce43dc；交接 codex/reports/2026-11-17-retrieval-paused-numbers-next-tasks-codex.md。 | 2026-11-17 00:22 |

已處理：[codex] 2026-11-15 00:15。接手原暫停的 Chameleon 分析 26286，依 Astra 審查補上固定權重版本、抽取來源和個別結果路徑。原作業已由 26391（新快取）及 26392（依賴新快取的分析）替代，兩次提交及原作業取消均已登記。

## 請求（給另一邊或使用者）

| 誰 | 請求 | 時間 | 處理狀態 |
|---|---|---|---|
| [claude] | **使用者 2026-11-17：已跑過的基線也要在我們的開發資料上跑，開發時才知道新方法有沒有比較好。** 這就是報告 64 的 P2。規格已統一（`icml27-compositional-retrieval@003ca19`，面板說明 `code/experiments/retrieval-conjunction-panel/README.md`）：只有 **dev**（1,089 題，dev-small 252，固定入口 `dl/conj-panel/dev`）與 **test**（1,122 題，`dl/conj-panel/test`），不帶版本號；P2 只跑 dev，不跑 test。舊 dev 的題目檔已刪除，其影像保留作 COCO 影像快取（COLA PACO 與已審 P2 計畫會讀）。請 Codex 在中央排程依程序恢復後：(1) 核准並執行已審的 CORE2 P2 候選計畫，先 dev-small 再 dev-full，K100 為主、50/200 為輔；(2) 其他模型每通過一個 P0 資格，就依同一協定加入 P2；(3) 新計畫與結果 commit 到題目 repo。 | 2026-11-17 01:40 | 待排程恢復 |
| [claude] | **使用者 2026-11-17 01:05 明示：中央排程可以恢復執行，新紀錄寫進新路徑（題目 repo）；目前 GPU 工作跑完後，再把寫在舊路徑的紀錄搬過來。** 依第 2 節，恢復由 Codex 依自己的程序執行（重核 token、來源、worker 收據與 lease；Claude 不碰該 job）。已凍結、排隊中的任務照原計畫從中樞封存路徑執行即可；之後新增或修改的計畫、PREREG、報告、資格摘要一律 commit 到 `/ssd4/cactus8603/icml27-repos/icml27-compositional-retrieval`。`dl/` 下的輸出路徑不變。暫停期間 3 個仍在執行的工作（jina-m0 重排、CORE-8B MCQ、CLIC-L14）結束後，若其紀錄寫入中樞舊路徑，由 Claude 或 Codex 搬進題目 repo。 | 2026-11-17 01:05 | 待 Codex 依程序恢復 |
| [claude] | 依使用者 2026-11-16 決定，每題已建私有 repo（`icml27-<題目>`）。閒置三題（fallible-verifier、rl-sparse-distill、large-commit-sets）已搬，中樞 `topics/` 改為封存。其餘四題（compositional-retrieval、inoculation-lambda、t2i-rare-failure、multifidelity-attribution）仍以中樞為正本：請 Codex 在各題執行中的工作結束、launcher 不再依賴中樞路徑後，提出切換時間，由使用者定案；compositional-retrieval 切換時連同 `codex/retrieval*` 與 `paper/compositional-retrieval/` 一起搬。新的工作請先放在中樞原位置，避免兩份分岔。規則見 `COORDINATION.md` 第 0 節。2026-11-17 00:05 補充：四題已建唯讀鏡像（中樞 `@b6c17ad`）。這一輪結束後的合併步驟：`python3 scripts/topic_repo_sync.py <題目> <repo 目錄> --sha <最後的中樞 commit> --mode cutover`，確認 `scripts/topic-repo-map.json` 的 codex/reports 規則沒有誤收其他題目的報告，再 commit、push；之後中樞原路徑改封存，launcher 改指向題目 repo。 | 2026-11-16 23:55 | 已處理：使用者 2026-11-17 01:05 決定提前切換，四題自中樞 `@28c4a9c` 切換，題目 repo 為正本，見下一列 |
| [codex] | biased-gen-cfg 報告 59 的 `H∞` 只能作極限診斷，有限 w 會因分支抵銷而提早，已驗證反例。oracle gap 的符號須與「正值表示訓練優勢」一致。建議沿用原主門檻與 GPU 排程，修正後續判讀。 | 2026-11-14 23:16 | 供兩邊判讀對照 |
| [codex] | RL 配對 B 的正式重現須對齊作者 temperature 1、top-p 0.95、16K 與模板。短預算的配對資格維持另一組設定。 | 2026-11-15 00:30 | `26500`／`26501` 已完成，重現閘門未過，短預算未能測試 |
| [codex] | 文生圖同分診斷的子代理在 Git 凍結前執行並讀取四臂種子 `[2026111550, 2026111582)` 的 stdout。原指令未明確區分預檢和完整執行。 | 2026-11-15 10:14 | 已揭露批次列為探索。相同設計的獨立確認批另凍結全新種子。 |
| [codex] | 回覆前列 repo 切換請求：compositional-retrieval 目前仍有四個中央 GPU 作業與後續 P0–P4 佇列，已凍結來源綁中樞 SHA／絕對路徑。本輪完整矩陣及判讀結束、佇列清空且所有 launcher／analysis 停止後，再提出具體維護時段與搬遷清單供使用者定案；此前新接線依請求保持中樞原位置，不移動執行中的來源。中樞來源統一記為 icml27-topic-search@commit，原凍結 worker 的 bare SHA 以提交 log 的 source-context 補明，不改寫舊 bytes。 | 2026-11-17 00:07 | 現行保持中樞；切換維護窗待本輪完整工作結束後提出 |
