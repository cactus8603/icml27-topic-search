# 決定記錄

時間一律為 UTC+8（對話紀錄的 UTC 加 8 小時），日期 2026-11-14。分三類標示：「使用者決定」、「回報原文」、「主對話判斷／建議」（主對話即 Fable）。時間依 Agent 派出與回報的時間戳，依 Opus 審核報告（`reports/2026-11-14-26-repo-review.md`）修正。

## 2026-11-14

- 約 13:00 起：對 S2 做第二輪審查（研究規劃、撞題、程式現況、數據強度稽核），回報首次出現 13:13 起。13:40，使用者決定 S2 改判 NO-GO（依主對話建議）；文件同步見回報 10。判定卡：`/work/ChihChungHsu/cactus8603/icml27-s2-gated-sink-kv/cards/S2-round2-verdict.md`（該卡在 S2 repo 尚未 commit）。
- 13:40：使用者說「把 R1 也用同樣方式查一遍」。13:40–14:03：對 R1 派四項審查（研究規劃、撞題、程式現況、數據稽核），做法與 S2 相同。14:03、14:19，主對話兩次問使用者 R1「要不要在文件上改判，還是先維持 GO 加註」；對話中沒有使用者回覆。
- 14:51–14:58：對 R3、K3、T2 各做數據快篩與撞題兩項（沒有做研究規劃）。主對話判斷（14:58）：照現狀沒有一題宜投 ICML。
- 15:09–15:24：NCM 可行性與後續文獻調查。15:31–15:36：ELSA 與 NCM 的類比分析。
- 15:20 派出、15:37 回報：電腦視覺一輪（五個範圍）。15:53，使用者說「你說的那些題目我都覺得沒有很適合」。
- 16:00，使用者說「NCM 可以重新研究一下去跑跑看」。16:01，NCM 三項診斷實驗（解碼懲罰消融、各層可預測性、品質損失分解）送出，輸出在 `/work/ChihChungHsu/cactus8603/icml/ncm-diag/`，結果待補。
- 16:09：ICML 主流一輪（七組）完成；主對話依舊標準剔除候選，使用者未表態；待用新標準重看。
- 16:00–16:25：標準的演變。16:00 使用者說上一輪候選「太擠，不夠理論」；16:01 主對話操作化為較嚴版本（未經使用者確認）；16:25 使用者更正為現行標準。見 criteria.md。
- 16:11–16:18：理論備案查核（Focus and Dilution 延伸、Transformer 可生成輸出數推廣到 Mamba、混合序列模型表達力延伸、多尺度平行生成表達力分離）。16:18，主對話建議放掉混合序列模型與多尺度平行生成兩項，並表示不再建議 NCM 的純理論線；使用者未表態，無使用者決定。
- 16:2x：第三輪搜尋（七路，依現行標準）派出，結果待補。
- 約 16:26–16:35：建立本 repo（未 commit、未設 remote）。16:35，Opus 審核報告回報（`reports/2026-11-14-26-repo-review.md`）。
- 16:39：第三輪七組回報收到（JSONL 中該 hand-back 首次出現 2026-11-14T08:39:07.873Z，即 UTC+8 16:39:07；原文 `reports/2026-11-14-27-round3-headroom-scan-seven-groups.md`）。依審核修正 repo 的回報首次出現 16:38:45（`reports/2026-11-14-28-repo-fix-report.md`）。
- 之後（時刻未記）：依審核報告修正本 repo 的文件與腳本（主對話決定的補充處理見各檔標示）。

- 16:48，使用者補充（原話節錄）：「如果你覺得CV方向太少非主流 我們可以停掉CV 專心ICML的方向 另外efficient AI也可以 但機器人方面的先不要 然後再補一個，如果你有看到很酷的新點子或是可以做一個完整研究的題目，也可以不用往SOTA去做 但我還是希望跟ICML方向比較一致」。整理為：可停掉電腦視覺、efficient AI 可以、機器人先不要、新點子或完整研究型題目也可以。16:48 主對話派出量化深查與 efficient AI 補查（Opus）。
- 17:00（回報首次出現 17:00:32）：NCM 三項診斷實驗最終回報（`reports/2026-11-14-29-ncm-three-diagnostics-final.md`），約 0.89 GPU 小時。主對話判斷（回報之外）：NCM 建議收掉、不當主線（見 candidates.md 新增一節）。
- 17:01，使用者說：「但二元沒用單純的分析文章應該也不會好上頂會吧？ 你可以把停損實驗拿下去做 就題目也可以清乾淨、釋放卡了 十週很趕，但現在正適合拼！」（「二元」依上下文應指 2 位元量化題；這是我的解讀，未確認）。整理為：純負面分析上不了頂會。
- 17:02：量化題停損實驗派出（Sonnet，預計放 `killtests/lowbit-reasoning/`）；17:02:52 因量化深查回報（17:02:25 收到）顯示題目被打掉而停止（TaskStop），主對話 17:03 表示停損實驗還沒用到 GPU。17:03 改派協調者（Haiku）帶三個備案深查（線上蒸餾、兩跳推理、資料歸因）。
- 17:03：流程修正（主對話自述）：先深查再動 GPU；主對話承認「在深查回來之前就建議開始停損實驗，順序錯了」。回報原文：`reports/2026-11-14-31-reasoning-lowbit-quant-deep-dive.md`。
- 17:14，使用者補充：「還有iclr nips近年的題目也可以」。17:14 主對話派出 ICLR 2026 與 NeurIPS 2026 題目探查（各一路 Opus）。
- 17:17–17:18：三個備案深查回報（原文由主對話從子代理紀錄取出）：`32-online-distill-rise-then-fall-deep-dive`、`33-finetune-new-knowledge-two-hop-deep-dive`、`34-cheap-data-attribution-deep-dive`。前兩項回報結論為不做，第三項為有條件做、偏向不做。
- 17:19，使用者說：「停掉吧」。主對話將其理解為取消 T1 的 slurm 工作 25966（`t1-w1r`），17:19:21 執行 scancel；該工作已執行 9 小時 11 分，結果檔保留在 `/work/ChihChungHsu/cactus8603/icml/icml27-t1-kv-stable-rate/`，取消時跑到第 5 格之後，未跑完的部分沒有了。ICLR／NeurIPS 探查未停。這個工作不是主對話送的，T1 也沒有審過。
- 17:31：ICLR 2026 探查回報（`35-iclr2026-topic-probe`，一個有條件做、信心偏低的候選：Neon 搬到語言模型）。17:37：NeurIPS 2026 探查回報（`36-neurips2026-topic-probe`，無合格候選）。
- 17:38：派出三項：Neon 文字版停損實驗（Sonnet，目錄 `killtests/neon-lm/`）、完整版 Craftax 與 Dr-CiK 深查（Opus）、今日後段結果寫入 repo（Sonnet，即本次整理）。Neon 停損實驗與 Craftax、Dr-CiK 深查結果待補。
- 本次補入的 reports 第 29–37 份的取得方式見 `reports/INDEX.md`。

## 2026-11-16

- [claude] 23:55 使用者定案「中樞＋題目 repo」：每題一個私有 repo，GPU 送件仍登記在中樞並共用同一排隊機制（`COORDINATION.md` 第 0 節、第 2 節第 7 點）。建立七個題目 repo；閒置三題自 `icml27-topic-search@6ee74ce` 搬入新 repo，中樞改封存；進行中四題先建空 repo，切換請求見 `HOLDS.md`。

- [claude] 01:05（2026-11-17）使用者決定提前切換：compositional-retrieval、inoculation-lambda、t2i-rare-failure、multifidelity-attribution 自中樞 `@28c4a9c` 以 cutover 同步到題目 repo，題目 repo 成為正本，中樞舊路徑封存。七個題目 repo 的本機 checkout 在 `/ssd4/cactus8603/icml27-repos/`。使用者同意中央排程恢復，由 Codex 依程序執行（`HOLDS.md`）。

- [claude] 01:30（2026-11-17）使用者決定暫緩題目不需要 repo：刪除 GitHub `icml27-large-commit-sets`（只有匯入 commit，內容與中樞 `topics/large-commit-sets/` 相同），中樞恢復為正本並移除「已搬」標記。

## 未確認

- 上面各項只有回報首次出現的時間可核對（reports/INDEX.md）；使用者下指令的時刻，除審核報告引用到的幾處外沒有逐一核對。「約 13:00 起」「約 16:26–16:35」為概略時間。
- 使用者是否在對話之外對 R1 的改判、理論備案的取捨做過表態：只看得到對話紀錄。
- 2026-11-14 晚段（16:48 起）各時間取自對話紀錄時間戳（UTC 加 8 小時）；「二元」的解讀、主對話對「停掉吧」的解讀，都是對話紀錄裡主對話的說法，使用者沒有另外確認。
- 17:19 取消 T1 時使用者是否清楚知道是哪個工作：主對話 17:02 曾向使用者問「你要我取消它嗎？」，17:19 使用者回「停掉吧」；是否指這個工作，只能依主對話的理解（未確認）。

[codex] 2026-11-17 00:22 使用者要求暫停以統整 repo；已停止唯一中央的新派工（exact parent PID 2921085 SIGSTOP，worker／model 保留），子代理已停筆。已確認數字、資料範圍、34任務快照、WIP與接續任務見 codex/reports/2026-11-17-retrieval-paused-numbers-next-tasks-codex.md；來源 icml27-topic-search@f69b876213f98d1aa53788a62ecb6c67a9f233e0，中央 icml27-topic-search@f6780742ec62d7e50773054ea747c3692fce43dc。目標已由使用者清除；待其明示恢復。
