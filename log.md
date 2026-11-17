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

- [claude] 01:55（2026-11-17）整理：經使用者同意，結束 2026-11-14 07:34 起的孤兒監控程序（PID 2780395，腳本與 lock 所在資料夾已不存在，未登記），刪除其寫入的未追蹤 `runs/`。`COORDINATION.md` 第 0 節新增本機 3090／H200 叢集對照表；`HOLDS.md` 新增排隊機制跨題目化與叢集切換兩項待辦。

- [claude] 02:40（2026-11-17）使用者決定暫停 rl-sparse-distill（完整論文約需 500–800 H200 GPU 小時，時間不夠），並照規則刪除題目 repo：GitHub `icml27-rl-sparse-distill` 只有匯入 commit 加上同步的報告 63，與中樞 `topics/rl-sparse-distill/` 的差別只有連結改寫；中樞恢復為正本並移除「已搬」標記。叢集 clone 已刪；本機 3090 的 `/ssd4/cactus8603/icml27-repos/icml27-rl-sparse-distill/` 請在本機那邊刪除。

## 未確認

- 上面各項只有回報首次出現的時間可核對（reports/INDEX.md）；使用者下指令的時刻，除審核報告引用到的幾處外沒有逐一核對。「約 13:00 起」「約 16:26–16:35」為概略時間。
- 使用者是否在對話之外對 R1 的改判、理論備案的取捨做過表態：只看得到對話紀錄。
- 2026-11-14 晚段（16:48 起）各時間取自對話紀錄時間戳（UTC 加 8 小時）；「二元」的解讀、主對話對「停掉吧」的解讀，都是對話紀錄裡主對話的說法，使用者沒有另外確認。
- 17:19 取消 T1 時使用者是否清楚知道是哪個工作：主對話 17:02 曾向使用者問「你要我取消它嗎？」，17:19 使用者回「停掉吧」；是否指這個工作，只能依主對話的理解（未確認）。

[codex] 2026-11-17 00:22 使用者要求暫停以統整 repo；已停止唯一中央的新派工（exact parent PID 2921085 SIGSTOP，worker／model 保留），子代理已停筆。已確認數字、資料範圍、34任務快照、WIP與接續任務見 codex/reports/2026-11-17-retrieval-paused-numbers-next-tasks-codex.md；來源 icml27-topic-search@f69b876213f98d1aa53788a62ecb6c67a9f233e0，中央 icml27-topic-search@f6780742ec62d7e50773054ea747c3692fce43dc。目標已由使用者清除；待其明示恢復。

[codex] 2026-11-17 03:28 P2 dev-only 恢復預檢停止：原中央身分、singleton、四張 lease 及四份完成收據相符；題目 repo P2 入口錯把舊中樞 SHA／資產當成本 repo，來源預檢 exit128。舊中央／候選含本輪禁跑 test，未直接恢復，未新送 GPU。依使用者明示停止條件更新 HOLDS 請求與 P2 狀態；dev 指標尚未量測。報告：[icml27-compositional-retrieval@127b8e1063cc2f20c13d2d6462ec9e93bfb7dfa7](https://github.com/cactus8603/icml27-compositional-retrieval/blob/127b8e1063cc2f20c13d2d6462ec9e93bfb7dfa7/codex/reports/2026-11-17-p2-dev-baselines-codex.md)。

[codex] 2026-11-17 04:19：Claude 代使用者 03:40 明示恢復後，來源預檢、原 token／四 worker 收據／singleton／leases 全核對；03:41 精確恢復舊中央，04:01 僅退役原父程序並移交唯一中央 3865368（start_ticks 273577676），保留原 34 任務與所有 worker，新增三組 P2。題目 repo 程式／凍結計畫／稽核與進度報告 icml27-compositional-retrieval@8018317b9e61364a4e74426509445940c2b50072；官方 benchmark test 可量，仍禁自建 panel test、下載、修改封存。CLIC 兩組在純合成 PNG smoke 發生原作者 image.layers 例外，將另凍結 JPEG 修正並保留失敗輸出；Qwen rerank8 原設定 OOM，不放寬。dev 尚無獨立核定數字。中樞遠端另有 Claude H200 紀錄，本次在乾淨 clone 合併本人狀態與追加行，保留遠端各行，不 rebase／stash／restore 共用 checkout。

[codex] 2026-11-17 04:31：JPEG P2 producer與非實作者重算來源已凍結並push（題目repo47eaeeb415a0219746f05eabb19f7403a27b14f7、cfc51e193466738e0da2171a857ddce626962d2b）；已兩次核對四個worker／receipt／lease，僅退役父3865368並移交唯一中央3934090／start273704952。前37任務逐項不變，新增CLIC JPEG兩組。CORE2正在圖庫推論，非實作者14合成測試PASS，尚無正式dev數字。進度與移交完整稽核已推送icml27-compositional-retrieval@3fe60cb；原PNG失敗和Qwen8 OOM都保留、不降門檻。

[codex] 2026-11-17 04:57：CORE2首批三個dev資料完整完成，fresh非實作者1578題×20000原始分數/排序/ID/正例名次、全指標與overall/by-k題級bootstrap CI、成本均PASS。dev mAP30.5149% [28.4136,32.6476]、small28.8087%、否定dev19.7564%；完整worker1675.988013s、allocated/reserved4.263/4.611GiB。逐題結果、hash-chain讀取收據與兩份報告已push icml27-compositional-retrieval@e81133843b0ab43de9434a667fe4dc83ca7ca0d9。CLIC JPEG兩組仍推論；初輪重排source已獨立審查/凍結但未送GPU，候選只K100。唯一中央3934090不變；不跑自建test、不下載、不修改封存或其他session。

[claude] 2026-11-17 05:10：H200 叢集逐題切換到題目 repo。multifidelity-attribution 的期望真值停損（smoke `27584`、陣列 `27590`、分析 `27591`）與 inoculation-lambda 的西語臂間停損（`27594`／`27595`）都從 `/work/ChihChungHsu/cactus8603/icml27-repos/` 的題目 repo checkout 送件。COORDINATION 第 0 節對照表已更新。使用者指示 H200 以多保真度為最優先：`27590` 最多同時 4 張；`27190_11` 重新排隊，給 Codex 的待辦見 `HOLDS.md`；Λ 排在多保真度之後。

[codex] 2026-11-17 05:12:57+0800 Claude代使用者04:52定案：dev-clean為P2主要指標；原版dev補充、既讀歷史揭露。CLIC-B32已GPU完成並fresh逐題重算PASS；完整矩陣仍由唯一中央3934090執行。題目repo@20cd31f 凍結native P0 CPU接續，無新增GPU排程器。

[codex] 2026-11-17 06:11:16+0800 三模型dev-clean fresh獨立PASS主表已發布8acaea7；GME2資格PASS、SigLIP2資格FAILED不放寬；唯一中央依來源/token/worker/lease兩次核對交接26290/c49d2d6。GPU0只本方小工作精確讓出給8B，原partial保留，Qwen8 batch1已running；完整矩陣持續，未跑自建test/下載。

[codex] 2026-11-17 06:31:06+0800 初輪四組clean K100計畫56b1da9與唯一中央schedule b0bdf35凍結；獨立源審PASS，兩次核對交接104711/start274417143，原45任務完整保留，加4P2。GPU5已跑CORE2接CLIC-L14，總4GPU工作；純CPU d1a3fae後處理不派GPU。完整矩陣持續，未碰自建test/下載。

[codex] 2026-11-17 07:26:29+0800 初輪四組clean K100計畫56b1da9與唯一中央schedule b0bdf35凍結；獨立源審PASS，兩次核對交接104711/start274417143，原45任務完整保留，加4P2。GPU5已跑CORE2接CLIC-L14，總4GPU工作；純CPU d1a3fae後處理不派GPU。完整矩陣持續，未碰自建test/下載。

[codex] 2026-11-17 07:27:27+0800 原P0資格CLIPS-L14 PASS、Qwen3-VL-Reranker-2B FAILED，獨立一致不替代資格；GME2 PASS與SigLIP2 FAILED保持。原CORE2 MCMR完成，唯一中央104711自動接第二組CORE2 clean重排，仍cap4/單lease。新batch1獨立CLI 7b46c5b、27測試與102來源gate PASS，尚未讀新8B效果；GME2/CLIPS P2 v1未送GPU且來源要求補強，v2準備。題目報告2d910c4更新；前次發布因狀態欄解析assert未套用HOLDS，本次已修正兩列處理狀態；無自建test/下載。

[claude] 2026-11-17 07:40：多保真度期望真值停損 `27590`（10 種子）與分析 `27591` 於 07:26 完成。Fable 獨立重算一致：現象 PASS（+0.137）、空間 PASS（精確 K=4 比最佳便宜集成高 +0.197 [0.153, 0.241]）、訊號 GREY（+0.020）。方向存活並升主線，證據 2.5→3.5、潛力維持 4。判定存於 icml27-multifidelity-attribution@c8cb2b1 報告 68。H200 優先權解除，Λ `27594` 已於 07:25 接續。

[codex] 2026-11-17 08:06:55+0800 依使用者實際排程器核對失敗條件停止P2/完整矩陣：v1 a11f0e2成功保留49+2工作；VLM2Vec2 MCMR完成後GME2 dev自動接GPU1；07:59 v2 3775853 first capture因新GME2 child未列literal allowlist而拒絕。未繞過、未退休原父，finally已自動恢復354478/start274947512，四GPU worker與singleton保留、原51佇列仍可自行接續。CPU自動followup/獨立代理已停止，不讀新效果；HOLDS新增停止請求並更新原P2/矩陣兩列。初始三模型dev-clean非實作者PASS數字保持，矩陣未完成；兩份報告/原P0資格與事故稽核見icml27-compositional-retrieval@5b1fe9b153766d4ea7b3931af80872dbeefdeb82。無自建test/下載/其他session修改。

[claude] 2026-11-17 08:50：依使用者指示，多保真度歸因升為 H200 主線；本題 H200 作業優先於其他所有題目（README、shortlist、HOLDS 已更新）。同成本便宜基線停損 PREREG 草稿完成、Fable 審查中。Λ 西語 join `27595` 因 PHM 收據的設計雜湊欄位被覆蓋而失敗；簿記修正 `caddb65`（不動分數與門檻，PREREG §6 第 10 點揭露），join 重送為 `27745`。

[codex] 2026-11-17 15:30:55+0800 Claude代使用者授權接續thread01a1e5cb，已在HOLDS登記完整新要求、devsets新版c64ea8 manifest、8B同worker單lease鏈／時段與搶占規則、批次一致性、COLA敏感度、CLIC MCMR NA、CPU下載WhatsUp官方資料。中央尚未交接，原354478與現存worker保持。READ_STARTED335305已用原append協定追加READ_FAILED行政作廢，原ledger45筆與anchor91筆prefix完整、無payload讀取；非實作者metadata核驗PASS。r2候選e59審查7項必修保留收據，修正版CPU回歸中；未讀任何新效果。同步原中央新增送件紀錄，未修改封存來源或他session。

[codex] 2026-11-17 15:45:09+0800 重新授權後central handoff成功：GME2 literal allowlist已補，51原task/order不變、12CPU回歸PASS、2b23來源fresh非實作者PASS、046 schedule凍結。無signal實機預檢及SIGSTOP後兩次完整capture均PASS，只退役exact舊父354478；唯一新中央2903985/start277764779/boot f34b259e-fe82-4dd1-ad7e-82b9e76bf226已完成來源gate並採用GPU0 CORE8 MCMR及GPU4/5兩CORE2原worker/lease，未signal worker。新版時段、chain、preemption escrow gates啟用；GME2批次1checkpoint續跑及新dev工作計畫待追加。READ_STARTED335305行政作廢獨立PASS。兩已完成8B任務純CPU有登記mechanical已啟動，資格與新主數字仍待fresh獨立；初始clean三模型表不變。收據icml27-compositional-retrieval@38bd6c9，報告持續更新，其他session修改保留。

[codex] 2026-11-17 16:54:41+0800 矩陣分批更新：中央2903985唯一並保留原三worker；What’sUp六原標註與四影像完整下載、SHA與所有引用存在PASS，不跑模型，作者commit7c1f255，指定SSD收據已保存，題目副本460a5bc。新dev3來源1f8、計畫7c27、fresh sourcePASS，56工作中央候選6009待來源審查及實機交接；未送未核定效果。Qwen8單筆native fresh獨立一致但真資格仍待；CORE8 embedding獨立成本欄修正新reader重算。兩指定報告已更新，來源/結果只於本方clone提交，其他session修改不碰。

[claude] 2026-11-17 11:30：依使用者決定集中資源在兩條主線（多條件檢索、多保真度歸因）。Λ 跑完西語停損後暫緩；文生圖（潛力 4→3）與強驗證器暫緩，佇列中的作業照常跑完。多保真度另發現先前的「Spell」baseline 實為 Mage 類；離線真 Spell（0 GPU）已實作，開發面板上 B=20 期望 LDS 0.168（舊版 0.121）、B=40 0.259（0.244）。
