# ICML 2027 選題過程記錄

## 目的

記錄「找 ICML 2027 投稿題目」的過程：訂了什麼標準、看過哪些候選、每個候選當時怎麼判斷、依據哪一版標準。

## 目前狀態（2026-11-14 晚；時間一律 UTC+8）

標示說明：「使用者決定」、「回報原文」與「主對話判斷／建議」（主對話即 Fable）分開寫。細節見 `candidates.md`「深查結果與目前狀態（2026-11-14 晚）」與 `log.md`。

- 現行標準見 `criteria.md`。標準在同一天內改過一次（較嚴版本是主對話對使用者「太擠、不夠理論」的操作化，未經使用者確認；16:25 使用者更正為現行標準），因此部分候選需要用新標準重看。
- S2 第二輪改判 NO-GO（依主對話建議，使用者決定）。R1 做了四項審查，主對話判斷目前 GO 站不住，文件上是否改判使用者未決定。R3、K3、T2 做了快篩，主對話判斷照現狀不宜投 ICML。
- 使用者 16:48 補充：可停掉電腦視覺、efficient AI 可以、機器人先不要、新點子或完整研究型題目也可以；17:01 表示純負面分析上不了頂會；17:14 表示之後補 ICLR、NeurIPS 近年題目也可以。電腦視覺候選（含 Physics-IQ）已移除。
- NCM：三項診斷實驗完成（約 0.89 GPU 小時，回報 29）；主對話建議收掉，不當主線。
- 第三輪（七路）候選與備案的深查結果：推理模型 2 位元量化（主對話判斷不做；停損實驗派出後停止，未使用 GPU）、線上蒸餾先升後降（回報：不做）、微調新知識兩跳推理（回報：不做）、低成本資料歸因（回報：有條件做、偏向不做）。
- ICLR 2026 探查：一個候選「把 Neon 搬到語言模型」（回報：有條件做、信心偏低）；停損實驗 2026-11-14 晚派出（`killtests/neon-lm/`），結果待補。NeurIPS 2026 探查：無合格候選。
- 完整版 Craftax 少資料世界模型、Dr-CiK 文件檢索差距：深查進行中。
- 流程修正：先深查再動 GPU。17:19 依使用者「停掉吧」取消 slurm 工作 25966（t1-w1r，已跑 9 小時 11 分，結果檔保留）。
- 本 repo 尚未 commit，也沒有 remote。已經 Opus 審核（`reports/2026-11-14-26-repo-review.md`，審核對象是先前版本），並依審核意見修正；本次晚間補入的內容尚未經 Opus 審核。

## 目錄

| 路徑 | 內容 |
|---|---|
| `criteria.md` | 選題標準、標準的演變、流程做法 |
| `candidates.md` | 候選總表，每列附對應回報 |
| `reports/` | 從對話紀錄抽出的子代理回報原文（36 份，含第 26 份審核報告；另有 1 份因含電子郵件格式字串未收錄）；`reports/INDEX.md` 有對照 |
| `scripts/` | `extract_reports.py`（抽回報）、`stats.py`、`kw.py`（ICML 會場資料統計） |
| `dl/` | ICML 會場資料 JSON（`icml25.json` 14 MB、`icml26.json` 16 MB），已列入 `.gitignore`（不進 git，檔案留在磁碟上）。統計腳本以腳本所在位置推出路徑，任何目錄都能跑 |
| `log.md` | 依時間記錄做了什麼決定 |

## 資料來源

- 對話紀錄：本次對話的 JSONL（不放進 repo）。重新抽取：`python3 -I scripts/extract_reports.py <jsonl> <輸出資料夾>`，輸出為 raw-NN 檔。
- ICML 會場資料（`dl/` 不進 git，是公開資料、可重新下載；檔內有第三方作者的姓名、個人頁網址，institution 欄還混有作者電子郵件，所以不進 git）：
  - ICML 2026：下載網址 `icml.cc/static/virtual/data/icml-2026-orals-posters.json`；下載日期 2026-11-14；存成 `dl/icml26.json`；md5 `f670e6e57625ea10cbaae0312abfa234`。
  - ICML 2025：存成 `dl/icml25.json`；md5 `6f5627d182048d4e804f0e07e06fa3bb`。網址依回報 22 第 13 行「與 2025 年同名檔案」，網址形式推測為 `icml.cc/static/virtual/data/icml-2025-orals-posters.json`；這個網址沒有實際驗證過。
  - 檔名 `icml25.json`、`icml26.json` 是原工作資料夾的命名。
- 統計腳本跑法：`python3 -I scripts/stats.py`、`python3 -I scripts/kw.py`（任何目錄都能跑，已從 repo 根目錄與 `scripts/` 各跑一次，輸出相同；輸出未存檔）。
- 既有審查材料（只連結、未複製）：
  - `/work/ChihChungHsu/cactus8603/icml27-s2-gated-sink-kv/cards/S2-round2-verdict.md`（在 S2 repo 尚未 commit）
  - 總覽 repo `/work/ChihChungHsu/cactus8603/icml/ICML2027`
- NCM 診斷輸出：`/work/ChihChungHsu/cactus8603/icml/ncm-diag/`。

## 下一步

1. 使用者檢視修正後的 repo（尚未 commit）。
2. 等 Neon 文字版停損實驗、Craftax 與 Dr-CiK 深查的結果，補進 `candidates.md`。
3. 使用者決定下一個要做的題目；R1 是否改判仍待決定。
