Both repos are updated, documentation only. Nothing was committed or pushed, and no file was deleted, moved, or had its code or results changed. While checking the facts against the source files I found three places where the brief does not quite match them. In each case I wrote what the source file says (details in section 3).

## 1. Topic repo `/work/ChihChungHsu/cactus8603/icml27-s2-gated-sink-kv`

```
 README.md                       | 7 +++++--
 cards/S2-Astra-method-design.md | 2 ++
 cards/S2-W4.md                  | 2 ++
 cards/S2-repair-exploration.md  | 2 ++
 4 files changed, 11 insertions(+), 2 deletions(-)
?? cards/S2-round2-verdict.md   (new)
```

**README.md diff:**
```diff
@@ -1,5 +1,7 @@
 # KV Cache Compression Research
 
+> **狀態：2026-11-14 第二輪改判 NO-GO，方法論文方向結案（使用者決定）。** W1 的現象（gate 讓保護前 4 個 token 的收益縮減，R=0.967／0.933）不撤回；但 W2 的逐 head 機制未過、gated 模型沒有更好壓縮、gval 在 RULER DEV 全部落後最強 baseline、repair 對 SnapKV 沒有可辨識的增益，且主要做法已有接近工作。W4 確認實驗與下一輪六設定不再排程；下方內容保留作紀錄。理由與出處見 [第二輪判定卡](cards/S2-round2-verdict.md)。
+
 研究固定 KV 記憶體預算下的 token 選擇與 attention-output repair。...
@@ -115,9 +117,10 @@
-| 2026-11-13 | S2-W3 | ... | [原始彙總](results/s2-w3/summary.md)；W4 新書確認，保留資料重用限制 |
-| 2026-11-13 | RULER DEV | ... | [完整分數](results/ruler4k-dev/summary.md)；補 controls／baseline admission；C2 前置門檻未達 |
+| 2026-11-13 | S2-W3 | ... | [原始彙總](results/s2-w3/summary.md)；W4 新書確認，保留資料重用限制（2026-11-14 結案，未執行） |
+| 2026-11-13 | RULER DEV | ... | [完整分數](results/ruler4k-dev/summary.md)；補 controls／baseline admission；C2 前置門檻未達（2026-11-14 結案，未執行） |
 | 2026-11-13 | S2-B2 建置 | ... |
+| 2026-11-14 | S2 第二輪判定 | [S2-round2-verdict](cards/S2-round2-verdict.md) | 無新 job | NO-GO | 使用者決定改判，W1 GO 不撤回。W2 P3 ρ=0.265（門檻 0.50）；RULER DEV 9 格 gval 全部落後最強 baseline 4.3–12.2 個百分點；repair Projected 37.98 對同 runner SnapKV 37.31，區間 [−1.23, 2.66] 跨 0 | 結案。W4 C1／C2／E1–E3、六設定比較、repair 移植 8B 不再排程；W1 現象與 baseline 基礎設施保留 |
```
(The "..." parts are unchanged text. The W3 and RULER DEV rows only had the bracketed note appended to their "下一步" cell.)

**cards/S2-W4.md** (appended at the end of the existing addendum):
```diff
+
+2026-11-14 結案：S2 已於 2026-11-14 第二輪改判 NO-GO（使用者決定），本卡 C1／C2／E1–E3 未執行，不再排程。見 [第二輪判定卡](S2-round2-verdict.md)。本卡原有內容與數字不改。
```
**cards/S2-repair-exploration.md** (appended at the end, after the latest section):
```diff
+
+2026-11-14 結案：S2 已於 2026-11-14 第二輪改判 NO-GO（使用者決定），本卡最後一節的六設定比較與 smallDEV52 不再排程。見 [第二輪判定卡](S2-round2-verdict.md)。本卡原有內容與數字不改。
```
**cards/S2-Astra-method-design.md** (new paragraph right after the line-3 status):
```diff
+2026-11-14 結案：[S2-repair-exploration](S2-repair-exploration.md) 的 DEV208 結果中，對應本卡 §5 A／B／C 的 Mean／Tail／Tail+Gate 都未達 +2 個百分點的繼續投入門檻；S2 已於同日第二輪改判 NO-GO（使用者決定），見 [第二輪判定卡](S2-round2-verdict.md)。本卡原有內容與數字不改。
+
```

**New file `cards/S2-round2-verdict.md` (full text):**
```markdown
# S2 第二輪判定：改判 NO-GO，方法論文方向結案

2026-11-14。**判定：NO-GO（第二輪改判）。** 這是使用者的決定，不是依某一張卡 §3 自動產生的判定。各輪設計卡原有的判定與數字不改；本卡只記錄結案理由、保留項目與不再排程的工作。

## 1. 判定範圍

- **W1 的 GO 不變、不撤回。** gate 讓保護前 4 個 token 的收益縮減：留出集 W=256，headwise R=0.967 [0.966, 0.969]、elementwise R=0.933 [0.920, 0.943]；W2 的 P0′ 在 3090 重現到小數點後三位，數字經重算確認（`cards/S2-W1.md`、`cards/S2-W2.md` §11）。它證明的是現象，不是方法。
- 結案的是「以 gate 訊號做 KV 淘汰／repair」的方法論文方向。

## 2. 改判理由

1. **機制未過。** W2 的 P3（每個 head 的 sink 質量與保護收益的 Spearman）baseline ρ=0.265 [0.238, 0.278]，門檻 0.50；依 W2 卡整體 NO-GO（`results/s2-w2/perhead/part.json`、`cards/S2-W2.md` §11）。
2. **現象沒有實際收益。** gated 模型並沒有比較好壓縮：協定 B、W=256 各自用最佳策略時，gated − baseline 的絕對壓縮損失為 +0.008 [+0.004, +0.013]、+0.012 [+0.008, +0.017]，W=1024 縮到 0.001–0.003。這是看過資料後的計畫外分析（`cards/S2-W2.md` §11）。
3. **gval 下游失敗。** PG-19 壓縮額外 NLL 相對 Observed 改善 12.6%（headwise W=256；書集已在 W2 用過）（`results/s2-w3/summary.md`、`cards/S2-W3.md`）。但 RULER 4 任務 DEV 上，3 個模型 × W512／1024／2048 共 9 格，gval／gval_win 的較佳者全部落後同格最強 baseline 4.3–12.2 個百分點；headwise W1024 落後 CAOTE 11.48 個百分點，未達 W4 D0 的 +2 門檻（`results/ruler4k-dev/summary.json`）。
4. **repair 無可辨識效果。** RULER13 DEV208、headwise、W1024：Projected 37.98，同 runner 的 SnapKV 37.31（+0.67 個百分點，探索性配對區間 [−1.23, 2.66] 跨 0；11 題升、4 題降、193 題同分）。Mean 37.75、Tail 37.65、Tail+Gate 37.72：加 gate 的 Tail+Gate 沒有高過 Mean。不同 runner 的 Afterglow+SnapKV 89.30、Full KV 91.43（`results/repair-exploration/per-task-comparison.json`、`results/baseline-suite/development-report.md`、`cards/S2-repair-exploration.md`）。對照 `cards/S2-Astra-method-design.md` §5／§7 的停止規則：B（Tail）不優於 A（Mean），尾部機制未受支持；C（Tail+Gate）只比 B 高 0.07 個百分點，距 +2 個百分點的繼續投入門檻很遠（對同 runner SnapKV 只有 +0.42）。依該卡 §7 第 3 步「未通過就刪除 gate 敘事、換代理或終止方向」，gate 敘事放棄。
5. **撞題重查（2026-11-14）。** repair 的做法與「交換幾乎無效」的結論與 Trust the Mass（arXiv 2608.25230）重疊；gate × value 作為逐 token 訊號有 GateSight（arXiv 2610.14754）的反面結果（其 Appendix D：gate 幾乎不改 head 內排序）；「sink 被抑制的模型需要不同淘汰法」的敘事已有 ValueDiff（arXiv 2609.23314）；W_O 投影已有 LaProx（arXiv 2605.07234）與 CriticalKV（arXiv 2502.03805）。前兩篇與 CriticalKV 的已查範圍見 `cards/S2-repair-exploration.md`、`cards/S2-Astra-method-design.md` §2。
6. **現有 benchmark 改進空間不足。** headwise RULER13 W1024 上，Afterglow+SnapKV 89.30 離 Full KV 91.43 只差 2.14 個百分點 [0.93, 3.57]。LongBench native4k 已完成的 baseline／headwise matrix 中，最強的幾個方法與 Full 的配對差距區間跨 0（例如 headwise W1024 的 DropKV 0.68 [−0.86, 2.29]、Afterglow+SnapKV 1.10 [−0.56, 2.86]）；elementwise 與 Qwen3-8B 的 LongBench matrix 尚未完成（`results/baseline-suite/development-report.md`）。

## 3. 保留項目

- **W1 現象**，含「gated 模型不因此更好壓縮」，可作為分析／負結果短文或 workshop 的素材（`cards/S2-W1.md`、`cards/S2-W2.md` §11）。
- **baseline 評測基礎設施**可供其他題目沿用：凍結的 DEV 矩陣（`results/baseline-suite/`、`cards/S2-B3.md`）、KVpress 接入（`baselines/README.md`）、已下載的模型與資料（README「已備妥的資產」「開發資料」兩節）。

## 4. 未做、不再排程

- W4 的 C1／C2／E1–E3 確認實驗（`cards/S2-W4.md`）。
- 六設定比較：Afterglow／KeyDiff 各自 ± Projected（`cards/S2-repair-exploration.md` 最後一節、`configs/next-comparison.json`）。
- repair 移植到 8B。

## 5. 其他

- `paper/` 的論文工作稿已過時：仍是方法論文的敘事（`paper/README.md` 標示更新於 2026-11-13），本次未更新。
- 本次只改文件；程式、結果檔與既有卡片的判定、數字都不動。
```

## 2. Overview repo `/work/ChihChungHsu/cactus8603/icml/ICML2027`

```
 README.md                  | 7 ++++---
 go-topics/README.md        | 2 +-
 go-topics/S2-詳細.md       | 2 ++
 go-topics/S2.md            | 2 ++
 4 files changed, 9 insertions(+), 4 deletions(-)
```
(The untracked `.claude/` and `CLAUDE.md` were already there before I started; I did not touch them.)

**README.md** (other topics untouched):
- **W1 GO table, S2 row:** appended to the end of the "通過依據" cell: `**後續：2026-11-14 第二輪改判 NO-GO，結案**（使用者決定），W1 結果不撤回，見下方待判定表。`
- **"兩題 GO 的檢查", S2 bullet:** appended: `**後續（2026-11-14）：** W2 的 P3 ρ = 0.265，未達 0.50，W2 整體 NO-GO；撞題重查後，repair 與 gate × value 的做法都已有接近工作。第二輪改判 NO-GO，結案，見下方待判定表。`
- **W2 design-card table, S2 row:** appended to the last column: `**已跑完（2026-11-13）：整體 NO-GO**（P3 ρ = 0.265）；2026-11-14 第二輪改判 NO-GO，結案`
- **"待判定" table:** new S2 row placed above SD, following the SD and P3 wording:
  `| **S2** | Gated Attention 與 KV sink 保留 | **2026-11-14 第二輪改判 NO-GO，結案**（使用者決定，無新 job）。W1 GO 不撤回：gate 讓保護前 4 個 token 的收益縮減（留出 W=256，R = 0.967／0.933），但這是現象，不是方法。改判理由：W2 的 P3 ρ = 0.265 [0.238, 0.278] < 0.50（W2 整體 NO-GO）；gated 模型沒有更好壓縮；gval 的 PG-19 壓縮額外 NLL 改善 12.6%，但 RULER 4 任務 DEV 9 格全部落後最強 baseline 4.3–12.2 個百分點；repair 的 Projected 37.98 對同 runner SnapKV 37.31，區間跨 0，加 gate 沒有高過 Mean；撞題重查有 Trust the Mass、GateSight、ValueDiff、LaProx、CriticalKV 等接近工作；現有 benchmark 上最強 baseline 已接近 Full KV。 [第二輪判定卡](https://github.com/cactus8603/icml27-s2-gated-sink-kv/blob/main/cards/S2-round2-verdict.md) | 結案。W4 確認實驗（C1／C2／E1–E3）、六設定比較、repair 移植 8B 不再排程。W1 現象可作分析／負結果短文素材，baseline 評測基礎設施供其他題目沿用。 |`
- **Snapshot section:** unchanged. It already says S2 has no snapshot folder.

**go-topics/README.md:** in the S2 row, the "第二輪狀態" cell changed from `W2 卡與程式已寫好，3090 跑，未開跑` to `**2026-11-14 第二輪改判 NO-GO，結案**（使用者決定）。W2 整體 NO-GO（P3 ρ = 0.265）；W1 結果不撤回。見 [第二輪判定卡](…/cards/S2-round2-verdict.md)`.

**go-topics/S2.md and go-topics/S2-詳細.md:** the same lines were added to the opening blockquote of each, following the SD.md and SD-詳細.md pattern:
```
>
> **狀態：2026-11-14 第二輪改判 NO-GO，結案**（使用者決定）。W1 的 GO 不撤回；W2 的 P3 未過（整體 NO-GO），後續 gval 下游與 repair 都沒有可辨識的收益，撞題重查也有接近工作。理由與出處見 [第二輪判定卡](https://github.com/cactus8603/icml27-s2-gated-sink-kv/blob/main/cards/S2-round2-verdict.md)。以下內容保留作紀錄，其中的第二輪設計與下一步不再排程。
```
The line goes after the first line of the blockquote: in S2.md that is "2026-11-13 整理…", and in 詳細 it is "整理日期…", just before "主要來源".

## 3. Places where the brief does not match the source files (I followed the source)

1. **Point 4, the stop rule.** The Astra card's stop rules (§5) are "B 不優於 A 則尾部機制未受支持" and "C 不優於 B 則放棄 gate 敘事". `cards/S2-repair-implementation.md:127` defines A=Mean, B=Tail, C=Tail+Gate. Tail+Gate (37.72) is in fact 0.07 points higher than Tail (37.65), so the rule that drops the gate narrative was **not** literally triggered. What was triggered:
   - B is not better than A (Tail 37.65 < Mean 37.75).
   - The pilot did not reach the +2-point threshold for further investment.
   - Under §7 step 3 ("未通過就刪除 gate 敘事、換代理或終止方向"), that drops the gate narrative.

   The card cites it this way. The fact you gave ("Tail+Gate 37.72 沒有高過 Mean 37.75") is correct and is kept.
2. **Point 6, LongBench.** "各方法與 Full 的差距都在誤差內" does not hold for every method:
   - In the headwise LongBench W1024 matrix, the paired CIs for random0, TOVA, KNorm, KeyDiff, ValueNorm and ValueDiff do not include 0.
   - At W512, most methods' CIs do not include 0.
   - What holds is that the strongest few methods' CIs include 0: DropKV 0.68 [−0.86, 2.29] and Afterglow+SnapKV 1.10 [−0.56, 2.86].
   - Also, only the baseline and headwise LongBench matrices are finished; elementwise and Qwen3-8B are still pending, and the report status is in_progress.

   The card is written in this narrower form.
3. **Point 5, the prior-work recheck.** LaProx (2605.07234) does not appear anywhere in either repo. The repos also never describe ValueDiff with the narrative "sink 被抑制的模型需要不同淘汰法"; they only record it as a value-deviation scorer. The "交換幾乎無效" conclusion is also not written explicitly in the repo's Trust the Mass notes. I wrote point 5 as the 2026-11-14 recheck you supplied, and pointed only Trust the Mass, GateSight and CriticalKV to the repo files where they were already reviewed.

Everything else matches the source files:
- ρ=0.265 [0.238, 0.278] with P3_verdict NO-GO.
- I recomputed the 9 RULER cells from summary.json: −4.30 to −12.19, and headwise W1024 −11.48 against CAOTE.
- Repair: 37.98 / 37.31 / [−1.23, 2.66] / 11 up, 4 down, 193 tied / 37.72 / 37.75.
- Afterglow 89.30, Full 91.43, a gap of 2.14 [0.93, 3.57].
- W3: 12.6%.

## 4. Format choices and open questions

- **New card name.** I kept `S2-round2-verdict.md`. The repo already mixes round names like -W1…W4 with descriptive names like repair-exploration and Astra-method-design. This is not an experiment round, so there was no -W name to use. The card does not follow the 11-section design-card template; its sections are 判定範圍／理由／保留／未做／其他, in the plain style of the B-series cards.
- **Topic-repo summary table.** Under the CLAUDE.md rule, a round's verdict can only be set by its card's §3. So I left the existing GO / NO-GO / 待判定 values untouched and added a new "S2 第二輪判定" row. I also appended "（2026-11-14 結案，未執行）" to the "下一步" cell of the W3 and RULER DEV rows.
- **README banner position.** It goes right after the H1, above the original intro paragraph. The "目前狀態" section below it, including "六設定尚待實作", is untouched; the banner already says that content is kept as a record.
- **Topic repo's own `go-topics/S2.md` and `S2-詳細.md`: not changed.** These are the files that actually carry a "歷史來源快照" warning, but `go-topics/upstream-source.json` records each one's `destination_sha256`, and editing them would break those records. If you want a pointer there anyway, the topic repo's `go-topics/README.md` "來源之後的進度修正" table is a better place for one more row. I have not done this.
- **The overview repo's go-topics S2 files have no "歷史快照警告".** So I followed the SD pattern and added a status line to the opening blockquote.
- **Not changed:** the overview repo's `experiments/README.md` summary table. CLAUDE.md lists it as one of the four places a result is recorded, but it was not in your list. Decide whether to add a line there when you commit.
- **Link not live yet.** The overview repo's links to the verdict card point to GitHub `main`, so they will 404 until the topic repo is pushed.
- **Side finding, not edited.** In `results/s2-w3/summary.md`, the elementwise line also prints "rel 12.6%", but `cards/S2-W3.md:86` says elementwise is 7.5%. The summary file probably printed the headwise ratio on both lines. This does not affect the verdict.
