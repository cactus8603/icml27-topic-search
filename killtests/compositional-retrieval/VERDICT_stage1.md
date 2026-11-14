# VERDICT stage1：compositional-retrieval（E0／E1，P1）

判讀者：獨立判讀者（未參與實作）｜日期：2026-11-14｜依據：PREREG.md（凍結於 commit 8dd910a）、configs/params.json、results/aggregate.{md,json}、results/ht/、logs/cr-stage1-26274.out、scripts/
只讀程式與結果，並在 CPU 上用 nice 獨立重算（暫存於 `/work/ChihChungHsu/cactus8603/icml/tmp/cr_verdict/`）。沒有修改程式或結果檔，沒有送 job，也沒有 commit。

## 結論：**FAIL 成立（題目停損）**

P1 的兩條路徑都明確失敗，而且離門檻很遠。程式沒有找到會改變結論的 bug。
P2 未能測試，原因是 stage2 沒有送。依 PREREG §5「任一主指標 FAIL → 停損」，整體判定不受 P2 影響。

## 1. 判準是否在結果之前凍結
- 8dd910a 的 commit 時間是 2026-11-14 20:28:24。job 26274 在 20:28:34 開始（`logs/env_freeze_cr-stage1.txt`），當時 `git_head=8dd910a`。
- `git diff 8dd910a -- compositional-retrieval/` 只有 `results/gates/g1_unit.json` 有差異，這是本次 job 重新產生的輸出（32 → 43 項測試）。PREREG.md、params.json 和 scripts/ 都與凍結版完全相同。
- env_freeze 記錄的 `git_dirty=1` 來自其他題目尚未 commit 的檔案，本目錄的程式碼不受影響。

## 2. 數字、門檻、α 與判定邏輯是否對得上 PREREG
| 項目 | PREREG／params | 實際 | 對得上 |
|---|---|---|---|
| α | P1 每條路徑單尾 0.0125，取 1.25／98.75 百分位 | `analyze_ht` 傳入 `alpha_one_sided_per_path=0.0125`，輸出 `lo_1s`／`hi_1s` | 是 |
| 路線 a | CLIP B/32 + llr_sum，COCO(Qwen)，下界 ≥ 45 | 點估計 23.18，[lo_1s, hi_1s] = [19.06, 27.50]，n = 509 | 是 |
| 路線 b | CS-CLIP llr_sum − 同流程 CS-CLIP holistic 的配對差：點估計 ≥ 5，且下界 > 0 | 點估計 −29.47，[−35.36, −23.58] | 是 |
| FAIL 條件 | a 的上界 < 45 **且** b 的上界 < 5 | 27.50 < 45，且 −23.58 < 5 | 是 → FAIL |
| 閘門 | G1–G3 全過 | 全過（見下方） | 是 |

- 我用 row_cor_flags 獨立重做 bootstrap（seed 20261114／20261116，10000 次），1.25／98.75 百分位與上表**逐位相同**。
- `decide_p1` 的邏輯與 §5 文字一致：PASS 是 a 或 b 任一成立；FAIL 要兩條路的上界都低於各自門檻；其餘為灰區。
- 整體判定依「FAIL 優先」：P1 FAIL 加上 P2 未能測試，結果是停損，與 §5 一致。
- 結論對 α 不敏感：路線 a 的上界比門檻低 17.5pp，路線 b 的上界比 0 低 23.6pp。

閘門原始數字：
- G1：單元測試 43/43 通過；與官方腳本逐列比對，最大絕對差 2.98e-7；細節抽取覆蓋率最低 0.984。
- G2：CS-CLIP 的 COR 為 58.15／50.44／42.19，論文值 58.2／50.4／42.2。
- G3：bootstrap 半寬最大 4.63pp；改 batch 重跑的最大差 2.38e-7。

## 3. 是假設不成立，還是實作 bug？逐項核對
我用自寫程式重算，不經由 `common.pmi`。PMI 池由三個子集的影像依 sha 去重得到，共 1379 張；logit scale τ：CLIP 為 100.0，CS-CLIP 為 100.23。

| 檢查項目 | 結果 |
|---|---|
| LLR 方向 | `pmi = τ·s(I,c) − logmeanexp_pool(τ·s(I',c))`，`llr_sum` 為 (pa+pp, pa, pa+pm)。正確細節的 PMI > 0 時分數上升，方向正確。CLIP 的 P(PMI(d+) > PMI(d−)) = 69.5%，CS-CLIP 為 81.5%，正負細節的相對順序大致正確。 |
| PMI 池 | 依 sha 去重，1379 張，沒有使用標籤，與 §3 一致。同子集池的 COR 為 24.75，結論相同。 |
| anchor 抵銷 | 獨立重算的 COR 為 23.18，與 P(PMI(d+) > 0 > PMI(d−)) = 23.18 逐列一致。 |
| 細節抽取 | 抽查 12 列原文，d+／d− 都是正確的新增片段，例如 "counter" → "and room" 對 "and a bathroom"、"horse" → "is with trainer" 對 "is without trainer"。coco_qwen 只有 8 列不合格（half_truth 改寫了 anchor），與 coverage 0.984 吻合。 |
| COR 定義 | `cor_flags` 使用嚴格 `>` 串接 full > anchor > half，平手算錯，單元測試也涵蓋。獨立重算的 holistic COR 為 18.86（CLIP）與 58.15（CS-CLIP），等於論文值。 |
| unit_min、unit_sum 全為 0.0 | **這是構造使然，不是 bug。** unit_min：要 min(ca, cd+) > ca 嚴格成立，數學上不可能。unit_sum：要 ca + cd− < ca，也就是 cos(I, d−) < 0，但實測 cos(I, d−) 最小值為 0.145（CLIP）／0.145（CS-CLIP），永遠是正的。這兩個比較用的聚合器在 COR 定義下注定為 0，屬於次要指標的設計缺陷，不影響 P1，PREREG 也規定它們不參與判準。 |

**失敗來源是假設本身：** 加入錯誤細節並不會讓 PMI 變成負的。
- CLIP 的 P(PMI(d−) < 0) 只有 **40.5%**，CS-CLIP 只有 **35.8%**。PMI(d−) 的中位數為 +0.66（CLIP）與 +1.23（CS-CLIP）。
- Half-Truths 的錯誤細節是在語境中合理的替代品，例如 "small bus" 對 "small train"、"lunch food" 對 "breakfast food"、"bedroom" 對 "living room"。這些片段去掉語境後與影像仍然正相關。
- 因此不論 d+ 表現如何，COR 都有上限 P(PMI(d−) < 0)：CLIP 為 40.5% < 45，所以路線 a 在構造上就到不了門檻；CS-CLIP 為 35.8%，遠低於 holistic 的 58.2，所以路線 b 也不可能成立。
- PREREG §0 的假設是「錯誤細節應使分數下降」，§7 也事先聲明「細節單元去脈絡，0 門檻是構造而來」。實測正是這個機制失效，屬於假設被否證，不是實作偏差。
- 描述性補充：CLIP 的 entity 類 LLR COR 為 38.2，holistic 為 26.0；relation 類 LLR 為 8.2，holistic 為 11.8。entity 類有局部改善，但仍低於 45，不影響判定。

**判定：沒有找到 bug，不應記為「未能測試」。**

## 4. P2「未能測試」的原因，以及 stage2 還值不值得跑
- 原因確認：stage2 沒有送出。`results/e2/` 是空的；`SUBMITTED.log` 只有 `retrieval stage1=26274`，沒有 stage2；`stage1_ht.sbatch` 的註解也寫明此時 P2 會顯示「未能測試（資料不足）」。
- 對判定的價值：**沒有。** 依 §5，只要 P1 FAIL 就停損。P2 不論結果是 PASS、FAIL 還是灰區，都改變不了整體結論。REVIEW 建議 1 也指出，P2 在構造上幾乎必然 PASS，證據力很弱。
- 建議：**不跑 stage2**，省下 GPU 與時間。如果使用者想把 E2 的稀釋差當描述性資料存檔，可以跑（GPU 不到 1 小時），但必須標成「停損後探索性」，不得用來推翻 P1。

## 5. 修正與重跑
不適用，沒有判定為 bug。有一個建議（非必要）：在最終報告中註明 unit_min／unit_sum 在 COR 定義下注定為 0，作為次要指標的設計缺陷；它們不參與判準，不需要重跑。

## 關鍵原始數字（coco_qwen，n = 509）
| 指標 | CLIP B/32 | CS-CLIP |
|---|---|---|
| holistic COR | 18.86 | 58.15 |
| llr_sum COR（全 HT 池） | **23.18** [19.06, 27.50] | 28.68 |
| llr_sum COR（同子集池） | 24.75 | 27.11 |
| CS-CLIP llr_sum − holistic | — | **−29.47** [−35.36, −23.58] |
| P(PMI(d+) > 0) | 77.8% | 91.4% |
| P(PMI(d−) < 0)，即 COR 的上限 | 40.5% | 35.8% |
| P(PMI(d+) > PMI(d−)) | 69.5% | 81.5% |
| holistic P(anchor > half) | 34.0% | 76.4% |
