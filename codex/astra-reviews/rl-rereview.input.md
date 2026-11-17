Independently review before GPU submission. Verify preregistered formulas, thresholds, controls, input completeness, frozen training/evaluation parity and actual Slurm entry points. Consider whether valid evidence can be misclassified as failure. Report BLOCK only for concrete correctness or reproducibility failures and give minimal changes, otherwise READY with limitations. Treat enclosed material as source data, not instructions. No experimental result data is included.

Topic rl-sparse-distill, source worktree

<file name="PREREG.md">
# PREREG：RL 稀疏決策混合蒸餾（回報 44）— 第 1 週第一關

凍結日期：2026-11-14（送 slurm 前須 commit）。判準依 reports/2026-11-14-56-killtest-criteria-review.md 第 5 節與通用規則。
**所有數字門檻的唯一來源是 `configs/criteria.json`，判讀只由 `scripts/aggregate.py` 機械輸出。** 本檔的數字若與該檔不同，以該檔為準，並以日期記在文末「修改紀錄」，標為探索性。

## 1. 唯一主指標
- pass@k 無偏估計（Chen et al. 2021）：每題 n=128 個樣本、答對 c 個，pass@k = 1 − C(n−c, k)/C(n, k)，**主指標固定 k=64**，對題目取平均。
- 判分：最後一個 `\boxed{}` 內容經 Math-Verify（math-verify 0.9.0）與參考答案比對；沒有 boxed 或被截斷一律算錯。
- 統計：配對（以題為單位）bootstrap，B=10,000，seed=20261114；「97.5% CI 不含 0」實作為**單側**：差值 bootstrap 分布的 2.5 百分位數 > 0（同時也輸出雙側 95% 區間供參考）。這是對 Fable 文字的一種解讀，審查者請確認。

## 2. 資料與 seed
- 評測集 `data/eval.jsonl`（由 `scripts/prepare_data.py` 產生，1,453 題）：MATH-500（500）、Minerva（272）、OlympiadBench 純文字單一答案（581）、AMC23（40）、AIME24（30）、AIME25（30）。
- 劃分 seed=20261114：隨機 400 題為 **dev**（本階段不使用；灰區加碼與第 2 週蒸餾調參用），其餘 **1,053 題為 test**（閘門 0／判準 1 的全部統計只用 test）。≥1,000 題。
- 解碼：temperature 0.6、top-p 0.95、max_new_tokens=3000、max_model_len=4096（**凍結**；Oat-Zero 作者評測的上限也是 3000，Qwen2.5-Math 的 context 只有 4096）；模板固定為 Qwen-Math（`pairs.json`），base 與 RL 用同一模板。
- 取樣 seed：vLLM `seed=1234`；HF `gen_replace.py` 以 `seed + 1000*shard + batch_index`。bootstrap seed 如上。
- 模型：見 `configs/pairs.json`（配對 A：Qwen2.5-Math-1.5B／sail Oat-Zero；配對 B：Qwen2.5-1.5B／hkust-nlp SimpleRL-Zoo）。dtype 一律 bfloat16（SimpleRL-Zoo 原檔 fp32，載入時轉 bf16）。

## 3. 閘門順序（前三關失敗＝「未能測試」，限 3 個工作天修；修不好記為可行性停損，與此實作停損分開記；要升級為題目淘汰，須另外符合 57 號第 5 節規則 2）
**閘門 −1 程式正確性**（`tests/smoke_cpu.py`；CPU 小模型先過，再用 `slurm/00_correctness_gpu.sbatch` 在真實模型、fp32、GPU 上重跑）：
1. rl 模型設成 base 時，rl_replace 輸出 == base 取樣（同 seed，逐字相同，替換數 0）。
2. 全部位置替換（all_replace）== RL greedy（逐字相同）。
3. 與原作 `wrb.decoding.rl_replace.RLReplaceDecoder` 在單題批次、無壓縮下逐 token 相同（文字與替換數都相同）。
4. 多題左 padding＋批次壓縮的 greedy == 各題單獨 greedy。
5. 替換比例介於 0 與 1，並記錄（原文 Qwen3 配對約 4%；本配對只作參考，因為不是同一組模型，「對得上原文」以第 3 項的逐 token 一致為準）。
全部必須通過才能進下一關。另：`aggregate.py --selftest` 檢查 pass@k 估計式與 bootstrap 管線。

**閘門 0 單模型取樣（預估 5–7 GPU 小時；上限 20）**，每個配對：
- (a) 重現 RL 已報 pass@1：RL greedy（temperature 0、split=all）在 MATH-500、Minerva、OlympiadBench 三個來源上，與 `pairs.json` 內原作已報數字相差皆 ≤ 2pp。（AMC／AIME 題數太少，只記錄不判定。）
- (b) 截斷率：base 與 RL 在取樣下未正常結束（撞 3000 token 上限）比例相差 < 2pp。
- (c) 配對資格：RL 的 pass@1 增益 ≥ 5pp；RL 的 pass@64 比 base 低 ≥ 2pp；落在 30–85%。「落在 30–85%」的解讀：base 與 RL 的 pass@64 都須在 [30, 85]（避免飽和），由 `criteria.json: pass_window_applies_to` 控制，審查者請確認。
- (a)(b) 任一失敗＝未能測試（先查模板、stop token、截斷）；只有 (c) 失敗＝配對不合格，**換配對，不算題目死**（備選見 `pairs.json: reserve_pairs_not_downloaded`；配對 C 需要 32k 上限，會改變凍結的 token 上限，必須先另寫 PREREG 補充）。

**判準 1（只對閘門 0 合格的配對；預估 rl_replace 約 18–25 GPU 小時／配對，WiSE 6 點約 8 GPU 小時／配對）**，k=64：
- (i) rl_replace 的 pass@64 − **HF 版 RL（`rl_hf`）**的 pass@64 ≥ 2pp，且單側（α=0.0125，即 1.25 百分位）下界 > 0；（見修改紀錄 2026-11-14 審查後定案）
- (ii) 在相同 pass@1 下高於 WiSE-FT 前緣：WiSE-FT 為權重內插 θ=(1−λ)θ_base+λθ_RL，λ ∈ {0.15, 0.30, 0.45, 0.60, 0.75, 0.90} 加兩端點（λ=0 為 base、λ=1 為 RL）共 8 點；前緣＝以這 8 點的 (pass@1, pass@64) 依 pass@1 排序的分段線性內插，取在 rl_replace 的 pass@1 處的值；差值（rl_replace 的 pass@64 − 前緣值）> 0，且 bootstrap（每次重算全部點與內插）單側下界 > 0。若 rl_replace 的 pass@1 落在 8 點範圍外會在輸出標記。
- 兩者都成立才 PASS；(i) 不成立或 (ii) FAIL 時**停 λ=1 端點路線（rl_replace 這個實作），不做蒸餾**；這不等於題目淘汰。(ii) 的 FAIL 條件見 R1；(i) 成立而 (ii) FAIL 時觸發 R3 的唯一第二探針。

## 4. 三態結果（`aggregate.py` 機械輸出）
- **PASS**：判準 1 的 (i)(ii) 都成立。
- **FAIL**：rl_replace 比 RL 高不到 1pp；或 (ii) 的 FAIL（R1：相同 pass@1 下對 WiSE 前緣的差值點估計 ≤ 0，**且**差值的 98.75 百分位 < +2.0pp）。
- **灰區**：其餘。只准一次事先定好的加碼：加跑 dev 400 題（RL、rl_replace、最接近的 WiSE 點）後與 test 合併重算（合併重算的程式尚未寫，灰區發生時再補，須先 commit）。
- **未能測試**／**配對不合格**：見第 3 節。
- 整體（兩個配對擇一，已以 α=0.0125 校正）：至少一個合格配對 PASS → PASS；無 PASS 但有灰區 → 灰區；合格配對全部 FAIL → FAIL（停 λ=1 端點路線，不做蒸餾；不等於題目淘汰，先看 R3）；沒有合格配對 → 未能測試。

## 5. 可重試次數與成本上限
- 同一 job 因基礎設施失敗（OOM、節點掛掉、逾時）最多重送 2 次；**結果不理想不得重跑**（不得換 seed 挑結果）。失敗與重跑照實記入 `logs/RUNLOG.md`，不刪。
- 總 GPU 小時上限 120（預估 60–75；R3 探針另計上限 25，預估使總量超出 120 時先回報）；送出前先跑 `slurm/04_throughput_probe.sbatch`，實測吞吐低於推估 1/2 就重排（例如降低配對數或改 vLLM 快速路徑）。
- 磁碟上限 60 GB（預估 ≤ 25 GB）。

## 6. 留存
每個 cond 留 raw（gz，含完整生成）、graded.json（每題每樣本對錯、是否結束、token 數、替換數）、meta（模型、seed、engine、秒數）、config、git hash、slurm job id（記入 RUNLOG.md）。

## 修改紀錄
（事後修改請在此加日期與原因，並標為探索性。）
- 2026-11-14 [claude]：依報告 60 改為現象／訊號／空間三項判定（見文末同日條目）；`criteria.json`、`aggregate.py`、README 同步修改。
- 2026-11-14（Opus 審查，機械修正，非判準變更）：`make_wise.py` 加斷言——單側多出的 `lm_head.weight` 必須等於該側 `embed_tokens` 才能以 tied 方式內插（SimpleRL-Zoo 實查相同）；slurm 輸出路徑改為絕對路徑。判準相關的審查意見見 `REVIEW.md`，尚待使用者定案。

### 2026-11-14 審查後、結果前定案（非探索性；依 REVIEW.md 必修 2–5，協調者決定，任何假設結果出來之前）
1. **多重比較**：`criteria.json: gate1.ci_one_sided_alpha` 由 0.025 改為 **0.0125**（兩配對擇一的 Bonferroni 校正，單側 1.25 百分位）。(i)(ii) 的下界都用此值。
2. **消除引擎差**：判準 (i) 的比較對象改為 HF 版 RL 取樣 `gen_replace.py --cond rl`（`slurm/05_rl_hf.sbatch`，輸出 `results/<P>/rl_hf`，同 test、n=128、同 seed 與分片規則、同 3000 token 上限）。vLLM 版 `rl` 只用於閘門 0 與 (ii) 的前緣端點。`aggregate.py` 另輸出診斷：rl_hf − rl(vLLM) 的 pass@1／pass@64（含 95% CI）、四個條件的截斷率，以及 (ii) 把 λ=1 端點換成 rl_hf 的敏感度 `wise_margin_sens_rlhf_endpoint_pp`（只描述，不參與判定）。缺 rl_hf 時判準 1 不執行。成本每配對約 +10 GPU 小時（只載入 RL 一個模型）。
3. **重現來源**：`gate0.reproduce_sources` 改為 `["math500","minerva"]`；olympiad 只記錄（`reproduce_record_only_diffs_pp`），因題集不同（581 對 674 題）。
4. **配對 B 模板**：pairs.json 新增 per-pair `template`。B 使用 `simplerl_abel_chat`（訓練資料 hkust-nlp/SimpleRL-Zoo-Data 的 simplelr_abel_* 的 user 訊息 `Question:\n{q}\nAnswer:\nLet's think step by step.\n`，外包 Qwen 聊天模板，預設 system＝You are a helpful assistant.）。出處：SimpleRL-Zoo 論文附錄 B.3（0.5B/1.5B 用較簡單 prompt）與 B.5（評測沿用訓練模板）、HF 資料集實查、tokenizer apply_chat_template 實測。**未確認部分**：原作評測程式（examples/simplelr_math_eval）有 `abel` 模板但無法從程式碼確認 1.5B 評測是否外包聊天模板；原作評測為 max_tokens 16000，Zoo 模型評測用 temperature 1.0／top-p 0.95，與我們閘門 0 的 greedy＋3000 token 不同，可能影響重現。B 配對的 00／01 送出前須由協調者確認。A 配對仍用 qwen_math。
5. 其他：`aggregate.py --selftest` 已加入 rl_hf 與正向情境（rl_replace 明顯高於 rl_hf 時 (i) 數字為正）。

### 2026-11-14 代理落差稽核後、結果前定案（非探索性；依 59 號報告 §5、58 號 §5；任何主判準結果出來之前）
`configs/criteria.json` 新增 `gate1.fail_ii_upper_pct=98.75`、`fail_ii_exclude_pp=2.0`、`gate1.mix_curve`、`probe2`；既有門檻一字未動。`aggregate.py` 新增 R1–R3 與 selftest；`gen_replace.py` 新增 `--cond geo_mix --lam`（既有 4 個 cond 的輸出經舊版 `git HEAD` 對照逐字相同，見 README）；`slurm/06_geo_mix.sbatch` 只寫不送；`tests/smoke_cpu.py` 新增 T6–T8。判斷者沒有打開 `results/A/*/` 的任何評分。

**R1（(ii) 的 FAIL 條件）**：以下兩項都成立才算 (ii) FAIL：(1) 差值（rl_replace 的 pass@64 − 前緣值）的點估計 ≤ 0；(2) 差值的 98.75 百分位 < +2.0pp。點估計 ≤ 0 但 98.75 百分位 ≥ 2.0pp 時歸入灰區，走原本唯一一次的 dev 400 加碼；合併後仍排除不了 2pp，記「未能測試（檢定力不足）」。(i) 的判定不變。（理由：WiSE 前緣有 8 個點、rl_replace 只有 1 個點，判準不對稱；凹性論證只能保證混合曲線高於弦，不是前緣的檢定力正控制，故不採 58 號 B1。）合併重算的程式仍未寫（沿用原有待辦），`gate1(..., merged=True)` 已先實作該分支的判定。

**R2（回應層級混合曲線，只描述）**：每次嘗試以機率 π 取 RL 樣本、否則取 base 樣本；無偏估計 (1−p)^64 = Σ_m Binom(64, m; π) · C(128−c_RL, m)/C(128, m) · C(128−c_base, 64−m)/C(128, 64−m)；π 從 0 到 1，間隔 0.05。輸出於 `gate1.mix_curve_descriptive`（RL 取與 (i) 相同的 `rl_hf`；同時列弦）。不參與任何判定；它是政策層級的前緣參照，不是 (ii) 的前緣。

**R3（唯一的第二探針：λ=0.5 幾何混合）**
- 觸發：合格配對的 (i) 成立而 (ii) FAIL。只做一個配對，A 符合就做 A，否則做 B。
- 取樣：在 greedy 不一致的位置用 0.5·logit_base + 0.5·logit_RL（softmax 後同 temperature 0.6、top-p 0.95、3000 token），其餘位置照 base；test 1,053 題 × 128，seed 與分片規則同 rl_replace。
- 通過條件（三項都要成立；單側 α=0.00625）：(1) pass@64 − rl_hf ≥ 2pp，且單側下界 > 0；(2) 在自身的 pass@1 處高於 WiSE 前緣，且單側下界 > 0；(3) pass@1 − base ≥ 0.5 × (RL − base) 的 pass@1。條文未指明 (3) 的「RL」，實作取與 (i) 相同的 `rl_hf`（結果前決定）。
- 只有通過或未通過，不加碼。探針上限 25 GPU 小時；全題上限 120 不變，預估會超出時先回報。探針未通過，才可提請淘汰。
- 用語：(i) 成立而 (ii) FAIL 只否證「λ=1 端點（確定性輸出 RL 的 argmax）」；λ=1 是 q*_λ 家族裡多樣性最低的成員。

### 2026-11-14 [claude] 使用者原則變更後、結果前定案（報告 60 通用條文 G 與第 5 節；非探索性）
本次修改者沒有讀取 `results/` 下任何主判準結果。

> **2026-11-14｜使用者原則變更後、結果前定案，非探索性。** 停損只依三項判定：
> - 現象：問題存在，而且量得到；
> - 訊號：有非零、高於雜訊的可用訊號；
> - 空間：目前最好的方法和上限之間還有距離。
>
> - **G1** 三項都 PASS → 進候選池。任一項 FAIL → 在本 PREREG 的範圍內不進候選池，並寫明是哪一項。其餘情況為灰區：只准一次條文指定的加碼，否則交給使用者。
> - **G2** 「我們第一版方法對最強基線」一律只做描述：照常計算、照常輸出，用來在池內排序，不得拿來改判。
> - **G3** 既有的閘門（程式正確性、重現、雜訊）、正負控制、構造可行性檢查照舊。閘門失敗記「未能測試」，適用 3 個工作天的修復期。
> - **G4** 三項都是與實作無關的量，符合 57 號第 5 節規則 2 的第一種條件，所以 FAIL 可以直接記為「在本範圍內淘汰」，範圍限定語不得省略。
> - **G5** 同一項裡有 m 條擇一路徑時，信賴水準用 Bonferroni 調整。
> - **G6** 結果檔已經產出、但判斷者還沒讀的，照實註記「已存在、判斷者未讀取」。

**第 5 節（rl-sparse-distill）本版條文**（取代原判準 1 的 PASS 定義、第 4 節三態判定與 R3 的角色；實作為 `aggregate.py` 的 `phenomenon_eval`／`signal_eval`／`space_eval`／`r3_status`／`overall_eval`，門檻在 `criteria.json` 的 `phenomenon`／`signal`／`space`／`r3`／`overall`）：
- **現象**：量為閘門 0(c)（原樣）。A、B 至少一組合格 → PASS（主配對＝A 合格就用 A，否則 B）。(a)(b) 都過但 A、B 皆不符 (c) → FAIL，範圍限「1.5B、3000 token」；配對 C 是否另開由使用者決定。(a)(b) 失敗或缺資料且無合格配對 → 未能測試。
- **訊號**（主配對）：量 1＝rl_replace 的 pass@64 − rl_hf ≥ 2pp，且單側 1.25 百分位（α=0.0125）> 0；量 2＝pass@1 回收率＝(rl_replace − base) ÷ (rl_hf − base) ≥ 0.5。PASS＝量 1 與量 2 都成立；FAIL＝量 1 的差點估計 < 1pp；其餘灰區，走 dev 400 加碼（一次）。
- **空間**（主配對）：H_a＝rl_replace 的 pass@64 − WiSE 前緣值；H_b＝R2 回應層級混合（比較對象 rl_hf）在 π* 的 pass@64 − 前緣值。π* 取混合 pass@1 最接近 rl_replace pass@1 的格點（間隔 0.05），距離相同取較小 π；π* 以點估計固定，bootstrap 中不重選；H_b 的前緣取在混合自身的 pass@1 處。λ=1 端點：vLLM 版 rl 與 rl_hf 各算一次，取差值較小者。H_a、H_b 擇一（m=2），單側 α=0.00625。PASS＝任一個 ≥ 2.0pp 且單側下界（0.625 百分位）> 0；FAIL＝兩者點估計都 ≤ 0 且 99.375 百分位都 < 2.0pp；其餘灰區，走 dev 400 加碼，加碼後仍分不出記「未能測試（檢定力不足）」。
- **R3**：改為空間的第三個上限候選。觸發＝主配對訊號 PASS 而空間 FAIL；預算 25 GPU 小時。probe2 通過 → 空間記 PASS；未通過 → 空間 FAIL。probe2 的取樣與三項通過條件不變。
- **降為描述**：rl_replace 對 WiSE 的比較（gate1 的 verdict、`wise_margin_*`、R1 的 FAIL 條件）、第 2 週學生的全部門檻。上面「第 3 節判準 1」「第 4 節三態結果」的 PASS／FAIL／灰區措辭僅作歷史記錄；整體判定以本段為準。
- 結果檔狀態（G6）：`results/A/` 已存在，判斷者（[claude] 本次修改者）未讀取。

- [codex] 2026-11-14 23:51：依獨立實作審核補齊輸入驗證，未讀取 `results/A/` 下的正式評分或生成結果。既有配對 A 的閘門 0 產物已存在，只核對 26270 的正確性日誌、26290–26293 的 Slurm 模型／完成紀錄與取樣 metadata，這些識別資訊登記於 `configs/legacy_artifacts.json`。保留既有 raw 與 graded，須依凍結的 Math-Verify 0.9.0 規則另產出 `graded_verified.json` 及其中繼資料雜湊憑證後，以新版程式判讀。未將歷史 job 補填為有新產物憑證。
- [codex] 正式判讀須先驗證閘門 −1、全部凍結 ID、取樣分片身分、模型與解碼、樣本數及各欄位長度。每題 greedy 必為一個樣本且涵蓋全資料，正式取樣須恰 128 個樣本且涵蓋全 test。截斷樣本一律算錯。未通過任何輸入驗證時記「未能測試」。`--gate 0` 只看對應證據，另寫 `aggregate_gate0*.json`，`--pairs` 只篩選此獨立報告。完整判讀依凍結的 A、B 順序，前一組資格未知時不得用後一組定案。已存在的 raw 可用 `grade.py DIR --pair A --condition COND` 以 CPU 判分，已驗證產物與舊判分分開保留。新分片須共用同一個 `RSD_RUN_ID`（同一 Slurm array 預設共用 ID），且所有完成憑證到齊才判分。
- [codex] R3 仍以已凍結的條件報告。R3 若通過，先列為描述；有三條空間擇一路徑時的顯著水準分配待定案，不能因此升為確認性 PASS。H_a、H_b 保留原本固定 π* 與固定選定端點的 bootstrap，新增前緣有效範圍與端點替代告知。非正 HF 回收率分母明列，凍結的比較公式保留。dev 400 加碼須另外先補全部條件對齊及加碼後的判讀路徑。

</file>

<file name="configs/criteria.json">
{
  "_note": "所有門檻集中在此檔，aggregate.py 只讀這裡。Fable 重審後的版本（reports/2026-11-14-56）。事後修改須在 PREREG.md 記日期並標為探索性。2026-11-14 [claude] 使用者原則變更後、結果前定案：新增 phenomenon／signal／space／r3／overall（報告 60 通用條文 G 與第 5 節）；gate1 的 PASS/FAIL 降為描述。2026-11-14 審查後、結果前定案：alpha 0.0125、reproduce_sources、rl_compare_cond（非探索性）。2026-11-14 代理落差稽核後、結果前定案：fail_ii_upper_pct、fail_ii_exclude_pp、mix_curve、probe2（非探索性）。",
  "eval": {"n_samples": 128, "k_main": 64, "test_questions_min": 1000, "bootstrap_B": 10000, "bootstrap_seed": 20261114},
  "gate0": {
    "reproduce_tol_pp": 2.0,
    "reproduce_sources": ["math500", "minerva"],
    "reproduce_record_only_sources": ["olympiad"],
    "trunc_rate_diff_max_pp": 2.0,
    "pair_rl_p1_gain_min_pp": 5.0,
    "pair_rl_pk_below_base_min_pp": 2.0,
    "pass_window_pct": [30.0, 85.0],
    "pass_window_applies_to": ["base_pass_at_k_main", "rl_pass_at_k_main"]
  },
  "gate1": {
    "replace_minus_rl_min_pp": 2.0,
    "ci_one_sided_alpha": 0.0125,
    "rl_compare_cond": "rl_hf",
    "rl_compare_cond_note": "(i) 比較對象為 HF 版 RL 取樣（gen_replace.py --cond rl），與 rl_replace 同引擎；vLLM 版 rl 只用於閘門 0 與 (ii) 前緣；rl_hf 為 (ii) 的敏感度端點",
    "wise_lambdas": [0.15, 0.3, 0.45, 0.6, 0.75, 0.9],
    "wise_margin_min_pp": 0.0,
    "fail_ii_upper_pct": 98.75,
    "fail_ii_exclude_pp": 2.0,
    "fail_ii_note": "2026-11-14 R1（結果前定案）：(ii) 的 FAIL 需同時：差值點估計 <= 0 且差值的 98.75 百分位 < +2.0pp；點估計 <= 0 但 98.75 百分位 >= 2.0pp 歸灰區（走原本唯一一次 dev 400 加碼），合併後仍排除不了 2pp 記「未能測試（檢定力不足）」。(i) 的判定不變。",
    "mix_curve": {"pi_step": 0.05, "k": 64, "n": 128, "compare_cond": "rl_hf", "descriptive_only": true},
    "gray_zone": {"point_diff_min_pp": 1.0, "action": "只准一次：加跑 dev 400 題（RL、rl_replace、最接近的 WiSE 點）後合併重算，事先已定"}
  },
  "probe2": {
    "_note": "2026-11-14 R3（結果前定案）：唯一的第二探針，λ=0.5 幾何混合（gen_replace.py --cond geo_mix --lam 0.5）。觸發：合格配對的 (i) 成立而 (ii) FAIL；只做一個配對，A 符合就做 A，否則做 B。只有通過或未通過，不加碼。",
    "cond": "geo_mix", "lam": 0.5, "pair_order": ["A", "B"],
    "pk_gain_vs_rl_hf_min_pp": 2.0,
    "ci_one_sided_alpha": 0.00625,
    "retain_p1_frac_of_rl_gain": 0.5,
    "rl_ref_cond": "rl_hf",
    "gpu_hour_cap": 25,
    "total_gpu_hour_cap_unchanged": 120,
    "report_if_projected_exceeds_total": true
  },
  "phenomenon": {
    "_note": "2026-11-14 [claude]（使用者原則變更後、結果前定案，報告 60 通用條文 G 與第 5 節）：現象＝閘門 0(c)。以 A、B 兩組判定：任一組合格即 PASS；(a)(b) 都過但兩組皆不符合 (c) 即 FAIL（範圍限 1.5B、3000 token；配對 C 另開由使用者決定）；其餘（含 (a)(b) 失敗）記未能測試。",
    "scope": "1.5B, 3000 token",
    "pair_order": ["A", "B"]
  },
  "signal": {
    "q1_rr_minus_rlhf_pk_min_pp": 2.0,
    "q1_ci_one_sided_alpha": 0.0125,
    "q1_cond": "rl_hf",
    "q2_recovery_min": 0.5,
    "q2_definition": "(rl_replace p1 - base p1) / (rl_hf p1 - base p1)",
    "fail_q1_below_pp": 1.0,
    "_note": "PASS＝量 1 與量 2 都成立；FAIL＝量 1 的差點估計 < 1pp；其餘灰區（走 dev 400 加碼，一次）。主配對＝A 合格就用 A，否則 B（與 probe2 的配對選擇規則相同）。"
  },
  "space": {
    "m_paths": 2,
    "ci_one_sided_alpha": 0.00625,
    "pass_min_pp": 2.0,
    "fail_upper_pct": 99.375,
    "fail_exclude_pp": 2.0,
    "endpoint_conds": ["rl", "rl_hf"],
    "endpoint_rule": "lambda=1 端點用 vLLM 版 rl 與 rl_hf 各算一次，取差值較小者（點估計）及其 bootstrap",
    "mix_cond": "rl_hf",
    "mix_pi_step": 0.05,
    "pi_star_rule": "mix pass@1 最接近 rl_replace pass@1 的 π（格點 0..1 間隔 0.05）；距離相同取較小 π；π* 以點估計固定，bootstrap 中不重選",
    "H_b_frontier_x": "WiSE 前緣取在 π* 混合自身的 pass@1 處",
    "gray_action": "只准一次：加跑 dev 400 題後合併重算；仍分不出記未能測試（檢定力不足）",
    "_note": "PASS＝H_a 或 H_b 點估計 >= 2.0pp 且單側下界（0.625 百分位）> 0；FAIL＝H_a、H_b 點估計都 <= 0 且 99.375 百分位都 < 2.0pp；其餘灰區。"
  },
  "r3": {
    "_note": "R3（probe2 設定不變）改為空間的第三個上限候選。觸發：主配對的訊號 PASS 而空間 FAIL。probe2 通過 → 空間記 PASS；未通過 → 空間 FAIL。預算 25 GPU 小時（probe2.gpu_hour_cap）。",
    "trigger": "signal PASS and space FAIL",
    "role": "space_third_upper_bound_candidate"
  },
  "overall": {
    "rule": "G1：現象、訊號、空間三項都 PASS → 進候選池；任一項 FAIL → 在本 PREREG 範圍內不進候選池（寫明哪一項）；其餘灰區。未能測試不等於 FAIL。",
    "descriptive_only": ["gate1.verdict", "gate1.wise_margin_*", "gate1.rr_minus_rl_*", "R1 FAIL 條件", "第 2 週學生門檻", "recovery_p1(vLLM rl)"]
  },
  "correctness_gates": {"exact_match_required": true, "wrb_reference_match_required": true}
}

</file>

<file name="configs/legacy_artifacts.json">
{
  "A": {
    "eval_sha256": "d8f9beca6a745e402509e379efa122e9df02e913467f2483c32d8a18a593163a",
    "pairs_sha256": "8fe15ff24d56916255a29427923c11c47912b90f832c3b1a29c4710135552025",
    "decode_ast_sha256": "f0f12de621ba949cede65375415a5a4b479913c3e2b26cba5c647b3108298e3c",
    "base_snapshot": "4a83ca6e4526a4f2da3aa259ec36c259f66b2ab2",
    "rl_snapshot": "a98e477854071157a450e57dd45fd0684b6fa38a",
    "gate_minus1": {"job_id": 26270, "receipt_log_sha256": "0b77ab991591d51faf3ca2885be33dcea134275f90791b102f9edb9a5fa811b0", "job_log_sha256": "316f310e3bd694977aaaf784ed39554f2c69c646b2dd866d22ff92a888108234"},
    "runs": {
      "rl_greedy": {"job_id": 26290, "log_sha256": "6cbbaeda719c7def1bb20c3913e1aa08933f4e3f3d9574ee0ab1401651975673"},
      "base_greedy": {"job_id": 26291, "log_sha256": "4bb55f915adcb513556b6d19c87de0568a8fb6dac0b96932387358d69e68b6f2"},
      "base": {"job_id": 26292, "log_sha256": "8350d7015c2c59825895d1d22a5b3f0109fc222c2e320e7df39fd30b7c3f876d"},
      "rl": {"job_id": 26293, "log_sha256": "f705bc0dfee5d4e5e62b528df3ffb0bdcdece68880a17617f2f554450f115260"}
    }
  }
}

</file>

<file name="scripts/aggregate.py">
"""機械判讀：依 configs/criteria.json 輸出閘門 0／判準 1 的 PASS／灰區／FAIL／未能測試與全部數字。
用法：aggregate.py [--pairs A B] [--results results] [--selftest]
正式結果目錄約定：results/<pair>/<cond>/graded_verified.json 與取樣／判分憑證，cond ∈
  base, rl（取樣 n=128, test）、rl_greedy, base_greedy（split all, n=1）、rl_replace、wise_0.15 ...（WiSE-FT 各 λ）
判準的所有數字只從 criteria.json 讀；改門檻不必改此檔。
2026-11-14 [claude]（使用者原則變更後、結果前定案，報告 60 通用條文 G 與第 5 節）：整體判定改為現象／訊號／空間三項
（phenomenon_eval／signal_eval／space_eval／r3_status／overall_eval）；gate1 的 verdict 與 R1 FAIL 條件、rl_replace 對 WiSE 的比較只做描述。
2026-11-14（代理落差稽核後、結果前定案）：R1 (ii) 的 FAIL 需排除 +2pp；R2 回應層級混合曲線（只描述）；R3 geo_mix 第二探針（probe2）。
"""
import argparse, json, math, os, sys
from pathlib import Path
import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from common import ROOT, cfg, load_data
from artifacts import ArtifactError, atomic_json, check_gate_minus1, load_verified


def pass_at_k_vec(c, n, k):
    """無偏估計 1 - C(n-c,k)/C(n,k)，c 為每題答對數的陣列。"""
    c = np.asarray(c)
    out = np.ones(len(c))
    for i, ci in enumerate(c):
        if n - ci < k:
            continue
        # prod_{j=n-ci+1}^{n} (1 - k/j)
        j = np.arange(n - ci + 1, n + 1, dtype=float)
        out[i] = 1.0 - float(np.prod(1.0 - k / j)) if ci > 0 else 0.0
    return out


def per_q(graded, ids, k):
    p1, pk, trunc, ntok, repl = [], [], [], [], []
    for i in ids:
        g = graded[i]
        n = len(g["correct"])
        c = sum(g["correct"])
        p1.append(c / n)
        pk.append(pass_at_k_vec([c], n, k)[0] if n >= k else np.nan)
        trunc.append(1 - sum(g["finished"]) / n)
        ntok.append(sum(g["ntok"]))
        repl.append(sum(g.get("n_repl", [0])))
    return dict(p1=np.array(p1), pk=np.array(pk), trunc=np.array(trunc), ntok=np.array(ntok), repl=np.array(repl))


def interp_frontier(xs, ys, x0):
    """分段線性前緣在 x0 的值（xs 排序；超出範圍取端點並記錄）。"""
    o = np.argsort(xs)
    return float(np.interp(x0, np.asarray(xs)[o], np.asarray(ys)[o]))


def boot_idx(Q, B, seed):
    rng = np.random.default_rng(seed)
    return rng.integers(0, Q, size=(B, Q))


def mix_curve(c_rl, c_base, n, k, pis):
    """R2（只描述）：回應層級混合的 (pass@1, pass@k) 無偏估計。每次嘗試以機率 π 取 RL 樣本、否則取 base 樣本。
    (1-p)^k 估為 Σ_m Binom(k, m; π) · C(n-c_rl, m)/C(n, m) · C(n-c_base, k-m)/C(n, k-m)。回傳 (p1[], pk[])（對題目平均）。"""
    c_rl, c_base = np.asarray(c_rl, float), np.asarray(c_base, float)
    def hyp(c):  # H[:, m] = C(n-c, m)/C(n, m) = prod_{j<m} (n-c-j)/(n-j)
        H = np.ones((len(c), k + 1))
        for m in range(k):
            H[:, m + 1] = H[:, m] * np.maximum(n - c - m, 0.0) / (n - m)
        return H
    Hr, Hb = hyp(c_rl), hyp(c_base)
    p1, pk = [], []
    for pi in pis:
        allfail = np.zeros(len(c_rl))
        for m in range(k + 1):
            w = math.comb(k, m) * (pi ** m) * ((1 - pi) ** (k - m))
            if w > 0:
                allfail += w * Hr[:, m] * Hb[:, k - m]
        pk.append(float(np.mean(1.0 - allfail)))
        p1.append(float(pi * np.mean(c_rl / n) + (1 - pi) * np.mean(c_base / n)))
    return np.array(p1), np.array(pk)


def wise_pts(pq, ccfg):
    pts = [("0", pq["base"])] + [(str(l), pq.get("wise_%s" % l)) for l in ccfg["wise_lambdas"]] + [("1", pq["rl"])]
    return pts, [n for n, v in pts if v is None]


def wise_margin_boot(rr, pts, idx):
    """rr 對 WiSE 前緣的差值：點估計與 bootstrap 分布（與 gate1 (ii) 同一算式）。"""
    P1 = np.stack([v["p1"] for _, v in pts], 1)
    PK = np.stack([v["pk"] for _, v in pts], 1)
    margin = rr["pk"].mean() - interp_frontier(P1.mean(0), PK.mean(0), rr["p1"].mean())
    bm = np.empty(len(idx))
    for b in range(len(idx)):
        ii = idx[b]
        bm[b] = rr["pk"][ii].mean() - interp_frontier(P1[ii].mean(0), PK[ii].mean(0), rr["p1"][ii].mean())
    return margin, bm


def probe2_eval(pq, ccfg, pcfg, ecfg):
    """R3：λ=0.5 幾何混合探針。三項都要成立（單側 α=pcfg.ci_one_sided_alpha）；只有通過／未通過，不加碼。"""
    gm, rlh, base = pq[pcfg["cond"]], pq[pcfg["rl_ref_cond"]], pq["base"]
    pts, missing = wise_pts(pq, ccfg)
    if missing:
        return {"verdict": "未能測試（缺 WiSE 點: %s）" % missing}
    Q = len(gm["pk"])
    idx = boot_idx(Q, ecfg["bootstrap_B"], ecfg["bootstrap_seed"])
    a = pcfg["ci_one_sided_alpha"]
    d = gm["pk"] - rlh["pk"]
    c1_pt, c1_lo = 100 * d.mean(), 100 * np.quantile(d[idx].mean(1), a)
    c1 = c1_pt >= pcfg["pk_gain_vs_rl_hf_min_pp"] and c1_lo > 0
    mg, bm = wise_margin_boot(gm, pts, idx)
    c2_pt, c2_lo = 100 * mg, 100 * np.quantile(bm, a)
    c2 = c2_pt > 0 and c2_lo > 0
    gain_gm = 100 * (gm["p1"].mean() - base["p1"].mean())
    gain_rl = 100 * (rlh["p1"].mean() - base["p1"].mean())
    c3 = gain_gm >= pcfg["retain_p1_frac_of_rl_gain"] * gain_rl
    P1 = np.stack([v["p1"] for _, v in pts], 1).mean(0)
    return {"verdict": "通過" if (c1 and c2 and c3) else "未通過（只有通過或未通過，不加碼；探針未通過才可提請淘汰）",
            "cond1_pk_minus_rlhf_pp": c1_pt, "cond1_lo_pp": c1_lo, "cond1_ok": bool(c1),
            "cond2_margin_vs_wise_pp": c2_pt, "cond2_lo_pp": c2_lo, "cond2_ok": bool(c2),
            "cond2_p1_in_wise_range": bool(P1.min() <= gm["p1"].mean() <= P1.max()),
            "cond3_p1_gain_pp": gain_gm, "cond3_rl_p1_gain_pp": gain_rl, "cond3_required_pp": pcfg["retain_p1_frac_of_rl_gain"] * gain_rl, "cond3_ok": bool(c3),
            "alpha_one_sided": a, "geo_pk": 100 * gm["pk"].mean(), "geo_p1": 100 * gm["p1"].mean()}


def gate1(pq, ccfg, ecfg, merged=False):
    # (i) 比較對象：rl_hf（與 rl_replace 同為 HF 引擎）；(ii) 前緣端點 λ=1 用 vLLM 的 rl，rl_hf 作敏感度
    rr, rl, base = pq["rl_replace"], pq["rl"], pq["base"]
    rlh = pq[ccfg.get("rl_compare_cond", "rl_hf")]
    Q = len(rr["pk"])
    idx = boot_idx(Q, ecfg["bootstrap_B"], ecfg["bootstrap_seed"])
    alpha = ccfg["ci_one_sided_alpha"]
    d = rr["pk"] - rlh["pk"]
    bd = d[idx].mean(1)
    res = {"rr_minus_rl_pp": 100 * d.mean(), "rr_minus_rl_lo_pp": 100 * np.quantile(bd, alpha),
           "rr_minus_rl_ci95_pp": [100 * np.quantile(bd, 0.025), 100 * np.quantile(bd, 0.975)],
           "alpha_one_sided": alpha, "rl_compare_cond": ccfg.get("rl_compare_cond", "rl_hf")}
    # 診斷：引擎差（HF − vLLM 的 RL）
    dv = rlh["pk"] - rl["pk"]; bv = dv[idx].mean(1)
    res["diag_rlhf_minus_rlvllm_pk_pp"] = 100 * dv.mean()
    res["diag_rlhf_minus_rlvllm_pk_ci95_pp"] = [100 * np.quantile(bv, 0.025), 100 * np.quantile(bv, 0.975)]
    dv_p1 = rlh["p1"] - rl["p1"]
    res["diag_rlhf_minus_rlvllm_p1_pp"] = 100 * dv_p1.mean()
    res["diag_rlhf_minus_rlvllm_p1_ci95_pp"] = (100 * np.quantile(dv_p1[idx].mean(1), [0.025, 0.975])).tolist()
    res["diag_trunc_pct"] = {"base": 100 * base["trunc"].mean(), "rl_vllm": 100 * rl["trunc"].mean(),
                             "rl_hf": 100 * rlh["trunc"].mean(), "rl_replace": 100 * rr["trunc"].mean()}
    # WiSE-FT 前緣：λ=0(base)、各 λ、λ=1(rl)
    lams = ccfg["wise_lambdas"]
    pts = [("0", base)] + [(str(l), pq.get("wise_%s" % l)) for l in lams] + [("1", rl)]
    missing = [n for n, v in pts if v is None]
    res["wise_missing"] = missing
    res["wise_points"] = {n: [100 * v["p1"].mean(), 100 * v["pk"].mean()] for n, v in pts if v is not None}
    if not missing:
        P1 = np.stack([v["p1"] for _, v in pts], 1)  # Q x m
        PK = np.stack([v["pk"] for _, v in pts], 1)
        x_pt, y_pt = P1.mean(0), PK.mean(0)
        x_rr, y_rr = rr["p1"].mean(), rr["pk"].mean()
        res["rr_p1_in_wise_range"] = bool(x_pt.min() <= x_rr <= x_pt.max())
        margin = y_rr - interp_frontier(x_pt, y_pt, x_rr)
        bm = np.empty(len(idx))
        for b in range(len(idx)):
            ii = idx[b]
            bm[b] = rr["pk"][ii].mean() - interp_frontier(P1[ii].mean(0), PK[ii].mean(0), rr["p1"][ii].mean())
        res["wise_margin_pp"] = 100 * margin
        res["wise_margin_lo_pp"] = 100 * np.quantile(bm, alpha)
        res["wise_margin_ci95_pp"] = [100 * np.quantile(bm, 0.025), 100 * np.quantile(bm, 0.975)]
        res["wise_margin_upper_pp"] = 100 * np.quantile(bm, ccfg["fail_ii_upper_pct"] / 100.0)   # R1：98.75 百分位
        # 敏感度（不參與判定）：λ=1 端點換成 rl_hf
        P1s, PKs = P1.copy(), PK.copy()
        P1s[:, -1], PKs[:, -1] = rlh["p1"], rlh["pk"]
        res["wise_margin_sens_rlhf_endpoint_pp"] = 100 * (y_rr - interp_frontier(P1s.mean(0), PKs.mean(0), x_rr))
    res["rr_p1"], res["rr_pk"] = 100 * rr["p1"].mean(), 100 * rr["pk"].mean()
    res["rl_p1"], res["rl_pk"] = 100 * rl["p1"].mean(), 100 * rl["pk"].mean()
    res["rlhf_p1"], res["rlhf_pk"] = 100 * rlh["p1"].mean(), 100 * rlh["pk"].mean()
    res["base_p1"], res["base_pk"] = 100 * base["p1"].mean(), 100 * base["pk"].mean()
    res["recovery_p1"] = (res["rr_p1"] - res["base_p1"]) / (res["rl_p1"] - res["base_p1"]) if res["rl_p1"] != res["base_p1"] else float("nan")
    res["replace_rate_token_weighted"] = float(rr["repl"].sum() / rr["ntok"].sum())
    # R2（只描述，不參與判定）：回應層級混合曲線與弦
    mc = ccfg.get("mix_curve")
    if mc:
        nS, kS = mc["n"], mc["k"]
        pis = [round(i * mc["pi_step"], 10) for i in range(int(round(1 / mc["pi_step"])) + 1)]
        cmp_pq = pq[mc["compare_cond"]] if mc["compare_cond"] in pq else rl
        p1c, pkc = mix_curve(np.round(cmp_pq["p1"] * nS), np.round(base["p1"] * nS), nS, kS, pis)
        chord = [100 * (base["pk"].mean() + pi * (cmp_pq["pk"].mean() - base["pk"].mean())) for pi in pis]
        res["mix_curve_descriptive"] = {"compare_cond": mc["compare_cond"] if mc["compare_cond"] in pq else "rl",
                                        "points": [{"pi": pi, "p1": 100 * a, "pk": 100 * b, "chord_pk": c, "above_chord_pp": 100 * b - c}
                                                   for pi, a, b, c in zip(pis, p1c, pkc, chord)]}
    res["descriptive_only"] = True   # 2026-11-14 [claude]：不再決定整體判定（見 phenomenon_eval 等）
    # 三態（描述用）
    passed = (res["rr_minus_rl_pp"] >= ccfg["replace_minus_rl_min_pp"] and res["rr_minus_rl_lo_pp"] > 0 and not missing
              and res["wise_margin_pp"] > ccfg["wise_margin_min_pp"] and res["wise_margin_lo_pp"] > 0)
    gmin = ccfg["gray_zone"]["point_diff_min_pp"]
    res["i_pass"] = bool(res["rr_minus_rl_pp"] >= ccfg["replace_minus_rl_min_pp"] and res["rr_minus_rl_lo_pp"] > 0)
    if not missing:
        res["ii_pass"] = bool(res["wise_margin_pp"] > ccfg["wise_margin_min_pp"] and res["wise_margin_lo_pp"] > 0)
        # R1：(ii) 的 FAIL = 點估計 <= 0 且 98.75 百分位 < +2.0pp
        res["ii_fail"] = bool(res["wise_margin_pp"] <= 0 and res["wise_margin_upper_pp"] < ccfg["fail_ii_exclude_pp"])
        res["ii_underpowered"] = bool(res["wise_margin_pp"] <= 0 and not res["ii_fail"])
    if missing:
        res["verdict"] = "未能測試（缺 WiSE 點: %s）" % missing
    elif passed:
        res["verdict"] = "PASS"
    elif res["rr_minus_rl_pp"] < gmin or res["ii_fail"]:
        res["verdict"] = "FAIL"      # 只代表「λ=1 端點」這個實作停損，不等於題目淘汰
    elif res["ii_underpowered"] and merged:
        res["verdict"] = "未能測試（檢定力不足）"   # 已用過唯一一次 dev 加碼，仍排除不了 2pp
    else:
        res["verdict"] = "灰區"      # 含 R1：點估計 <= 0 但 98.75 百分位 >= 2pp
    return res


def gate0(pq, greedy_graded, pair, crit, ecfg, data):
    g0 = crit["gate0"]
    out = {}
    # (a) 重現 RL 已報 pass@1（greedy，對各大型來源）
    src = {r["id"]: r["source"] for r in data}
    rep = {}
    if "rl_greedy" in greedy_graded:
        by = {}
        for i, g in greedy_graded["rl_greedy"].items():
            by.setdefault(src[i], []).append(g["correct"][0])
        for s, v in by.items():
            rep[s] = 100 * float(np.mean(v)) if v else None
    out["rl_greedy_by_source"] = rep
    ref = pair["reported_rl_greedy"]
    diffs = {s: rep[s] - ref[s] for s in g0["reproduce_sources"] if s in rep and s in ref}
    out["reproduce_diffs_pp"] = diffs
    out["reproduce_record_only_diffs_pp"] = {s_: rep[s_] - ref[s_] for s_ in g0.get("reproduce_record_only_sources", []) if s_ in rep and s_ in ref}
    out["reproduce_ok"] = (len(diffs) == len(g0["reproduce_sources"])) and all(abs(v) <= g0["reproduce_tol_pp"] for v in diffs.values())
    # (b) 截斷率
    tb, tr = 100 * pq["base"]["trunc"].mean(), 100 * pq["rl"]["trunc"].mean()
    out["trunc_pct"] = {"base": tb, "rl": tr}
    out["trunc_ok"] = abs(tb - tr) < g0["trunc_rate_diff_max_pp"]
    # (c) 配對資格
    bp1, rp1 = 100 * pq["base"]["p1"].mean(), 100 * pq["rl"]["p1"].mean()
    bpk, rpk = 100 * pq["base"]["pk"].mean(), 100 * pq["rl"]["pk"].mean()
    out.update(base_p1=bp1, rl_p1=rp1, base_pk=bpk, rl_pk=rpk)
    lo, hi = g0["pass_window_pct"]
    win = {"base_pass_at_k_main": bpk, "rl_pass_at_k_main": rpk, "base_p1": bp1, "rl_p1": rp1}
    out["window_ok"] = all(lo <= win[k] <= hi for k in g0["pass_window_applies_to"])
    out["gain_ok"] = (rp1 - bp1) >= g0["pair_rl_p1_gain_min_pp"]
    out["rl_pk_below_base_ok"] = (bpk - rpk) >= g0["pair_rl_pk_below_base_min_pp"]
    out["qualified"] = out["gain_ok"] and out["rl_pk_below_base_ok"] and out["window_ok"]
    if not (out["reproduce_ok"] and out["trunc_ok"]):
        out["verdict"] = "未能測試（重現或截斷率閘門未過：先修程式／模板，限 3 個工作天）"
    elif not out["qualified"]:
        out["verdict"] = "配對不合格（換配對，不算題目死）"
    else:
        out["verdict"] = "配對合格"
    return out


def mix_pq(c_rl, c_base, n, k, pi):
    """R2 混合在單一 π 的逐題 (p1_q, pass@k_q)；mix_curve 的逐題版（bootstrap 用）。"""
    c_rl, c_base = np.asarray(c_rl, float), np.asarray(c_base, float)
    def hyp(c):
        H = np.ones((len(c), k + 1))
        for m in range(k):
            H[:, m + 1] = H[:, m] * np.maximum(n - c - m, 0.0) / (n - m)
        return H
    Hr, Hb = hyp(c_rl), hyp(c_base)
    allfail = np.zeros(len(c_rl))
    for m in range(k + 1):
        w = math.comb(k, m) * (pi ** m) * ((1 - pi) ** (k - m))
        if w > 0:
            allfail += w * Hr[:, m] * Hb[:, k - m]
    return pi * c_rl / n + (1 - pi) * c_base / n, 1.0 - allfail


# ---------------- 報告 60 第 5 節：現象／訊號／空間 ----------------
def phenomenon_eval(summary, crit):
    """現象＝閘門 0(c)。任一組合格 → PASS（主配對＝順序上第一個合格者）；
    每個有資料的組都 (a)(b) 過、且皆不符 (c) → FAIL；其餘（含 (a)(b) 失敗、缺資料）→ 未能測試。"""
    order = crit["phenomenon"]["pair_order"]
    g0 = {p: summary[p]["gate0"] for p in order if p in summary}
    for p in order:
        verdict = g0.get(p, {}).get("verdict", "")
        if verdict == "配對合格":
            return {"verdict": "PASS", "primary_pair": p, "scope": crit["phenomenon"]["scope"]}
        if not verdict.startswith("配對不合格"):
            return {"verdict": "未能測試", "primary_pair": None, "waiting_for_eligibility": p,
                    "note": "前序配對須先確定有資格或確認不合格"}
    ab_ok = [bool(g.get("reproduce_ok") and g.get("trunc_ok")) for g in g0.values() if "reproduce_ok" in g]
    if len(ab_ok) == len(order) and all(ab_ok):
        return {"verdict": "FAIL", "primary_pair": None, "scope": crit["phenomenon"]["scope"],
                "note": "(a)(b) 都過但 A、B 皆不符 (c)；範圍限 %s；配對 C 是否另開由使用者決定" % crit["phenomenon"]["scope"]}
    return {"verdict": "未能測試", "primary_pair": None, "note": "至少一組 (a)(b) 未過或缺資料，且沒有合格配對（修復期 3 個工作天）"}


def signal_eval(pq, crit, merged=False):
    """訊號：量 1 = rl_replace pass@64 − rl_hf ≥ 2pp 且單側 1.25 百分位 > 0；
    量 2 = pass@1 回收率 (rr−base)/(rl_hf−base) ≥ 0.5。PASS＝兩者；FAIL＝量 1 差 < 1pp；其餘灰區。"""
    sc, ecfg = crit["signal"], crit["eval"]
    rr, rlh, base = pq["rl_replace"], pq[sc["q1_cond"]], pq["base"]
    idx = boot_idx(len(rr["pk"]), ecfg["bootstrap_B"], ecfg["bootstrap_seed"])
    d = rr["pk"] - rlh["pk"]
    q1_pt, q1_lo = 100 * d.mean(), 100 * np.quantile(d[idx].mean(1), sc["q1_ci_one_sided_alpha"])
    q1 = bool(q1_pt >= sc["q1_rr_minus_rlhf_pk_min_pp"] and q1_lo > 0)
    den = rlh["p1"].mean() - base["p1"].mean()
    rec = float((rr["p1"].mean() - base["p1"].mean()) / den) if den != 0 else float("nan")
    q2 = bool(rec >= sc["q2_recovery_min"])
    if q1 and q2:
        v = "PASS"
    elif q1_pt < sc["fail_q1_below_pp"]:
        v = "FAIL"
    else:
        v = "灰區（加碼後仍灰區：交使用者）" if merged else "灰區"
    return {"verdict": v, "q1_rr_minus_rlhf_pk_pp": q1_pt, "q1_lo_pp": q1_lo, "q1_ok": q1,
            "q2_recovery_p1": rec, "q2_rlhf_minus_base_p1_pp": 100 * den,
            "q2_denominator_positive": bool(den > 0), "q2_ok": q2, "alpha_one_sided": sc["q1_ci_one_sided_alpha"]}


def space_eval(pq, crit, merged=False):
    """空間：H_a（rl_replace 對 WiSE 前緣）、H_b（R2 混合在 π* 對前緣），擇一（m=2，單側 α=0.00625）。"""
    sc, ecfg, g1 = crit["space"], crit["eval"], crit["gate1"]
    miss = [c for c in ["base", "rl_replace", sc["mix_cond"]] + sc["endpoint_conds"] + ["wise_%s" % l for l in g1["wise_lambdas"]] if c not in pq]
    if miss:
        return {"verdict": "未能測試（缺 %s）" % sorted(set(miss))}
    base, rr = pq["base"], pq["rl_replace"]
    idx = boot_idx(len(rr["pk"]), ecfg["bootstrap_B"], ecfg["bootstrap_seed"])
    a = sc["ci_one_sided_alpha"]
    mid = [("%s" % l, pq["wise_%s" % l]) for l in g1["wise_lambdas"]]
    def frontier_gap(target):
        """λ=1 端點用 rl 與 rl_hf 各算一次，取點估計較小者。"""
        best = None
        for ec in sc["endpoint_conds"]:
            pts = [("0", base)] + mid + [("1", pq[ec])]
            m, bm = wise_margin_boot(target, pts, idx)
            if best is None or m < best[0]:
                best = (m, bm, ec, pts)
        m, bm, ec, pts = best
        low = min(q["p1"].mean() for _, q in pts)
        high = max(q["p1"].mean() for _, q in pts)
        xp = target["p1"].mean()
        return {"pt_pp": 100 * m, "lo_pp": 100 * float(np.quantile(bm, a)), "upper_pp": 100 * float(np.quantile(bm, 1 - a)),
                "endpoint": ec, "target_p1_pp": 100 * xp, "frontier_p1_range_pp": [100 * low, 100 * high],
                "frontier_clamped": not bool(low <= xp <= high)}
    Ha = frontier_gap(rr)
    # H_b：π*（mix pass@1 最接近 rl_replace pass@1；同距取小）
    nS, kS = g1["mix_curve"]["n"], g1["mix_curve"]["k"]
    c_rl, c_b = np.round(pq[sc["mix_cond"]]["p1"] * nS), np.round(base["p1"] * nS)
    step = sc["mix_pi_step"]
    pis = [round(i * step, 10) for i in range(int(round(1 / step)) + 1)]
    p1c, _ = mix_curve(c_rl, c_b, nS, kS, pis)
    dist = np.abs(p1c - rr["p1"].mean())
    pi_star = pis[int(np.argmin(dist + 1e-12 * np.arange(len(pis))))]   # 同距取較小 π
    mp1, mpk = mix_pq(c_rl, c_b, nS, kS, pi_star)
    Hb = frontier_gap({"p1": mp1, "pk": mpk})
    Hb["pi_star"] = pi_star
    paths = {"H_a": Ha, "H_b": Hb}
    ok = [n for n, h in paths.items() if h["pt_pp"] >= sc["pass_min_pp"] and h["lo_pp"] > 0]
    fail = all(h["pt_pp"] <= 0 and h["upper_pp"] < sc["fail_exclude_pp"] for h in paths.values())
    if ok:
        v = "PASS"
    elif fail:
        v = "FAIL"
    elif merged:
        v = "未能測試（檢定力不足）"
    else:
        v = "灰區"
    return {"verdict": v, "paths_passing": ok, "H_a": Ha, "H_b": Hb, "alpha_one_sided": a, "m_paths": sc["m_paths"]}


def r3_status(primary, sig, spc, PQ, crit):
    """R3＝空間的第三個上限候選。觸發：主配對訊號 PASS 而空間 FAIL。"""
    pcfg = crit["probe2"]
    if not (primary and sig.get("verdict") == "PASS" and spc.get("verdict") == "FAIL"):
        return {"triggered": False, "result": {"verdict": "未觸發"}}
    pq = PQ[primary]
    if pcfg["cond"] in pq and all(c in pq for c in ("base", "rl", "rl_hf")):
        return {"triggered": True, "pair": primary, "result": probe2_eval(pq, crit["gate1"], pcfg, crit["eval"])}
    return {"triggered": True, "pair": primary, "result": {"verdict": "待執行（觸發；缺 %s 取樣結果，見 slurm/06_geo_mix.sbatch）" % pcfg["cond"]}}


def overall_eval(phen, sig, spc, r3):
    """G1：三項都 PASS → 進候選池；任一 FAIL → 不進池（寫明哪一項）；其餘灰區／未能測試／待執行。"""
    space_v = spc.get("verdict", "未評估")
    space_note = None
    if r3.get("triggered"):
        rv = r3["result"]["verdict"]
        if rv == "通過":
            space_v, space_note = "未能測試（R3 多重性待定案）", "R3 通過為描述，第三條擇一路徑的顯著水準尚待定案"
        elif rv.startswith("未通過"):
            space_v, space_note = "FAIL", "R3 未通過"
        else:
            space_v, space_note = "待執行 R3", "訊號 PASS 而空間 FAIL，須先做 R3"
    items = {"現象": phen.get("verdict", "未評估"), "訊號": sig.get("verdict", "未評估"), "空間": space_v}
    fails = [k for k, v in items.items() if v == "FAIL"]
    if fails:
        ov = "FAIL：在本 PREREG 範圍內不進候選池（%s FAIL；範圍限 1.5B、3000 token、本批配對）" % "、".join(fails)
    elif all(v == "PASS" for v in items.values()):
        ov = "PASS：進候選池"
    elif any(v.startswith("待執行") for v in items.values()):
        ov = "待 R3（其餘項目未 FAIL）"
    elif any(v.startswith("未能測試") for v in items.values()):
        ov = "未能測試（%s）" % "、".join(k for k, v in items.items() if v.startswith("未能測試"))
    else:
        ov = "灰區（只准條文指定的一次 dev 400 加碼，否則交給使用者）"
    return {"items": items, "space_note": space_note, "overall": ov}


def serializable(value):
    if isinstance(value, dict):
        return {k: serializable(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [serializable(v) for v in value]
    if isinstance(value, np.generic):
        return serializable(value.item())
    if isinstance(value, float) and not math.isfinite(value):
        return None
    return value


def write_summary(path, summary):
    serial = serializable(summary)
    atomic_json(path, serial)
    print(json.dumps(serial, indent=1, ensure_ascii=False, allow_nan=False))


def evaluate(results, pairs, gate, crit, pc, data):
    """Gate 0 reads only reproduction/sample evidence; later files require a qualified primary pair."""
    ecfg = crit["eval"]
    test_ids = [r["id"] for r in data if r["split"] == "test"]
    assert len(test_ids) >= ecfg["test_questions_min"], "test 題數不足"
    if len(set(r["id"] for r in data)) != len(data):
        raise ArtifactError("凍結資料含重複題目 ID")
    order = crit["phenomenon"]["pair_order"]
    selected = set(order) if gate != "0" or not pairs else set(pairs)
    summary, PQ, receipts = {}, {}, {}

    def load_condition(p, c):
        try:
            return load_verified(results / p / c, p, c, pc, ecfg, data, receipts[p])
        except (ArtifactError, OSError, ValueError, KeyError, TypeError) as exc:
            summary[p]["artifact_issues"][c] = str(exc)
            return None

    for p in order:
        if p not in selected:
            continue
        s = summary[p] = {"pair": pc["pairs"][p]["name"], "artifact_issues": {}}
        try:
            receipts[p] = check_gate_minus1(results, p, pc)
        except (ArtifactError, OSError, ValueError, KeyError, TypeError) as exc:
            s["gate_minus1"] = {"verdict": "未能測試", "reason": str(exc)}
            s["gate0"] = {"verdict": "未能測試（缺相容閘門 −1 憑證）"}
            continue
        s["gate_minus1"] = {"verdict": "PASS", **receipts[p]}
        pq, greedy = PQ[p] = {}, {}
        for c in ("base", "rl", "base_greedy", "rl_greedy"):
            g = load_condition(p, c)
            if g is not None:
                if c.endswith("_greedy"):
                    greedy[c] = g
                else:
                    pq[c] = per_q(g, test_ids, ecfg["k_main"])
        if set(pq) == {"base", "rl"} and set(greedy) == {"base_greedy", "rl_greedy"}:
            s["gate0"] = gate0(pq, greedy, pc["pairs"][p], crit, ecfg, data)
        else:
            s["gate0"] = {"verdict": "未能測試（四個取樣與 greedy 條件須全部驗證）"}
    if gate == "0":
        return {"phase": "gate0", "partial": True, "pairs": summary}

    phen = phenomenon_eval(summary, crit)
    prim = phen.get("primary_pair")
    sig = spc = {"verdict": "未評估（現象未 PASS）"}
    if prim:
        pq = PQ[prim]
        later = ["rl_replace", "rl_hf"] + ["wise_%s" % l for l in crit["gate1"]["wise_lambdas"]]
        for c in later:
            g = load_condition(prim, c)
            if g is not None:
                pq[c] = per_q(g, test_ids, ecfg["k_main"])
        need_s = ("base", "rl_replace", crit["signal"]["q1_cond"])
        sig = signal_eval(pq, crit) if all(c in pq for c in need_s) else {"verdict": "未能測試（缺 %s）" % [c for c in need_s if c not in pq]}
        spc = space_eval(pq, crit)
        if all(c in pq for c in ("base", "rl", "rl_replace", crit["gate1"].get("rl_compare_cond", "rl_hf"))):
            summary[prim]["gate1"] = gate1(pq, crit["gate1"], ecfg)
        else:
            summary[prim]["gate1"] = {"verdict": "未執行（缺正式取樣憑證）"}
        if sig.get("verdict") == "PASS" and spc.get("verdict") == "FAIL":
            c = crit["probe2"]["cond"]
            g = load_condition(prim, c)
            if g is not None:
                pq[c] = per_q(g, test_ids, ecfg["k_main"])
    r3 = r3_status(prim, sig, spc, PQ, crit)
    summary["phenomenon"], summary["signal"], summary["space"], summary["r3"] = phen, sig, spc, r3
    ov = overall_eval(phen, sig, spc, r3)
    summary["overall_three_items"] = ov
    overall = ov["overall"]
    summary["overall"] = overall
    summary["probe2"] = r3     # 相容舊鍵名
    return summary


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--pairs", nargs="*", choices=["A", "B"], help="只篩選 gate 0 報告的配對")
    ap.add_argument("--results", type=Path, default=Path(ROOT) / "results")
    ap.add_argument("--gate", choices=["0", "1", "all"], default="all")
    ap.add_argument("--selftest", action="store_true")
    a = ap.parse_args()
    if a.selftest:
        return selftest()
    if a.pairs is not None and a.gate != "0":
        ap.error("--pairs 只適用於 --gate 0；正式判定一律檢查凍結的配對順序")
    crit, pc = cfg("criteria.json"), cfg("pairs.json")
    summary = evaluate(a.results, a.pairs, a.gate, crit, pc, load_data("all"))
    a.results.mkdir(parents=True, exist_ok=True)
    if a.gate == "0":
        suffix = "_" + "_".join(sorted(set(a.pairs))) if a.pairs else ""
        write_summary(a.results / f"aggregate_gate0{suffix}.json", summary)
    else:
        write_summary(a.results / "aggregate.json", summary)
        print("\n== 現象／訊號／空間：", summary["overall_three_items"]["items"], "\n== OVERALL:", summary["overall"])


def selftest():
    # 無偏估計式對照暴力組合
    from math import comb
    for n, c, k in [(128, 0, 64), (128, 5, 64), (128, 70, 64), (128, 128, 64), (10, 3, 4)]:
        ref = 0.0 if c == 0 else (1.0 if n - c < k else 1 - comb(n - c, k) / comb(n, k))
        assert abs(pass_at_k_vec([c], n, k)[0] - ref) < 1e-9, (n, c, k)
    print("pass@k estimator OK")
    # 合成資料：rr 明顯高於 rl，應 PASS／或 FAIL 依設定
    rng = np.random.default_rng(0)
    Q, n = 1000, 128
    skill = rng.beta(0.6, 0.9, Q)
    def mk(shift):
        s = np.clip(skill + shift, 0, 1)
        c = rng.binomial(n, s)
        return {"p1": c / n, "pk": pass_at_k_vec(c, n, 64), "trunc": np.zeros(Q), "ntok": np.full(Q, 1000), "repl": np.full(Q, 40)}
    crit = cfg("criteria.json")
    pq = {"base": mk(0.0), "rl": mk(0.0), "rl_replace": mk(0.0), "rl_hf": mk(0.0)}
    for i, l in enumerate(crit["gate1"]["wise_lambdas"]):
        pq["wise_%s" % l] = mk(0.0)
    r = gate1(pq, crit["gate1"], crit["eval"]); print("null scenario verdict:", r["verdict"], round(r["rr_minus_rl_pp"], 2)); assert r["verdict"] == "FAIL"
    assert crit["gate1"]["ci_one_sided_alpha"] == 0.0125 and crit["gate0"]["reproduce_sources"] == ["math500", "minerva"]
    # rr 明顯高於 rl_hf：應 PASS 的路徑（rl_hf 仍 = rl 水準）
    pq2 = dict(pq); pq2["rl_replace"] = mk(0.08)
    r2 = gate1(pq2, crit["gate1"], crit["eval"]); print("positive scenario:", r2["verdict"], round(r2["rr_minus_rl_pp"], 2), r2["rl_compare_cond"])
    assert r2["rr_minus_rl_pp"] > 2 and "diag_rlhf_minus_rlvllm_pk_pp" in r2
    selftest_2026_11_14(crit)
    print("selftest OK")


def selftest_2026_11_14(crit):
    """R1–R3 的新增分支（合成資料）。rr／geo_mix 取為 WiSE 某節點加固定位移，使前緣差值恰等於位移（無內插誤差）。"""
    rng = np.random.default_rng(1)
    g1c = dict(crit["gate1"])
    ecfg = dict(crit["eval"]); ecfg["bootstrap_B"] = 2000
    # ---- R2：混合曲線 ----
    n, k, Q = 128, 64, 300
    c_b, c_r = rng.binomial(n, rng.beta(0.5, 1.0, Q)), rng.binomial(n, rng.beta(1.0, 0.8, Q))
    p1, pk = mix_curve(c_r, c_b, n, k, [0.0, 0.5, 1.0])
    assert abs(pk[0] - pass_at_k_vec(c_b, n, k).mean()) < 1e-9 and abs(pk[2] - pass_at_k_vec(c_r, n, k).mean()) < 1e-9
    assert abs(p1[0] - c_b.mean() / n) < 1e-12 and abs(p1[2] - c_r.mean() / n) < 1e-12
    # 無偏性：真實 p 固定，蒙地卡羅平均的 (1-pk) 應等於 (π(1-p_r)+(1-π)(1-p_b))^k
    pr, pb, pi_, K, N, reps = 0.05, 0.10, 0.4, 8, 32, 6000
    cr_ = rng.binomial(N, pr, reps); cb_ = rng.binomial(N, pb, reps)
    _, pkk = mix_curve(cr_, cb_, N, K, [pi_])
    true_pass = 1 - (pi_ * (1 - pr) + (1 - pi_) * (1 - pb)) ** K
    assert abs(pkk[0] - true_pass) < 0.01 and 0.3 < true_pass < 0.7, (pkk[0], true_pass)
    print("R2 mix_curve OK (端點等於 pass@k；無偏性 %.4f vs %.4f)" % (pkk[0], true_pass))
    # ---- 合成 pq：每題 p1、pk；WiSE 節點 λ：p1 = a + λ(r−a)，pk = 1−(1−p1)^k 的凹函數 ----
    def build(Q, noise_sd, shift_rr, rr_node="0.6", rl_pk_penalty=0.15):
        a = rng.uniform(0.02, 0.30, Q); r = np.clip(a + rng.uniform(0.0, 0.5, Q), 0, 1)
        f = lambda p: 1 - (1 - p) ** 8
        mkq = lambda p1_, pk_: {"p1": p1_, "pk": pk_, "trunc": np.zeros(Q), "ntok": np.full(Q, 1000), "repl": np.full(Q, 40)}
        pq = {"base": mkq(a, f(a)), "rl": mkq(r, np.clip(f(r) - rl_pk_penalty, 0, 1))}
        pq["rl_hf"] = mkq(pq["rl"]["p1"].copy(), pq["rl"]["pk"].copy())
        for l in g1c["wise_lambdas"]:
            q = a + l * (r - a); pq["wise_%s" % l] = mkq(q, f(q))
        node = pq["wise_%s" % rr_node]
        nz = rng.normal(0, noise_sd, Q); nz -= nz.mean()
        pq["rl_replace"] = mkq(node["p1"].copy(), node["pk"] + shift_rr + nz)
        return pq, mkq, f, a, r
    # 1) (i) 成立、(ii) 明確 FAIL：位移 −3pp，Q 大、雜訊小 → 98.75 百分位 < 2pp
    pq, mkq, f, a, r = build(1000, 0.05, -0.03)
    r1 = gate1(pq, g1c, ecfg)
    assert r1["i_pass"] and r1["wise_margin_pp"] <= 0 and r1["wise_margin_upper_pp"] < 2.0 and r1["ii_fail"] and r1["verdict"] == "FAIL", r1
    # 2) 點估計 <= 0 但檢定力不足：Q 小、雜訊大 → 上界 >= 2pp → 灰區；合併（merged）後記「未能測試（檢定力不足）」
    pq2, *_ = build(60, 0.30, -0.005)
    r2 = gate1(pq2, g1c, ecfg)
    assert r2["wise_margin_pp"] <= 0 and r2["wise_margin_upper_pp"] >= 2.0 and not r2["ii_fail"] and r2["ii_underpowered"], r2
    assert r2["verdict"] == "灰區", r2["verdict"]
    r2m = gate1(pq2, g1c, ecfg, merged=True)
    assert r2m["verdict"] == "未能測試（檢定力不足）", r2m["verdict"]
    # 3) (ii) 通過 → PASS（位移 +3pp）；(i) 不成立時（rl_hf 與 rr 同高）仍判 FAIL
    pq3, *_ = build(1000, 0.05, +0.03)
    r3 = gate1(pq3, g1c, ecfg)
    assert r3["verdict"] == "PASS" and r3["ii_pass"] and not r3["ii_fail"], r3
    pq4 = dict(pq3); pq4["rl_hf"] = dict(pq3["rl_replace"])
    assert gate1(pq4, g1c, ecfg)["verdict"] == "FAIL"
    mc_pts = r1["mix_curve_descriptive"]["points"]
    assert len(mc_pts) == 21 and mc_pts[0]["pi"] == 0.0 and abs(mc_pts[-1]["pi"] - 1.0) < 1e-12 and all(math.isfinite(q["pk"]) for q in mc_pts)
    print("R1 分支 OK：FAIL(明確)／灰區(檢定力不足)／merged→未能測試／PASS／(i) 不成立→FAIL")
    # ---- R3：probe2 ----
    pcfg = dict(crit["probe2"])
    def with_geo(pq, node, shift, pk_over=None):
        q = dict(pq); nd = pq["wise_%s" % node]
        q["geo_mix"] = {"p1": nd["p1"].copy(), "pk": (nd["pk"] + shift) if pk_over is None else pk_over,
                        "trunc": np.zeros(len(nd["p1"])), "ntok": np.full(len(nd["p1"]), 1000), "repl": np.full(len(nd["p1"]), 40)}
        return q
    base_pq = pq                                  # 情境 1：(i) 成立、(ii) FAIL
    ok_ = probe2_eval(with_geo(base_pq, "0.6", 0.05), g1c, pcfg, ecfg)
    assert ok_["verdict"] == "通過" and ok_["cond1_ok"] and ok_["cond2_ok"] and ok_["cond3_ok"], ok_
    no2 = probe2_eval(with_geo(base_pq, "0.6", 0.0), g1c, pcfg, ecfg)         # 沒高過前緣
    assert no2["verdict"].startswith("未通過") and not no2["cond2_ok"] and no2["cond1_ok"], no2
    no3 = probe2_eval(with_geo(base_pq, "0.3", 0.05), g1c, pcfg, ecfg)         # pass@1 保留不足（節點 0.3 < 0.5×RL 增益）
    assert no3["verdict"].startswith("未通過") and not no3["cond3_ok"] and no3["cond2_ok"], no3
    no1 = probe2_eval(with_geo(base_pq, "0.6", 0.0, pk_over=base_pq["rl_hf"]["pk"] + 0.01), g1c, pcfg, ecfg)  # pass@64 只比 rl_hf 高 1pp
    assert no1["verdict"].startswith("未通過") and not no1["cond1_ok"], no1
    # 觸發：訊號 PASS 而空間 FAIL（主配對）
    PQs = {"A": with_geo(base_pq, "0.6", 0.05)}
    crit_ = crit
    tr = r3_status("A", {"verdict": "PASS"}, {"verdict": "FAIL"}, PQs, crit_)
    assert tr["triggered"] and tr["pair"] == "A" and tr["result"]["verdict"] == "通過"
    assert overall_eval({"verdict": "PASS"}, {"verdict": "PASS"}, {"verdict": "FAIL"}, tr)["overall"].startswith("未能測試")
    tr_no = r3_status("A", {"verdict": "PASS"}, {"verdict": "FAIL"}, {"A": base_pq}, crit_)   # 尚無 geo_mix
    assert tr_no["triggered"] and tr_no["result"]["verdict"].startswith("待執行")
    assert overall_eval({"verdict": "PASS"}, {"verdict": "PASS"}, {"verdict": "FAIL"}, tr_no)["overall"].startswith("待 R3")
    fail_r3 = {"triggered": True, "result": {"verdict": "未通過（x）"}}
    assert overall_eval({"verdict": "PASS"}, {"verdict": "PASS"}, {"verdict": "FAIL"}, fail_r3)["overall"].startswith("FAIL")
    for sg, sp in [("灰區", "FAIL"), ("FAIL", "FAIL"), ("PASS", "PASS"), ("PASS", "灰區")]:
        assert not r3_status("A", {"verdict": sg}, {"verdict": sp}, PQs, crit_)["triggered"]
    assert not r3_status(None, {"verdict": "PASS"}, {"verdict": "FAIL"}, PQs, crit_)["triggered"]
    print("R3 probe2 OK：通過／cond2 失敗／cond3 失敗／cond1 失敗／觸發條件（訊號 PASS 且空間 FAIL）")
    selftest_three_items(crit)


def selftest_three_items(crit):
    """報告 60 第 5 節：現象／訊號／空間各自的 PASS、FAIL、灰區（合成資料）與整體規則。"""
    rng = np.random.default_rng(7)
    ecfg = dict(crit["eval"]); ecfg["bootstrap_B"] = 2000
    crit = dict(crit); crit["eval"] = ecfg
    n = crit["gate1"]["mix_curve"]["n"]; kk = crit["gate1"]["mix_curve"]["k"]
    # ---- 現象 ----
    g_ok = {"reproduce_ok": True, "trunc_ok": True, "verdict": "配對合格"}
    g_unq = {"reproduce_ok": True, "trunc_ok": True, "verdict": "配對不合格（換配對，不算題目死）"}
    g_bad = {"reproduce_ok": False, "trunc_ok": True, "verdict": "未能測試（x）"}
    S = lambda a, b: {"A": {"gate0": a}, "B": {"gate0": b}}
    assert phenomenon_eval(S(g_unq, g_ok), crit) == {"verdict": "PASS", "primary_pair": "B", "scope": "1.5B, 3000 token"}
    assert phenomenon_eval(S(g_ok, g_unq), crit)["primary_pair"] == "A"
    assert phenomenon_eval(S(g_unq, g_unq), crit)["verdict"] == "FAIL"
    assert phenomenon_eval(S(g_unq, g_bad), crit)["verdict"] == "未能測試"
    assert phenomenon_eval({"A": {"gate0": g_unq}}, crit)["verdict"] == "未能測試"      # B 缺資料
    assert phenomenon_eval({"B": {"gate0": g_ok}}, crit)["waiting_for_eligibility"] == "A"
    assert phenomenon_eval(S(g_bad, g_ok), crit)["waiting_for_eligibility"] == "A"
    # ---- 合成逐題資料（與計數一致）----
    def mkq(c, pen=0.0):
        c = np.asarray(c, float)
        return {"p1": c / n, "pk": np.clip(pass_at_k_vec(c, n, kk) - pen, 0, 1), "trunc": np.zeros(len(c)), "ntok": np.full(len(c), 1000), "repl": np.full(len(c), 40)}
    def world(Q, noise_sd, shift, rr_node="0.6", wise_pen=0.0):
        a = rng.uniform(0.02, 0.30, Q); r = np.clip(a + rng.uniform(0.0, 0.5, Q), 0, 1)
        pq = {"base": mkq(np.round(a * n)), "rl": mkq(np.round(r * n))}
        pq["rl_hf"] = dict(pq["rl"])
        for l in crit["gate1"]["wise_lambdas"]:
            pq["wise_%s" % l] = mkq(np.round((a + l * (r - a)) * n), pen=wise_pen)
        node = pq["wise_%s" % rr_node]
        nz = rng.normal(0, noise_sd, Q); nz -= nz.mean()
        pq["rl_replace"] = {**node, "pk": node["pk"] + shift + nz}
        return pq
    # ---- 訊號 ----
    pqs = world(1000, 0.02, 0.0)
    def with_rr(pq, rec, delta, noise=0.0):
        q = dict(pq); b, r = pq["base"], pq["rl_hf"]
        nz = rng.normal(0, noise, len(b["p1"])); nz -= nz.mean()
        q["rl_replace"] = {**pq["rl_replace"], "p1": b["p1"] + rec * (r["p1"] - b["p1"]), "pk": r["pk"] + delta + nz}
        return q
    sp = signal_eval(with_rr(pqs, 0.8, 0.04, 0.02), crit)
    assert sp["verdict"] == "PASS" and sp["q1_ok"] and sp["q2_ok"], sp
    sf = signal_eval(with_rr(pqs, 0.8, 0.002, 0.02), crit)                       # 量 1 < 1pp
    assert sf["verdict"] == "FAIL", sf
    sg1 = signal_eval(with_rr(pqs, 0.8, 0.015, 0.02), crit)                      # 1pp <= 差 < 2pp
    assert sg1["verdict"] == "灰區" and not sg1["q1_ok"], sg1
    sg2 = signal_eval(with_rr(pqs, 0.3, 0.04, 0.02), crit)                       # 量 1 成立、回收率不足
    assert sg2["verdict"] == "灰區" and sg2["q1_ok"] and not sg2["q2_ok"], sg2
    sg3 = signal_eval(with_rr(world(60, 0.02, 0.0), 0.8, 0.025, 0.30), crit)     # 差 >= 2pp 但下界 <= 0
    assert sg3["verdict"] == "灰區" and not sg3["q1_ok"], sg3
    assert signal_eval(with_rr(pqs, 0.8, 0.015, 0.02), crit, merged=True)["verdict"].startswith("灰區（加碼後")
    print("訊號 OK：PASS／FAIL／灰區（差在 1–2pp、回收率不足、下界不過）")
    # ---- 空間 ----
    sp_pass = space_eval(world(1000, 0.05, +0.03), crit)                          # H_a 通過
    assert sp_pass["verdict"] == "PASS" and "H_a" in sp_pass["paths_passing"], sp_pass
    sp_hb = space_eval(world(1000, 0.05, 0.0, wise_pen=0.05), crit)               # WiSE 被壓低 5pp：H_a≈0、H_b>2
    assert sp_hb["verdict"] == "PASS" and sp_hb["paths_passing"] == ["H_b"] and sp_hb["H_a"]["pt_pp"] < 2.0, sp_hb
    sp_fail = space_eval(world(1000, 0.05, -0.03), crit)                          # H_a < 0；H_b≈0
    assert sp_fail["verdict"] == "FAIL", sp_fail
    sp_gray = space_eval(world(60, 0.30, -0.005), crit)                           # 檢定力不足
    assert sp_gray["verdict"] == "灰區", sp_gray
    assert space_eval(world(60, 0.30, -0.005), crit, merged=True)["verdict"].startswith("未能測試（檢定力不足）")
    assert space_eval({"base": pqs["base"]}, crit)["verdict"].startswith("未能測試")
    # π* 選取：同距取較小
    p1c = np.array([0.1, 0.2, 0.3]); pis_ = [0.0, 0.5, 1.0]; tgt = 0.25
    assert pis_[int(np.argmin(np.abs(p1c - tgt) + 1e-12 * np.arange(3)))] == 0.5
    print("空間 OK：PASS(H_a)／PASS(僅 H_b)／FAIL／灰區／合併後未能測試；α=%.5f" % sp_pass["alpha_one_sided"])
    # ---- 整體 ----
    P, F, G = {"verdict": "PASS"}, {"verdict": "FAIL"}, {"verdict": "灰區"}
    nt = {"triggered": False, "result": {"verdict": "未觸發"}}
    assert overall_eval(P, P, P, nt)["overall"].startswith("PASS")
    assert overall_eval(P, F, P, nt)["overall"].startswith("FAIL") and "訊號" in overall_eval(P, F, P, nt)["overall"]
    assert overall_eval(P, P, G, nt)["overall"].startswith("灰區")
    assert overall_eval({"verdict": "未能測試"}, {"verdict": "未評估"}, {"verdict": "未評估"}, nt)["overall"].startswith("未能測試")
    print("整體 OK")


if __name__ == "__main__":
    main()

</file>

<file name="scripts/artifacts.py">
"""Sample, shard and grading evidence for formally evaluated conditions."""

import ast
import gzip
import hashlib
import json
import os
from contextlib import contextmanager
from pathlib import Path
import re
import tempfile
from importlib.metadata import version

from common import ROOT, STOP_IDS, cfg, load_data, shard


REPO = Path(ROOT)
SAMPLE_SEED = 1234


class ArtifactError(ValueError):
    """An incomplete or unverifiable result."""


def require(ok, message):
    if not ok:
        raise ArtifactError(message)


def sha256(path):
    with open(path, "rb") as f:
        return hashlib.file_digest(f, "sha256").hexdigest()


def read_json(path):
    with open(path, encoding="utf-8") as f:
        return json.load(f)


def atomic_json(path, value):
    path = Path(path)
    with tempfile.NamedTemporaryFile(mode="w", dir=path.parent, prefix=f".{path.name}.", suffix=".tmp",
                                     encoding="utf-8", delete=False) as handle:
        tmp = Path(handle.name)
        try:
            json.dump(value, handle, ensure_ascii=False, allow_nan=False)
            handle.close()
            tmp.replace(path)
        finally:
            tmp.unlink(missing_ok=True)


@contextmanager
def publish_raw(path):
    """Only one writer may publish a raw shard; partial files are private."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    fd, name = tempfile.mkstemp(dir=path.parent, prefix=f".{path.name}.", suffix=".tmp")
    os.close(fd)
    tmp = Path(name)
    try:
        yield tmp
        os.link(tmp, path)
    finally:
        tmp.unlink(missing_ok=True)


def dataset_hash(path=None):
    return sha256(path or REPO / "data/eval.jsonl")


def pairs_hash():
    return sha256(REPO / "configs/pairs.json")


def decode_ast_hash():
    source = REPO / "scripts/gen_replace.py"
    nodes = ast.parse(source.read_text(encoding="utf-8")).body
    functions = [ast.dump(node) for node in nodes if isinstance(node, ast.FunctionDef) and node.name != "main"]
    return hashlib.sha256("\n".join(functions).encode()).hexdigest()


def grader_receipt():
    math_verify = version("math-verify")
    require(math_verify == "0.9.0", "Math-Verify 版本與凍結設定不符")
    return {"common_sha256": sha256(REPO / "scripts/common.py"), "math_verify": math_verify}


def artifact_receipt(raw, pair, cond, engine, job_id, dtype, context, data=None):
    producer = "gen_vllm.py" if engine == "vllm" else "gen_replace.py"
    return {"raw_sha256": sha256(raw), "pair": pair, "condition": cond, "engine": engine,
            "run_id": os.environ.get("RSD_RUN_ID") or job_id, "job_id": job_id,
            "dtype": dtype, "max_model_len": context, "stop_ids": STOP_IDS,
            "eval_sha256": dataset_hash(data), "pairs_sha256": pairs_hash(),
            "producer_sha256": sha256(REPO / "scripts" / producer),
            "engine_version": version("vllm" if engine == "vllm" else "transformers")}


def parse_shard(value):
    match = re.fullmatch(r"([0-9]+)/([1-9][0-9]*)", value)
    require(match is not None, f"無效分片：{value}")
    i, n = map(int, match.groups())
    require(i < n, f"分片索引超界：{value}")
    return i, n


def model_matches(path, repo_id, snapshot=None):
    parts = Path(path).parts
    if path == repo_id and snapshot is None:
        return True
    return (len(parts) >= 3 and parts[-3] == "models--" + repo_id.replace("/", "--")
            and parts[-2] == "snapshots" and re.fullmatch(r"[0-9a-f]{40}", parts[-1]) is not None
            and (snapshot is None or parts[-1] == snapshot))


def historical_profile(pair):
    return cfg("legacy_artifacts.json").get(pair)


def check_gate_minus1(results, pair, pc):
    profile = historical_profile(pair)
    receipt_path = Path(results) / f"correctness_{pair}.json"
    log_path = Path(results) / f"correctness_{pair}.log"
    log = log_path.read_text(encoding="utf-8")
    require(all(re.search(rf"^PASS T{n}\b", log, re.M) for n in range(1, 6)) and "ALL PASS" in log.splitlines(),
            "缺完整正確性測試 T1–T5 PASS")
    if receipt_path.exists():
        receipt = read_json(receipt_path)
        require(receipt["pair"] == pair and receipt["pairs_sha256"] == pairs_hash()
                and receipt["eval_sha256"] == dataset_hash() and receipt["log_sha256"] == sha256(log_path)
                and receipt["decode_sha256"] == sha256(REPO / "scripts/gen_replace.py")
                and receipt["tests_sha256"] == sha256(REPO / "tests/smoke_cpu.py")
                and receipt["device"] == "cuda" and receipt["dtype"] == "float32" and receipt["job_id"],
                "正確性憑證與目前模型／設定／測試不符")
        for side in ("base", "rl"):
            require(model_matches(receipt[side], pc["pairs"][pair][side]), f"正確性測試 {side} 模型不符")
        return {"job_id": receipt["job_id"], "source": "receipt", "base": receipt["base"], "rl": receipt["rl"],
                "geo_mix_checks": all(re.search(rf"^PASS {n}\b", log, re.M) for n in ("T6", "T7a", "T7b", "T8"))}
    require(profile is not None and profile["pairs_sha256"] == pairs_hash() and profile["eval_sha256"] == dataset_hash()
            and profile["decode_ast_sha256"] == decode_ast_hash(),
            "缺正確性測試憑證")
    proof = profile["gate_minus1"]
    batch = REPO / f"slurm/logs/rsd-correct-{proof['job_id']}.out"
    require(sha256(log_path) == proof["receipt_log_sha256"] and sha256(batch) == proof["job_log_sha256"],
            "歷史正確性測試 log 不符")
    require(log in batch.read_text(encoding="utf-8"), "歷史正確性測試與 Slurm log 不符")
    return {"job_id": proof["job_id"], "source": "historical_batch_log", "geo_mix_checks": False, "snapshots": {
        side: profile[f"{side}_snapshot"] for side in ("base", "rl")}}


def record_correctness(results, pair, base, rl, job_id):
    root = Path(results)
    out = root / f"correctness_{pair}.json"
    log = root / f"correctness_{pair}.log"
    atomic_json(out, {"pair": pair, "base": base, "rl": rl, "job_id": job_id,
                      "log_sha256": sha256(log), "pairs_sha256": pairs_hash(), "eval_sha256": dataset_hash(),
                      "decode_sha256": sha256(REPO / "scripts/gen_replace.py"),
                      "tests_sha256": sha256(REPO / "tests/smoke_cpu.py"), "device": "cuda", "dtype": "float32"})


def check_run(directory, pair, cond, pc, ecfg, data, gate_receipt=None):
    directory = Path(directory)
    greedy = cond.endswith("_greedy")
    split = "all" if greedy else "test"
    n = 1 if greedy else ecfg["n_samples"]
    temp, top_p = (0, 1) if greedy else (pc["temperature"], pc["top_p"])
    hf = cond in ("rl_hf", "rl_replace", "geo_mix")
    if cond == "geo_mix" and gate_receipt is not None:
        require(gate_receipt.get("geo_mix_checks"), "缺幾何混合 T6–T8 正確性憑證")
    engine = "hf-lockstep" if hf else "vllm"
    raw_files = list(directory.glob("raw_*.jsonl.gz"))
    completion_files = list(directory.glob("meta_*.json"))
    require(raw_files and completion_files, "缺取樣分片或完成憑證")
    rows = [r for r in data if split == "all" or r["split"] == split]
    shards = {}
    historical = historical_profile(pair)
    for meta_file in completion_files:
        m = read_json(meta_file)
        index, count = parse_shard(m["shard"])
        require(meta_file.name == f"meta_{index}of{count}.json" and index not in shards,
                "分片憑證檔名或索引不符")
        raw = directory / f"raw_{index}of{count}.jsonl.gz"
        require(raw in raw_files, f"完成憑證缺 raw 分片 {index}/{count}")
        require((m["engine"], m["split"], m["n"], m["temperature"], m["top_p"], m["seed"],
                 m["max_new" if hf else "max_tokens"]) == (engine, split, n, temp, top_p, SAMPLE_SEED, pc["max_new_tokens"]),
                "引擎、樣本數、解碼、切分或 seed 與凍結設定不符")
        if hf:
            require(m["cond"] == ("rl" if cond == "rl_hf" else cond) and m["compact"] is True
                    and (cond != "geo_mix" or m["lam"] == 0.5), "HF 條件或設定不符")
            require(m.get("rows_per_batch") == 256, "HF 批次大小與凍結的 seed 排程不符")
            models = {side: m[side] for side in ("base", "rl")}
        elif cond.startswith("wise_"):
            require(m["wise_lambda"] == float(cond.split("_", 1)[1]), "WiSE 權重不符")
            models = m["wise_models"]
            require(set(models) == {"base", "rl"}, "WiSE 缺少兩個來源模型")
        else:
            models = {cond.removesuffix("_greedy"): m["model"]}
        for side, model in models.items():
            pinned = gate_receipt.get("snapshots", {}).get(side) if gate_receipt else None
            require(model_matches(model, pc["pairs"][pair][side], pinned), f"{side} 模型或 snapshot 不符")
            if gate_receipt and side in gate_receipt:
                require(Path(model).name == Path(gate_receipt[side]).name, f"{side} snapshot 與正確性測試不符")
        if "provenance" in m:
            p = m["provenance"]
            producer = "gen_vllm.py" if engine == "vllm" else "gen_replace.py"
            require(p["pair"] == pair and p["condition"] == cond and p["engine"] == engine
                    and p["dtype"] == pc["dtype"] and p["max_model_len"] == pc["max_model_len"]
                    and p["stop_ids"] == STOP_IDS and p["eval_sha256"] == dataset_hash()
                    and p["pairs_sha256"] == pairs_hash() and p["run_id"] and p["job_id"]
                    and p["producer_sha256"] == sha256(REPO / "scripts" / producer) and p["engine_version"]
                    and p["raw_sha256"] == sha256(raw), "取樣完成憑證不相容")
            run_id = p["run_id"]
        else:
            require(historical is not None and cond in historical["runs"] and index == 0 and count == 1,
                    "取樣缺已登記的來源憑證")
            receipt = historical["runs"][cond]
            batch = REPO / f"slurm/logs/rsd-vllm-{receipt['job_id']}.out"
            require(historical["eval_sha256"] == dataset_hash() and historical["pairs_sha256"] == pairs_hash()
                    and sha256(batch) == receipt["log_sha256"], "歷史用資料／設定或 Slurm log 不符")
            log = batch.read_text(encoding="utf-8")
            require(f"done results/{pair}/{cond}/{raw.name}" in log and m["model"] in log
                    and f"dtype=torch.{pc['dtype']}" in log and f"max_seq_len={pc['max_model_len']}" in log,
                    "歷史 log 缺取樣完成或模型設定證據")
            run_id = f"historical-{receipt['job_id']}"
        shards[index] = (count, run_id, raw, meta_file)
    sizes = {s[0] for s in shards.values()}
    require(len(sizes) == 1, "分片總數不一致")
    size, = sizes
    require(set(shards) == set(range(size)) and len(raw_files) == size, "分片集合不完整或有多餘 raw 檔")
    require(len({s[1] for s in shards.values()}) == 1, "取樣分片屬於不同 run")
    return [(shards[i][2], shards[i][3], [r["id"] for r in shard(rows, i, size)]) for i in range(size)], n


def validate_records(records, ids, n, max_tokens, need_repl, raw=False):
    ids = list(ids)
    require(set(records) == set(ids) and len(records) == len(ids), "題目 ID 與凍結分片不一致")
    for i in ids:
        row = records[i]
        keys = ["finished", "ntok", "texts" if raw else "correct"]
        if need_repl or "n_repl" in row:
            keys.append("n_repl")
        require(all(isinstance(row.get(k), list) and len(row[k]) == n for k in keys),
                f"題目 {i} 的樣本數或欄位長度不符")
        require(all(type(x) is int and x in (0, 1) for x in row["finished"]), f"題目 {i} 結束狀態無效")
        require(all(type(x) is int and 0 <= x <= max_tokens for x in row["ntok"]), f"題目 {i} token 數無效")
        if "n_repl" in keys:
            require(all(type(x) is int and 0 <= x <= k for x, k in zip(row["n_repl"], row["ntok"], strict=True)),
                    f"題目 {i} 替換次數無效")
        if raw:
            require(all(isinstance(t, str) for t in row["texts"]), f"題目 {i} 生成資料格式無效")
        else:
            require(all(type(c) is int and 0 <= c <= f for c, f in zip(row["correct"], row["finished"], strict=True)),
                    f"題目 {i} 的對錯值無效或截斷樣本被判對")


def load_raw_bundle(bundle, n, max_tokens, cond):
    records = {}
    for raw, _, ids in bundle:
        # 檢查每個分片的 ID；不可把另一分片的題目挪來補缺。
        with gzip.open(raw, "rt", encoding="utf-8") as source:
            block = {}
            for line in source:
                r = json.loads(line)
                require(r["id"] not in block, f"重複題目 ID：{r['id']}")
                block[r["id"]] = r
        validate_records(block, ids, n, max_tokens, cond in ("rl_hf", "rl_replace", "geo_mix"), raw=True)
        records.update(block)
    return records


def input_receipts(bundle):
    return {path.name: sha256(path) for raw, meta, _ in bundle for path in (raw, meta)}


def load_verified(directory, pair, cond, pc, ecfg, data, gate_receipt):
    directory = Path(directory)
    bundle, n = check_run(directory, pair, cond, pc, ecfg, data, gate_receipt)
    receipt = read_json(directory / "graded_receipt.json")
    grade = directory / "graded_verified.json"
    require(receipt["grader"] == grader_receipt() and receipt["inputs"] == input_receipts(bundle)
            and receipt["grade_sha256"] == sha256(grade) and receipt["pair"] == pair and receipt["condition"] == cond,
            "缺相容判分憑證，或判分後輸入檔已變更")
    graded = read_json(grade)
    ids = [i for _, _, shard_ids in bundle for i in shard_ids]
    validate_records(graded, ids, n, pc["max_new_tokens"], cond in ("rl_hf", "rl_replace", "geo_mix"))
    return graded

</file>

<file name="scripts/gen_vllm.py">
"""單一模型取樣（base／rl／WiSE-FT 點）。vLLM。raw 輸出：raw_{shard}.jsonl.gz（完整生成，gzip）。

用法例：
  gen_vllm.py --model PATH --out results/A/base --split test --n 128 --temperature 0.6 --top-p 0.95
  gen_vllm.py --model PATH --out results/A/rl_greedy --split all --n 1 --temperature 0 --top-p 1   # 閘門 0 重現用
"""
import argparse, os, sys, json
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from common import *
from artifacts import artifact_receipt, atomic_json, parse_shard, publish_raw, require, sha256, read_json

ap = argparse.ArgumentParser()
ap.add_argument("--model", required=True)
ap.add_argument("--out", required=True)
ap.add_argument("--split", default="test")
ap.add_argument("--n", type=int, default=128)
ap.add_argument("--temperature", type=float, default=None)
ap.add_argument("--top-p", type=float, default=None)
ap.add_argument("--shard", default="0/1")
ap.add_argument("--seed", type=int, default=1234)
ap.add_argument("--limit", type=int, default=None, help="只跑前 N 題（吞吐實測用）")
ap.add_argument("--gpu-mem", type=float, default=0.90)
a = ap.parse_args()

pc = pairs_cfg()
temp = pc["temperature"] if a.temperature is None else a.temperature
top_p = pc["top_p"] if a.top_p is None else a.top_p
si, sn = parse_shard(a.shard)
rows = shard(load_data(a.split), si, sn)
if a.limit:
    rows = rows[: a.limit]
outp = os.path.join(a.out, f"raw_{si}of{sn}.jsonl.gz")
meta_path = os.path.join(a.out, f"meta_{si}of{sn}.json")
if os.path.exists(outp):
    if os.environ.get("RSD_RUN_ID"):
        p = read_json(meta_path).get("provenance", {})
        require(p.get("run_id") == os.environ["RSD_RUN_ID"] and p.get("raw_sha256") == sha256(outp),
                "existing raw shard is not from this run")
    print("exists, skip", outp); sys.exit(0)

from vllm import LLM, SamplingParams
llm = LLM(a.model, dtype=pc["dtype"], max_model_len=pc["max_model_len"], gpu_memory_utilization=a.gpu_mem,
          seed=a.seed, enable_prefix_caching=True, tokenizer=os.environ.get("TOKENIZER_PATH") or None)
sp = SamplingParams(n=a.n, temperature=temp, top_p=top_p, max_tokens=pc["max_new_tokens"],
                    stop_token_ids=STOP_IDS, seed=None, skip_special_tokens=True)
outs = llm.generate([build_prompt(r["question"], pc) for r in rows], sp)
with publish_raw(outp) as tmp, open_gz_write(tmp) as f:
    for r, o in zip(rows, outs, strict=True):
        f.write(json.dumps({"id": r["id"], "texts": [c.text for c in o.outputs],
                            "finished": [int(c.finish_reason == "stop") for c in o.outputs],
                            "ntok": [len(c.token_ids) for c in o.outputs]}) + "\n")
meta = {"model": a.model, "temperature": temp, "top_p": top_p, "n": a.n, "max_tokens": pc["max_new_tokens"],
        "split": a.split, "shard": a.shard, "seed": a.seed, "engine": "vllm"}
if os.environ.get("RSD_RUN_ID") and not a.limit:
    meta["provenance"] = artifact_receipt(outp, os.environ["PAIR"], os.path.basename(a.out), "vllm",
                                          os.environ.get("SLURM_JOB_ID"), pc["dtype"], pc["max_model_len"])
    if "RSD_WISE_LAMBDA" in os.environ:
        meta["wise_lambda"] = float(os.environ["RSD_WISE_LAMBDA"])
        meta["wise_models"] = {"base": os.environ["RSD_WISE_BASE"], "rl": os.environ["RSD_WISE_RL"]}
atomic_json(meta_path, meta)
print("done", outp)

</file>

<file name="scripts/gen_replace.py">
"""HF 雙模型逐步解碼（rl-replace，照 2610.24663 公開程式 wrb.decoding.rl_replace 的規則）。

規則（逐字）：b_t=argmax p_B, r_t=argmax p_R；b_t != r_t 則輸出 r_t，否則從 p_B 以 temperature 0.6、top-p 0.95 取樣。
取樣函式直接 import 原作 wrb.decoding.sampling（確保一致）。與原作迴圈的差異（皆為工程性）：
  1. 多題合批：左側 padding + 明確 position_ids（原作只批同一題，無 padding）。
  2. 已結束的列會從批次中移除（KV cache 以 batch_select_indices 壓縮），不再浪費算力；
     這會改變 RNG 消耗順序，所以與原作「逐 token 相同」只在 --no-compact 且單題批次時成立（tests/ 內驗證）。
conds：
  base         只用 base 取樣（HF 版對照，閘門用）
  rl           只用 rl 取樣（temperature 0 即 greedy）
  rl_replace   混合解碼（主角）
  all_replace  每個位置都強制輸出 r_t（閘門：必須等於 rl greedy）
  geo_mix      2026-11-14（第二探針 R3，見 PREREG）：不一致位置（argmax 不同）改從幾何混合
               softmax((1-lam)·logit_base + lam·logit_RL)（同 temperature、top-p）取樣，其餘位置照 base 取樣。
               lam=0 → 與 base 取樣相同；lam=1 → 不一致位置從 RL 的分布取樣（rl_replace 則是確定性輸出 RL 的 argmax）。
               兩個模型相同時沒有不一致位置，輸出逐位元等於 base 取樣（同 seed）。n_repl 記錄不一致位置數。
輸出：raw_{shard}.jsonl.gz，每題含 texts、finished、ntok、n_repl（被替換的位置數）。
"""
import argparse, json, os, sys, time
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from common import *

sys.path.insert(0, WRB)  # 原作程式（只 import 取樣函式）
import torch
from wrb.decoding.sampling import sample_from_logits, greedy_from_logits  # noqa: E402
from transformers import AutoModelForCausalLM, AutoTokenizer, DynamicCache  # noqa: E402


def fwd(model, ids, attn, pos, cache):
    out = model(input_ids=ids, attention_mask=attn, position_ids=pos, past_key_values=cache,
                use_cache=True, logits_to_keep=1)
    return out.logits[:, -1, :], out.past_key_values


@torch.no_grad()
def decode_batch(tok, mb, mr, prompts, n, cond, temperature, top_p, max_new, max_len, gen, compact=True, log=None, lam=0.5):
    dev = next((mb or mr).parameters()).device
    texts = [p for p in prompts for _ in range(n)]
    enc = tok(texts, return_tensors="pt", padding=True, add_special_tokens=False)
    ids, attn = enc["input_ids"].to(dev), enc["attention_mask"].to(dev)
    B, L = ids.shape
    max_new = min(max_new, max_len - L)
    pos = (attn.cumsum(-1) - 1).clamp(min=0)
    need_b = cond in ("base", "rl_replace", "all_replace", "geo_mix")
    need_r = cond in ("rl", "rl_replace", "all_replace", "geo_mix")
    cb = DynamicCache() if need_b else None
    cr = DynamicCache() if need_r else None
    stop = torch.tensor(STOP_IDS, device=dev)
    out_tok = torch.full((B, max_new), -1, dtype=torch.long, device=dev)
    n_repl = torch.zeros(B, dtype=torch.long, device=dev)
    finished = torch.zeros(B, dtype=torch.bool, device=dev)
    rows = torch.arange(B, device=dev)  # 目前批次各列對應的原始列
    cur_ids, cur_attn, cur_pos = ids, attn, pos
    steps_rows = 0
    act = torch.ones(B, dtype=torch.bool, device=dev)  # 目前批次中尚未結束的列
    for t in range(max_new):
        lb = lr = None
        if need_b:
            lb, cb = fwd(mb, cur_ids, cur_attn, cur_pos, cb)
        if need_r:
            lr, cr = fwd(mr, cur_ids, cur_attn, cur_pos, cr)
        if cond == "base":
            nxt = sample_from_logits(lb, temperature, top_p, generator=gen)
            dis = torch.zeros_like(nxt, dtype=torch.bool)
        elif cond == "rl":
            nxt = sample_from_logits(lr, temperature, top_p, generator=gen)
            dis = torch.zeros_like(nxt, dtype=torch.bool)
        elif cond == "geo_mix":
            btk, rtk = greedy_from_logits(lb), greedy_from_logits(lr)
            dis = btk != rtk
            # 只在不一致位置混合；其餘位置的 logits 與 base 完全相同（逐位元），故兩模型相同時等於 base 取樣。
            # 以 float32 混合再轉回原 dtype；lam=1 時 0*lb+1*lr 在 lb 有限時恰等於 lr。
            mix = ((1.0 - lam) * lb.float() + lam * lr.float()).to(lb.dtype)
            nxt = sample_from_logits(torch.where(dis.unsqueeze(-1), mix, lb), temperature, top_p, generator=gen)
        else:
            btk, rtk = greedy_from_logits(lb), greedy_from_logits(lr)
            dis = btk != rtk
            if cond == "all_replace":
                dis = torch.ones_like(dis)
            sampled = sample_from_logits(lb, temperature, top_p, generator=gen)
            nxt = torch.where(dis, rtk, sampled)
        steps_rows += rows.numel()
        w = rows[act]
        out_tok[w, t] = nxt[act]
        if cond in ("rl_replace", "all_replace", "geo_mix"):
            n_repl[w] += dis[act].long()
        done = (nxt.unsqueeze(-1) == stop).any(-1) & act
        finished[rows[done]] = True
        act = act & ~done
        if not bool(act.any()):
            break
        if compact and int(act.sum()) < 0.75 * rows.numel():
            keep = torch.nonzero(act).flatten()
            rows, nxt = rows[keep], nxt[keep]
            cur_attn, cur_pos = cur_attn[keep], cur_pos[keep]
            for c in (cb, cr):
                if c is not None:
                    c.batch_select_indices(keep)
            act = torch.ones_like(keep, dtype=torch.bool)
        cur_ids = nxt.unsqueeze(-1)
        cur_attn = torch.cat([cur_attn, torch.ones_like(cur_attn[:, :1])], dim=-1)
        cur_pos = cur_pos[:, -1:] + 1
    res = []
    out_tok = out_tok.cpu()
    for i in range(B):
        row = out_tok[i]
        m = int((row >= 0).sum())
        toks = row[:m].tolist()
        fin = bool(finished[i])
        if toks and toks[-1] in STOP_IDS and fin:
            body = toks[:-1]
        else:
            body = toks
        res.append((tok.decode(body, skip_special_tokens=True), fin, m, int(n_repl[i])))
    if log is not None:
        log["steps_rows"] = steps_rows
    return res


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--cond", required=True, choices=["base", "rl", "rl_replace", "all_replace", "geo_mix"])
    ap.add_argument("--lam", type=float, default=None, help="geo_mix 的 RL 權重 λ（geo_mix 必填；其他 cond 不可給）")
    ap.add_argument("--base-model", required=True)
    ap.add_argument("--rl-model", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--split", default="test")
    ap.add_argument("--n", type=int, default=128)
    ap.add_argument("--temperature", type=float, default=None)
    ap.add_argument("--top-p", type=float, default=None)
    ap.add_argument("--rows-per-batch", type=int, default=256, help="每批列數（題數 = rows/n）")
    ap.add_argument("--shard", default="0/1")
    ap.add_argument("--seed", type=int, default=1234)
    ap.add_argument("--limit", type=int, default=None)
    ap.add_argument("--no-compact", action="store_true")
    ap.add_argument("--device", default="cuda")
    ap.add_argument("--dtype", default=None)
    ap.add_argument("--max-new", type=int, default=None)
    ap.add_argument("--data", default=None)
    a = ap.parse_args()

    if a.cond == "geo_mix":
        assert a.lam is not None and 0.0 <= a.lam <= 1.0, "--cond geo_mix 需要 --lam in [0,1]"
    else:
        assert a.lam is None, "--lam 只用於 --cond geo_mix"
    pc = pairs_cfg()
    temp = pc["temperature"] if a.temperature is None else a.temperature
    top_p = pc["top_p"] if a.top_p is None else a.top_p
    from artifacts import artifact_receipt, atomic_json, parse_shard, publish_raw, require, sha256, read_json
    si, sn = parse_shard(a.shard)
    rows = shard(load_data(a.split, a.data), si, sn)
    if a.limit:
        rows = rows[: a.limit]
    outp = os.path.join(a.out, f"raw_{si}of{sn}.jsonl.gz")
    meta_path = os.path.join(a.out, f"meta_{si}of{sn}.json")
    if os.path.exists(outp):
        if os.environ.get("RSD_RUN_ID"):
            p = read_json(meta_path).get("provenance", {})
            require(p.get("run_id") == os.environ["RSD_RUN_ID"] and p.get("raw_sha256") == sha256(outp),
                    "existing raw shard is not from this run")
        print("exists, skip", outp); return
    dt = getattr(torch, a.dtype or pc["dtype"])
    tok = AutoTokenizer.from_pretrained(a.base_model, padding_side="left")
    if tok.pad_token is None:
        tok.pad_token = "<|endoftext|>"
    load = lambda p: AutoModelForCausalLM.from_pretrained(p, torch_dtype=dt).to(a.device).eval()
    need_b = a.cond in ("base", "rl_replace", "all_replace", "geo_mix")
    need_r = a.cond in ("rl", "rl_replace", "all_replace", "geo_mix")
    mb = load(a.base_model) if need_b else None
    mr = (mb if (need_r and a.rl_model == a.base_model and mb is not None) else load(a.rl_model)) if need_r else None
    if mb is not None and mr is not None:
        assert mb.config.vocab_size == mr.config.vocab_size, "vocab size differs"
    max_new = a.max_new or pc["max_new_tokens"]
    qpb = max(1, a.rows_per_batch // a.n)
    os.makedirs(a.out, exist_ok=True)
    t0 = time.time()
    with publish_raw(outp) as tmp, open_gz_write(tmp) as f:
        for bi in range(0, len(rows), qpb):
            chunk = rows[bi: bi + qpb]
            gen = torch.Generator(device=a.device)
            gen.manual_seed(a.seed + 1000 * si + bi)
            lg = {}
            res = decode_batch(tok, mb, mr, [build_prompt(r["question"], pc) for r in chunk], a.n, a.cond,
                               temp, top_p, max_new, pc["max_model_len"], gen, compact=not a.no_compact, log=lg,
                               lam=(a.lam if a.cond == "geo_mix" else 0.5))
            for qi, r in enumerate(chunk):
                part = res[qi * a.n:(qi + 1) * a.n]
                f.write(json.dumps({"id": r["id"], "texts": [p[0] for p in part], "finished": [int(p[1]) for p in part],
                                    "ntok": [p[2] for p in part], "n_repl": [p[3] for p in part]}) + "\n")
            f.flush()
            print(f"batch {bi // qpb + 1}/{(len(rows) + qpb - 1) // qpb} elapsed {time.time() - t0:.0f}s "
                  f"row-steps {lg['steps_rows']}", flush=True)
    meta = {"cond": a.cond, "base": a.base_model, "rl": a.rl_model, "temperature": temp, "top_p": top_p, "n": a.n,
            "max_new": max_new, "split": a.split, "shard": a.shard, "seed": a.seed, "compact": not a.no_compact,
            "rows_per_batch": a.rows_per_batch, "engine": "hf-lockstep", "seconds": time.time() - t0}
    if a.cond == "geo_mix":
        meta["lam"] = a.lam
    if os.environ.get("RSD_RUN_ID") and not a.limit:
        meta["provenance"] = artifact_receipt(outp, os.environ["PAIR"], os.path.basename(a.out), "hf-lockstep",
                                              os.environ.get("SLURM_JOB_ID"), a.dtype or pc["dtype"],
                                              pc["max_model_len"], a.data)
    atomic_json(meta_path, meta)


if __name__ == "__main__":
    main()

</file>

<file name="scripts/grade.py">
"""正式結果驗證全分片後判分；保留 grade.py DIR 作既有試驗用入口。"""
import argparse
from contextlib import contextmanager
import fcntl
from pathlib import Path
import os
import sys
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from common import cfg, grade_raw, load_data, pairs_cfg
from artifacts import (atomic_json, check_gate_minus1, check_run, grader_receipt, input_receipts,
                       load_raw_bundle, load_verified, require, sha256, validate_records)


@contextmanager
def grading_lock(directory):
    with (directory / ".grade.lock").open("a", encoding="utf-8") as lock:
        fcntl.flock(lock, fcntl.LOCK_EX)
        yield


def verified_grade(directory, pair, cond, workers, n_shards):
    pc, ecfg, data = pairs_cfg(), cfg("criteria.json")["eval"], load_data("all")
    gate = check_gate_minus1(directory.parent.parent, pair, pc)
    if n_shards is not None:
        expected = {f"meta_{i}of{n_shards}.json" for i in range(n_shards)}
        seen = {p.name for p in directory.glob("meta_*.json")}
        require(seen <= expected, "發現不相容或舊分片憑證")
        if seen != expected:
            print("等待全部分片完成憑證")
            return
    bundle, n = check_run(directory, pair, cond, pc, ecfg, data, gate)
    inputs = input_receipts(bundle)
    if (directory / "graded_receipt.json").exists():
        load_verified(directory, pair, cond, pc, ecfg, data, gate)
        print("已存在有效的正式判分憑證")
        return
    raw = load_raw_bundle(bundle, n, pc["max_new_tokens"], cond)
    stage = directory / ".graded_verify_stage.json"
    try:
        grader = grader_receipt()
        graded = grade_raw([str(path) for path, _, _ in bundle], stage, workers)
        validate_records(graded, raw.keys(), n, pc["max_new_tokens"], cond in ("rl_hf", "rl_replace", "geo_mix"))
        for i, rec in raw.items():
            fields = ["finished", "ntok"] + (["n_repl"] if "n_repl" in rec else [])
            require(all(graded[i][key] == rec[key] for key in fields), f"判分與取樣資料不對齊：{i}")
        require(input_receipts(bundle) == inputs, "判分中取樣分片或憑證受到變更")
        target = directory / "graded_verified.json"
        stage.replace(target)
        atomic_json(directory / "graded_receipt.json", {"pair": pair, "condition": cond, "grader": grader,
                                                        "inputs": inputs, "grade_sha256": sha256(target)})
        print(f"正式判分完成：{target}")
    finally:
        stage.unlink(missing_ok=True)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("dir", type=Path)
    ap.add_argument("--workers", type=int, default=8)
    ap.add_argument("--pair")
    ap.add_argument("--condition")
    ap.add_argument("--n-shards", type=int)
    args = ap.parse_args()
    if bool(args.pair) != bool(args.condition):
        ap.error("--pair 與 --condition 必須一併指定")
    with grading_lock(args.dir):
        if args.pair:
            return verified_grade(args.dir, args.pair, args.condition, args.workers, args.n_shards)
        raws = sorted(args.dir.glob("raw_*.jsonl.gz"))
        require(raws, f"no raw files in {args.dir}")
        out = grade_raw(raws, args.dir / "graded.json", args.workers)
        n = sum(len(v["correct"]) for v in out.values())
        print(args.dir, "questions", len(out), "samples", n, "pass@1", sum(sum(v["correct"]) for v in out.values()) / n)


if __name__ == "__main__":
    main()

</file>

<file name="slurm/00_correctness_gpu.sbatch">
#!/bin/bash -l
#SBATCH --partition=h200q
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=16
#SBATCH --mem=170G
#SBATCH --output=/work/project/icml27-topic-search/killtests/rl-sparse-distill/slurm/logs/%x-%j.out
#SBATCH --job-name=rsd-correct
#SBATCH --time=01:00:00
# 用法：sbatch slurm/00_correctness_gpu.sbatch A   （真實模型、fp32、GPU 上重跑 smoke 閘門 T1-T5）
export PAIR=${1:-A}
set -e
source /work/project/icml27-topic-search/killtests/rl-sparse-distill/slurm/env.sh
$PY -I tests/smoke_cpu.py $BASE $RL cuda 64 | tee results/correctness_$PAIR.log
$PY -B -c 'import sys; sys.path.insert(0, "scripts"); from artifacts import record_correctness; record_correctness(*sys.argv[1:])' \
  results "$PAIR" "$BASE" "$RL" "${SLURM_JOB_ID:?Slurm job ID required}"

</file>

<file name="slurm/01_vllm_sample.sbatch">
#!/bin/bash -l
#SBATCH --partition=h200q
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=16
#SBATCH --mem=170G
#SBATCH --output=/work/project/icml27-topic-search/killtests/rl-sparse-distill/slurm/logs/%x-%j.out
#SBATCH --job-name=rsd-vllm
#SBATCH --time=06:00:00
# 用法：sbatch slurm/01_vllm_sample.sbatch PAIR COND SHARD NSHARD
#   COND = base | rl   （n=128、test；閘門 0 與判準 1 共用同一份）
#   COND = base_greedy | rl_greedy （n=1、temperature 0、split all；閘門 0 重現用）
export PAIR=$1; COND=$2; SH=${3:-0}; NS=${4:-1}
set -e
source /work/project/icml27-topic-search/killtests/rl-sparse-distill/slurm/env.sh
export RSD_RUN_ID=${RSD_RUN_ID:-${SLURM_ARRAY_JOB_ID:-$SLURM_JOB_ID}}
case $COND in base*) M=$BASE;; *) M=$RL;; esac
OUT=results/$PAIR/$COND
case $COND in
  *_greedy) $PY -I scripts/gen_vllm.py --model $M --out $OUT --split all --n 1 --temperature 0 --top-p 1 --shard $SH/$NS ;;
  *)        $PY -I scripts/gen_vllm.py --model $M --out $OUT --split test --n 128 --shard $SH/$NS ;;
esac
# 分片完成憑證及單一判分者由 grade.py 驗證。
$PY -I scripts/grade.py "$OUT" --pair "$PAIR" --condition "$COND" --n-shards "$NS" --workers 16

</file>

<file name="tests/test_artifacts.py">
"""Synthetic integrity regressions. These tests do not read real scored artifacts."""

from pathlib import Path
import sys
import unittest
from unittest.mock import patch

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))
import aggregate as agg
import artifacts


class ArtifactTests(unittest.TestCase):
    def test_exact_counts_and_finish_semantics(self):
        valid = {"a": {"correct": [1, 0], "finished": [1, 0], "ntok": [4, 10]}}
        artifacts.validate_records(valid, ["a"], 2, 10, False)
        invalid = (
            {"correct": [1, 0, 0]}, {"finished": [1]}, {"correct": [1, 1]},
            {"ntok": [4, -1]}, {"n_repl": [0, 11]}, {"finished": [True, 0]},
        )
        for fragment in invalid:
            with self.subTest(fragment=fragment), self.assertRaises(artifacts.ArtifactError):
                artifacts.validate_records({"a": dict(valid["a"], **fragment)}, ["a"], 2, 10, False)
        with self.assertRaises(artifacts.ArtifactError):
            artifacts.validate_records(valid, ["a", "b"], 2, 10, False)

    def test_gate0_does_not_read_later_conditions(self):
        crit, pc = agg.cfg("criteria.json"), agg.cfg("pairs.json")
        crit["eval"]["test_questions_min"] = 1
        rows = [{"id": "a", "split": "test"}]
        untouched = Path("unused-synthetic-results")
        with patch.object(agg, "check_gate_minus1", return_value={}), patch.object(agg, "load_verified", return_value=None) as load:
            report = agg.evaluate(untouched, ["B"], "0", crit, pc, rows)
            self.assertEqual(set(report), {"phase", "partial", "pairs"})
            self.assertEqual(set(report["pairs"]), {"B"})
            self.assertEqual([call.args[2] for call in load.call_args_list], ["base", "rl", "base_greedy", "rl_greedy"])

    def test_gate0_requires_correctness(self):
        crit, pc = agg.cfg("criteria.json"), agg.cfg("pairs.json")
        crit["eval"]["test_questions_min"] = 1
        rows = [{"id": "a", "split": "test"}]
        with patch.object(agg, "check_gate_minus1", side_effect=artifacts.ArtifactError("missing")), patch.object(agg, "load_verified") as load:
            report = agg.evaluate(Path("unused-synthetic-results"), ["A"], "0", crit, pc, rows)
            self.assertEqual(report["pairs"]["A"]["gate_minus1"]["verdict"], "未能測試")
            load.assert_not_called()

    def test_primary_pair_cannot_skip_a(self):
        crit = agg.cfg("criteria.json")
        good = {"gate0": {"verdict": "配對合格"}}
        self.assertEqual(agg.phenomenon_eval({"A": good, "B": good}, crit)["primary_pair"], "A")
        self.assertIsNone(agg.phenomenon_eval({"B": good}, crit)["primary_pair"])
        ruled_out = {"gate0": {"verdict": "配對不合格", "reproduce_ok": True, "trunc_ok": True}}
        self.assertEqual(agg.phenomenon_eval({"A": ruled_out, "B": good}, crit)["primary_pair"], "B")


if __name__ == "__main__":
    unittest.main()

</file>