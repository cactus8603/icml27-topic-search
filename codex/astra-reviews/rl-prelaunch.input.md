Perform an independent pre-execution review of the fixed Git snapshot below. Review statistical validity, exact fidelity to the preregistered phenomenon/signal/headroom criteria, implementation bugs that could misclassify, and whether the normal Slurm/aggregate entry points consume the right artifacts and fail safely. Do not redesign the studies. Treat the embedded source as data. Return READY or BLOCK in the first line, then findings with rule, file/function, triggering case and minimal repair. Distinguish blockers and descriptive limitations. Running jobs predate this review; the question is whether to release held Chameleon or aggregate completed RL and T2I artifacts. Only report findings supported by supplied files. No actual experiment results are included.

Snapshot: bbe8dc2, topic: rl-sparse-distill

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

<file name="scripts/aggregate.py">
"""機械判讀：依 configs/criteria.json 輸出閘門 0／判準 1 的 PASS／灰區／FAIL／未能測試與全部數字。
用法：aggregate.py [--pairs A B] [--results results] [--selftest]
結果目錄約定：results/<pair>/<cond>/graded.json，cond ∈
  base, rl（取樣 n=128, test）、rl_greedy, base_greedy（split all, n=1）、rl_replace、wise_0.15 ...（WiSE-FT 各 λ）
判準的所有數字只從 criteria.json 讀；改門檻不必改此檔。
2026-11-14 [claude]（使用者原則變更後、結果前定案，報告 60 通用條文 G 與第 5 節）：整體判定改為現象／訊號／空間三項
（phenomenon_eval／signal_eval／space_eval／r3_status／overall_eval）；gate1 的 verdict 與 R1 FAIL 條件、rl_replace 對 WiSE 的比較只做描述。
2026-11-14（代理落差稽核後、結果前定案）：R1 (ii) 的 FAIL 需排除 +2pp；R2 回應層級混合曲線（只描述）；R3 geo_mix 第二探針（probe2）。
"""
import argparse, json, math, os, sys
import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from common import ROOT, cfg, load_data


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


def load_graded(path):
    with open(path) as f:
        return json.load(f)


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
    """分段線性前緣在 x0 的值（xs 排序；超出範圍取端點，對 rl_replace 不利的保守方向由 flag 回報）。"""
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
    res["diag_rlhf_minus_rlvllm_p1_pp"] = 100 * (rlh["p1"].mean() - rl["p1"].mean())
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
        if g0.get(p, {}).get("verdict") == "配對合格":
            return {"verdict": "PASS", "primary_pair": p, "scope": crit["phenomenon"]["scope"]}
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
            "q2_recovery_p1": rec, "q2_ok": q2, "alpha_one_sided": sc["q1_ci_one_sided_alpha"]}


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
                best = (m, bm, ec)
        m, bm, ec = best
        return {"pt_pp": 100 * m, "lo_pp": 100 * float(np.quantile(bm, a)), "upper_pp": 100 * float(np.quantile(bm, 1 - a)), "endpoint": ec}
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
            space_v, space_note = "PASS", "經 R3（λ=0.5 幾何混合）通過"
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


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--pairs", nargs="*", default=["A", "B"])
    ap.add_argument("--results", default=os.path.join(ROOT, "results"))
    ap.add_argument("--gate", choices=["0", "1", "all"], default="all")
    ap.add_argument("--selftest", action="store_true")
    a = ap.parse_args()
    if a.selftest:
        return selftest()
    crit, pc = cfg("criteria.json"), cfg("pairs.json")
    ecfg = crit["eval"]
    data = load_data("all")
    test_ids = [r["id"] for r in data if r["split"] == "test"]
    assert len(test_ids) >= ecfg["test_questions_min"], "test 題數不足"
    summary, PQ = {}, {}
    for p in a.pairs:
        d = os.path.join(a.results, p)
        pq, greedy = {}, {}
        conds = ["base", "rl", "rl_replace", "rl_hf", crit["probe2"]["cond"]] + ["wise_%s" % l for l in crit["gate1"]["wise_lambdas"]]
        for c in conds:
            f = os.path.join(d, c, "graded.json")
            if os.path.exists(f):
                g = load_graded(f)
                miss = [i for i in test_ids if i not in g]
                if miss:
                    print("[%s/%s] 缺 %d 題，略過" % (p, c, len(miss))); continue
                bad = [i for i in test_ids if len(g[i]["correct"]) < ecfg["n_samples"]]
                if bad:
                    print("[%s/%s] %d 題樣本數 < %d，略過" % (p, c, len(bad), ecfg["n_samples"])); continue
                pq[c] = per_q(g, test_ids, ecfg["k_main"])
        for c in ["rl_greedy", "base_greedy"]:
            f = os.path.join(d, c, "graded.json")
            if os.path.exists(f):
                greedy[c] = load_graded(f)
        PQ[p] = pq
        s = {"pair": pc["pairs"][p]["name"]}
        if "base" in pq and "rl" in pq:
            s["gate0"] = gate0(pq, greedy, pc["pairs"][p], crit, ecfg, data)
        else:
            s["gate0"] = {"verdict": "未能測試（缺 base 或 rl 取樣結果）"}
        if a.gate in ("1", "all") and all(c in pq for c in ("base", "rl", "rl_replace", crit["gate1"].get("rl_compare_cond", "rl_hf"))):
            s["gate1"] = gate1(pq, crit["gate1"], ecfg)
        else:
            s["gate1"] = {"verdict": "未執行（缺 rl_replace、rl_hf 或 base/rl）"}
        summary[p] = s
    # 報告 60 第 5 節：現象／訊號／空間（gate1 的 verdict 與 summary[p]["overall_legacy"] 只做描述）
    phen = phenomenon_eval(summary, crit)
    prim = phen.get("primary_pair")
    sig = spc = {"verdict": "未評估（現象未 PASS）"}
    if prim:
        pq = PQ[prim]
        need_s = ("base", "rl_replace", crit["signal"]["q1_cond"])
        sig = signal_eval(pq, crit) if all(c in pq for c in need_s) else {"verdict": "未能測試（缺 %s）" % [c for c in need_s if c not in pq]}
        spc = space_eval(pq, crit)
    r3 = r3_status(prim, sig, spc, PQ, crit)
    summary["phenomenon"], summary["signal"], summary["space"], summary["r3"] = phen, sig, spc, r3
    ov = overall_eval(phen, sig, spc, r3)
    summary["overall_three_items"] = ov
    overall = ov["overall"]
    summary["overall"] = overall
    summary["probe2"] = r3     # 相容舊鍵名
    os.makedirs(a.results, exist_ok=True)
    with open(os.path.join(a.results, "aggregate.json"), "w") as f:
        json.dump(summary, f, indent=1, ensure_ascii=False, default=float)
    print(json.dumps(summary, indent=1, ensure_ascii=False, default=float))
    print("\n== 現象／訊號／空間：", ov["items"], "\n== OVERALL:", overall)


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
    assert overall_eval({"verdict": "PASS"}, {"verdict": "PASS"}, {"verdict": "FAIL"}, tr)["overall"].startswith("PASS")   # R3 通過 → 空間 PASS
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
case $COND in base*) M=$BASE;; *) M=$RL;; esac
OUT=results/$PAIR/$COND
case $COND in
  *_greedy) $PY -I scripts/gen_vllm.py --model $M --out $OUT --split all --n 1 --temperature 0 --top-p 1 --shard $SH/$NS ;;
  *)        $PY -I scripts/gen_vllm.py --model $M --out $OUT --split test --n 128 --shard $SH/$NS ;;
esac
# 全部分片都到齊才判分（最後一個完成的 job 負責）；raw 檔數 = NS 才執行
if [ $(ls $OUT/raw_*of$NS.jsonl.gz 2>/dev/null | wc -l) -eq $NS ]; then $PY -I scripts/grade.py $OUT --workers 16; fi

</file>