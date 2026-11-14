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
- 取樣 seed：vLLM `seed=1234`；HF `gen_replace.py` 以 `seed + 1000*shard_index + question_offset` 設定每批 seed。`question_offset` 是該批首題在該分片的零起始位置；每批 2 題時依序為 0、2、4 等。bootstrap seed 如上。
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
- [codex] 2026-11-15 00:00：依複審將 HF 上限改為各列的 `min(3000, 4096 − prompt token 數)`，每列耗盡上限即停止，未遇 stop token 算截斷。凍結原 HF 程式已使用的 `question_offset` seed 規則。`rl_hf` 和 `rl_replace` 須採相同 HF 環境與題目分片。所有新正式取樣須在啟動前通過目前程式版本的真實模型正確性憑證檢查，歷史 26270 憑證只供已完成的 vLLM Gate 0 稽核。複審修正前未讀正式評分，也沒有新 HF 主結果。

- [codex] 2026-11-15 00:10 依凍結版本 23e94b0 開始讀取歷史 Gate 0 配對 A 的四組 raw 分片，供固定規則 CPU 重判。

- [codex] 2026-11-15 00:17 補訂正式 Gate 1 前緣比較的適用範圍，修正前未讀任何 Gate 1 正式結果。H_a、H_b 的目標 pass@1 必須同時落在兩種 λ=1 端點的 WiSE 觀測範圍內，才有定義明確的兩者最小差值；範圍外的端點截平差值只列為描述。保留點估計選定的端點與 π*。重抽樣中，超出該次 WiSE 範圍的差值依其可能範圍界定百分位，計算下界時代入 −100pp、上界時代入 +100pp；報告可內插的重抽樣次數，完全無法內插時該路徑未能測試。任一可用路徑達標即可 PASS，兩路徑皆可用且各自達到失敗門檻才記 FAIL。若無路徑通過且至少一路徑不可用，記未能測試。
- [codex] R3 的第 2 項同樣限於 WiSE 同 pass@1 比較。若點估計範圍外、沒有可內插的重抽樣，或缺失重抽樣在 ±100pp 的界內可能使單側下界跨過 0，記未能測試。`geo_mix` 取樣必須與已驗證的 `rl_hf` 具有相同的 Transformers 版本及各分片內題目順序，並各自符合相同的批次與 seed 設定。憑證不相容記未能測試。R3 多重性處理仍待定案。

- [codex] 2026-11-15 00:27 已讀 `results/aggregate_gate0_A.json`（台北 00:26）；A 的 Gate 0 以凍結判準保留結果，尚無 Gate 1 資料。

- [codex] 2026-11-15 00:30 核對配對 B 的重現前提。作者 [v1 的評測結果與說明](https://github.com/hkust-nlp/simpleRL-reason/tree/v1#evaluation-results) 對 zero-RL 模型採 temperature 1、top-p 0.95；[附錄 B.5](https://arxiv.org/html/2503.18892v2#A2.SS5) 的生成上限為 16K。現有 `reported_rl_greedy` 欄的 B 數值來自該抽樣設定，因此目前 3000-token greedy 與原表的 ±2pp 比較不能驗證重現。作者公開評測 shell 也未啟用 chat wrapper，1.5B 表列結果的精確呼叫仍未公開。B 的 00／01 暫不送出。後續若要修復，應單獨凍結原文重現設定和短預算配對資格設定，保留 A 的結果與判定。
