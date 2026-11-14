# rl-sparse-distill：停損實驗（第 1 週第一關）

## 目的與對應回報
對應 `reports/2026-11-14-44-deepcheck-rl-sparse-decisions.md`（RL 稀疏決策的混合目標 on-policy 蒸餾），以及 Fable 重審 `reports/2026-11-14-56-killtest-criteria-review.md` 第 5 節。
只回答一件事：**rl-replace 這個 λ=1 端點值不值得蒸餾？** 若在公開 1.5B 配對上，rl-replace 在 pass@64 上沒有顯著勝過 RL、或在相同 pass@1 下勝不過 WiSE-FT 權重內插前緣，停 λ=1 端點路線，不做蒸餾（這不等於題目淘汰；(i) 成立而 (ii) FAIL 時還有 R3 的 λ=0.5 探針）。注意 rl_replace 在不一致位置確定性輸出 RL 的 argmax，是 q*_λ 家族裡多樣性最低的成員，它只近似 pass@1 的上限，**不是 pass@64 的上限**。本資料夾**不含任何蒸餾程式**。

## 判準（凍結版見 PREREG.md；數字唯一來源 configs/criteria.json）
回報 44 原文：「第 1 週，D5 前：在 Oat-Zero-1.5B 與 SimpleRL-Zoo-1.5B（或 ORZ-1.5B）兩個配對上：rl-replace 必須用 ≤10% 的位置拿回 ≥70% 的 pass@1 增益；在 ≥500 題的配對 bootstrap 下，pass@64 必須比 RL 高，且 CI 不含 0。兩個配對都做不到就整題停。」
Fable 修正版（現行）：閘門 0（重現 RL 已報 pass@1 ±2pp、token 上限凍結且截斷率差 <2pp、配對資格：RL pass@1 增益 ≥5pp、RL pass@64 低於 base ≥2pp、落在 30–85%）；判準 1（≥1,000 題、每題 128 樣本、k=64、rl-replace 比 RL 高 ≥2pp 且 97.5% CI 不含 0、相同 pass@1 下高於 WiSE-FT 6 個 λ 點的前緣）。原案的「≤10% 位置、拿回 ≥70% 增益」改為只報告（`recovery_p1`、`replace_rate_token_weighted`），不再是判準。

## 模型配對（HF API 實查，皆可公開下載、未 gated）
| 配對 | base | RL | 大小 | 已報 greedy（MATH500）|
|---|---|---|---|---|
| A | Qwen/Qwen2.5-Math-1.5B | sail/Qwen2.5-Math-1.5B-Oat-Zero | 3.1 + 3.1 GB | base 33.0／RL 74.2（Oat-Zero Table 4，Qwen-Math 模板）|
| B | Qwen/Qwen2.5-1.5B | hkust-nlp/Qwen-2.5-1.5B-SimpleRL-Zoo | 3.1 + 7.1 GB（RL 為 fp32）| base 29.6／RL 59.0（SimpleRL-Zoo 論文表）|
- 備選（已確認存在，**未下載**）：Open-Reasoner-Zero/Open-Reasoner-Zero-1.5B（3.1 GB，base 同 B）；R1-Distill-Qwen-1.5B 對 DeepScaleR-1.5B-Preview（7.1 GB）或 Nemotron-Research-Reasoning-Qwen-1.5B（7.1 GB）——後者回報 44 說要 32k 上限才有 pass@k 交叉，成本高 10 倍以上，且會動到凍結的 token 上限。
- 查不到的：回報 44 提到的 `hkust-nlp/Qwen-2.5-Math-1.5B-SimpleRL-Zoo` 在 HF 回 401（不存在），故 B 用 Qwen2.5-1.5B。
- **模型權重已下載完成**（2026-11-14，4 個模型，CPU 檢查通過；`HF_HOME=/work/ChihChungHsu/cactus8603/icml/hf_home`）。B 配對 RL（SimpleRL-Zoo）為 fp32 且多存一份 `lm_head.weight`（339 對 338 個 tensor），但 config 為 `tie_word_embeddings: true`，且實查該 `lm_head` 與 `embed_tokens` 逐元素相同（max|diff|=0），故以 tied 方式處理在數學上等價；`make_wise.py` 已加斷言（不相同就停）。所有生成載入點都明確指定 bf16（vLLM `dtype=pairs.json:dtype`、HF `--dtype` 預設同值），正確性閘門 00 用 fp32 兩側一致。

## 檔案
- `configs/pairs.json`（模型、模板、解碼、已報數字）、`configs/criteria.json`（全部門檻）
- `scripts/prepare_data.py`（評測集）、`gen_vllm.py`（base／rl／WiSE，vLLM）、`gen_replace.py`（HF 雙模型 lock-step rl-replace，取樣函式直接用原作 wrb）、`make_wise.py`、`grade.py`、`aggregate.py`、`common.py`
- `tests/smoke_cpu.py`、`tests/make_noisy_copy.py`
- `slurm/06_geo_mix`（R3 第二探針，只寫不送）
- `slurm/`：`00_correctness_gpu`、`01_vllm_sample`、`02_rl_replace`、`03_wise`、`04_throughput_probe`（皆 partition h200q、`--gres=gpu:1`、16 CPU、170G）
- 外部：`/work/ChihChungHsu/cactus8603/icml/ext/WhatRLBrings`（原作程式，已 clone 並讀過 rl_replace.py、sampling.py、grading.py、metrics.py）；`pylib/`（math-verify 0.9.0 等純 python 套件，pip --target 安裝在本資料夾，**沒有動任何共用 env、沒有裝 torch**）。
- 執行環境：既有 `envs/vllm033`（vLLM 0.33.0、torch 2.14、transformers 5.20.2）。`vllm0251rkv` 沒用到。

## 方法重點
- rl-replace 規則（原作逐字）：b_t=argmax p_B、r_t=argmax p_R；b_t≠r_t 輸出 r_t，否則從 p_B 取樣（T=0.6，top-p 0.95）。
- 為什麼用 HF 而不用 vLLM 做 rl-replace：vLLM 無法在每個 token 同步兩個模型。HF 版做了多題合批（左 padding＋position_ids）與已結束列壓縮，效能工程與原作不同處已在檔頭註明，並由 smoke 閘門 T3／T4 驗證等價。
- base／rl／WiSE 用 vLLM（快）。**注意：rl_replace 用 HF、其餘用 vLLM，兩個引擎的數值有微小差異**；為排除引擎偏差，閘門 −1 另含 HF 版 base／rl 與 rl_replace 的等價測試，若審查者要求更嚴，**已定案（2026-11-14）：判準 (i) 的 RL 比較對象改用 HF 引擎的 `gen_replace.py --cond rl`（`slurm/05_rl_hf.sbatch`），每配對預估多 約10 GPU 小時**（合計預估每配對 40–50 h、兩配對約 80–100 h，仍在 120 h 上限內）。
- 資料：1,453 題，test 1,053／dev 400（dev 本階段不使用）。

## 執行順序與送出指令（**僅列出，未執行**；都在本資料夾下送出）
```bash
cd /work/ChihChungHsu/cactus8603/icml27-topic-search/killtests/rl-sparse-distill
# 0. 資料已產生：data/eval.jsonl（md5 見 logs/RUNLOG.md）；pylib 已裝
# 1. 閘門 -1：真實模型正確性 + 吞吐實測（各 <1 GPU 小時）
sbatch slurm/00_correctness_gpu.sbatch A ; sbatch slurm/00_correctness_gpu.sbatch B
sbatch slurm/04_throughput_probe.sbatch A
# 2. 閘門 0：單模型取樣（每配對 base、rl 各 1 job；greedy 很快）
for P in A B; do for C in base rl base_greedy rl_greedy; do sbatch slurm/01_vllm_sample.sbatch $P $C 0 1; done; done
python -I scripts/aggregate.py --gate 0        # 看每個配對是否「配對合格」
# 3. 判準 1：只對合格配對。rl_replace 切 8 片（每片約 2–3 小時），WiSE 6 點
P=A   # 或 B
for S in 0 1 2 3 4 5 6 7; do sbatch slurm/02_rl_replace.sbatch $P $S 8; done
for L in 0.15 0.3 0.45 0.6 0.75 0.9; do sbatch slurm/03_wise.sbatch $P $L; done
for S in 0 1 2 3 4 5 6 7; do sbatch slurm/05_rl_hf.sbatch $P $S 8; done   # (i) 的比較對象 rl_hf（HF 引擎 RL，約 +10 GPU 小時）
python -I scripts/aggregate.py                  # 機械輸出 PASS／灰區／FAIL／未能測試與全部數字 -> results/aggregate.json
```
注意：`02_rl_replace` 的 8 片各自 `raw_{i}of8.jsonl.gz`，最後完成的 job 會自動判分合併；若兩片同時結束可能都沒觸發，補跑 `python -I scripts/grade.py results/$P/rl_replace`。

## 預估 GPU 小時與空間（全為推估，未實測；送出前先跑 04 吞吐實測）
取樣量：1,053 題 × 128 樣本 = 134,784 條／cond／配對；平均長度假設 700–1,200 token。
| 項目 | 每配對 | 兩配對 |
|---|---|---|
| 閘門 0：base、RL 各 128 樣本（vLLM，約 30–40k tok/s）＋ 4 個 greedy | 3–4 h | 6–8 h |
| 判準 1：rl_replace（HF lock-step，256 列／批、有壓縮；約 80 ms/步、等效 1,500 全批步、527 批）| 18–25 h | 36–50 h |
| 判準 1：WiSE 6 點（vLLM）| 8 h | 16 h |
| 正確性閘門、吞吐實測 | 1–2 h | 2–4 h |
| 合計（兩配對全跑）| 約 30–40 h | **約 60–80 h**（上限 120）|
回報 44 說前兩週共 150–250 GPU 小時；本階段若 A 配對就不合格或 FAIL，用不到 B 的判準 1。「3 種解碼 × 2 配對 × 64 次取樣」改為 128 樣本（k=64 需要 n=128 才穩）並加 WiSE 6 點，故解碼條件實為 9 種／配對。
空間：模型快取 16.4 GB（4 個模型）；每 cond raw（gzip 含完整文字）約 0.1–0.3 GB，9 cond × 2 配對 ≤ 5 GB；graded.json 小；WiSE 內插模型 3.1 GB 暫存節點、用完刪除；noisy 假 RL 1.5 GB（smoke 用，之後刪）。總計 ≤ 25 GB（上限 60）。

## Smoke test 結果（CPU，無 GPU）
見下節「Smoke 紀錄」。

## 配對 B 評測模板（2026-11-14 查證）
- B 用 `pairs.json: pairs.B.template = simplerl_abel_chat`：user 訊息 `Question:\n{q}\nAnswer:\nLet's think step by step.\n`，外包 Qwen 聊天模板。依據：論文附錄 B.3／B.5、`hkust-nlp/SimpleRL-Zoo-Data` 的 simplelr_abel_* 實際 prompt、tokenizer 實測。HF model card 為空，GitHub 的 hkust-nlp/simpleRL-reason（v1 分支）eval 程式有 `abel` 模板但沒寫哪個模型用哪個模板。
- **未確認**：(1) 評測是否外包聊天模板（推定外包）；(2) 原作評測用 16000 token、Zoo 模型 temperature 1.0，我們用 3000 token、greedy。**B 的 00／01 送出前必須先由協調者確認**；閘門 0 的 MATH500／Minerva 重現是最終檢驗，不過就記「未能測試」。
- 實作：`common.build_prompt` 依環境變數 `PAIR`（sbatch 皆已 export）選 per-pair 模板；`gen_vllm.py`、`gen_replace.py` 未改動。

## 風險與未確認事項
1. **HF lock-step 吞吐是最大成本風險**：步時 80 ms、DynamicCache 每步拼接都是估計；若實測慢 2 倍，rl_replace 單配對 > 40 GPU 小時。備案：用「vLLM 取樣＋RL 預填找第一個不一致位置、截斷重生」的分輪方法（未實作、未驗證）。
2. rl_replace（HF）對 base／RL（vLLM）的引擎差異（見方法重點）。
3. CI：已改為單側 α=0.0125（Bonferroni，兩配對擇一）；「落在 30–85%」解讀為 base 與 RL 的 pass@64（見 PREREG）。
4. 已報數字取自論文圖表（Oat-Zero 為人工讀圖），且為 greedy、各自的題目版本；OlympiadBench 我們用 math-ai 版純文字單一答案 581 題（原作 674 題含多答案），Minerva 為 math-ai 272 題；重現 ±2pp 可能因題目版本不同而不過（這會記為「未能測試」，需查題目版本與判分）。SimpleRL-Zoo 原作評測的 token 上限可能不是 3000。
5. Qwen2.5-Math-1.5B 的 context 只有 4096，故 3000 token 上限；SimpleRL-Zoo 在 3000 token 下的表現可能略低於其原報告（原報告用較長上限，未確認）。
6. SimpleRL-Zoo 的 RL 檔是 fp32，bf16 載入有微小差異；WiSE 內插以 fp32 計算後轉 bf16。tie_word_embeddings 的 lm_head 缺失情形在 make_wise.py 已特判，未在真實 Zoo 檔上測過。
7. 灰區的「加碼 dev 400 題後合併重算」程式尚未寫。
8. 單題批次內只要有一條樣本跑到 3000 步，該批就要跑滿；壓縮後成本主要在前段，但長尾仍存在。
9. 未實測：vLLM 腳本（`gen_vllm.py`）與 slurm 腳本都只在審查前語法檢查，尚未在 GPU 上跑過。

## Smoke 紀錄（CPU，2026-11-14，Qwen3-0.6B 與其加雜訊副本當「假 RL」，副本已刪除；log：logs/smoke_cpu.log）
T1（rl=base 等於 base 取樣）、T2（全部替換等於 RL greedy）、T3（與原作 RLReplaceDecoder 逐字且替換數相同）、T4（左 padding＋壓縮 greedy 等於單題 greedy）、T5（替換比例 0.412，僅證明有替換）全部 PASS。`aggregate.py --selftest` 通過（pass@k 估計式對暴力組合、虛無情境判 FAIL）。
未跑：vLLM 腳本、WiSE 內插、真實 1.5B 模型（模型尚未下載、不佔 GPU）。

## 2026-11-14 代理落差稽核後補充（結果前定案；條文見 PREREG.md 修改紀錄 R1–R3）
- R1：(ii) 的 FAIL 須同時「點估計 ≤ 0」且「差值 98.75 百分位 < +2.0pp」，否則歸灰區；合併 dev 後仍排除不了 2pp → 「未能測試（檢定力不足）」。`aggregate.py` 輸出 `ii_fail`、`ii_underpowered`、`wise_margin_upper_pp`。
- R2：`gate1.mix_curve_descriptive`（回應層級混合曲線，只描述）。
- R3：`gen_replace.py --cond geo_mix --lam 0.5`；`aggregate.py` 輸出 `probe2`（triggered／pair／result）；送法：`for S in 0..7; do sbatch slurm/06_geo_mix.sbatch <PAIR> $S 8; done`（只在 probe2.triggered 且使用者放行後；預估與 rl_replace 同量級 18–25 GPU 小時／配對，探針上限 25，超出 120 總量先回報）。
- 既有 cond 不變的證據：以 `git HEAD` 的舊版 `gen_replace.py` 與新版比對 base／rl（含 temperature 0）／rl_replace（壓縮與不壓縮）／all_replace，輸出（文字、結束旗標、token 數、替換數）全部逐字相同（Qwen3-0.6B 與加雜訊副本，CPU）。
- smoke 新增 T6（兩模型相同時 geo_mix == base 取樣、替換數 0）、T7a（λ=1、temperature 0 == RL greedy）、T7b（λ=0 == base 取樣）、T8（λ=1 時與獨立參考迴圈逐字相同且不一致位置數相同）。GPU 版 `00_correctness_gpu.sbatch` 會一併跑。
- 刪除的說法：「rl_replace 是學生上限」只對 pass@1 近似成立，對 pass@64 不成立。

## 2026-11-14 [claude] 判定方式改為現象／訊號／空間（報告 60 通用條文 G 與第 5 節；條文見 PREREG.md 文末）
- `aggregate.py` 輸出 `phenomenon`、`signal`、`space`、`r3`、`overall_three_items`；`overall` 欄位改為三項合成（G1）。原 `gate1` 的 verdict、對 WiSE 的比較、R1 FAIL 條件只做描述。
- 現象＝閘門 0(c)（A、B 任一合格）；訊號＝量 1（pass@64 − rl_hf ≥ 2pp）＋量 2（pass@1 回收率 ≥ 0.5）；空間＝H_a 或 H_b（m=2，單側 α=0.00625，≥ 2pp）；R3 為空間的第三個上限候選（訊號 PASS 且空間 FAIL 時觸發，預算 25 GPU 小時）。
- 驗證：`python -I scripts/aggregate.py --selftest`（用 `envs/vllm033` 的 python）涵蓋現象、訊號、空間各自的 PASS／FAIL／灰區、H_b 單獨通過、合併後未能測試、R3 觸發與通過／未通過，以及整體規則。
