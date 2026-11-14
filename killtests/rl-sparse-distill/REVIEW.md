# REVIEW：rl-sparse-distill（閘門 −1、閘門 0、判準 1）

審查者：Opus｜日期：2026-11-14｜範圍：PREREG.md、configs/criteria.json、configs/pairs.json、scripts/*、tests/smoke_cpu.py、slurm/*、README｜依據：reports 44、56 第 5 節與通用規則；也納入協調者補充的 B 配對資訊。
只做了閱讀與 CPU 上的極小檢查（`aggregate.py --selftest` 通過、SimpleRL-Zoo 的 lm_head 比對），沒有送出任何 job，也沒有 commit。

## 結論：**修改後可送**
- 閘門 −1、閘門 0（配對 A）在 commit 後就能送。
- 判準 1（rl_replace、WiSE）送出前，要先處理必修 2–4。
- 配對 B 的閘門 0 送出前，要先處理必修 5。

## 這題怎麼判（給使用者）
1. **閘門 −1**：T1–T5 在真實模型、fp32 上都通過。不過就是程式問題，給 3 個工作天。
2. **閘門 0（每個配對）**
   - (a) RL greedy 重現已報 pass@1，±2pp；
   - (b) base 與 RL 的截斷率差 < 2pp。
   - (a)(b) 不過記「未能測試」。
   - (c) 配對資格：RL 的 pass@1 增益 ≥ 5pp、RL 的 pass@64 比 base 低 ≥ 2pp、兩者的 pass@64 都在 30–85%。(c) 不過記「配對不合格，換配對」，**不算題目死**。
3. **判準 1（只做合格配對）**，主指標是 test 1,053 題、每題 128 樣本的 pass@64：
   - (i) rl_replace − RL ≥ 2pp，且下界 > 0；
   - (ii) 在相同 pass@1 下，高於 WiSE 8 點的分段線性前緣，且下界 > 0。
   - 兩項都成立才 PASS。rl_replace − RL < 1pp，或前緣差的點估計 ≤ 0，判 FAIL：停，不做蒸餾。其餘是灰區，唯一加碼是加跑 dev 400 題。
4. 判讀全部由 `aggregate.py` 機械輸出，門檻只從 `criteria.json` 讀。我核對過，程式數字與 PREREG 文字一致。

## 對照 56 號修正的落實情況
| 項目 | 狀態 |
|---|---|
| WiSE-FT 比較排在蒸餾之前，上限贏不了前緣就停 | 已落實，本資料夾不含蒸餾 |
| 閘門 0 三項，不合格換配對、不算死 | 已落實（aggregate 的 gate0 verdict 分開記） |
| ≥ 1,000 題、128 樣本、無偏 pass@k、k=64 | 已落實（1,053 題；估計式已用暴力組合驗證） |
| ≥ 2pp + 97.5% CI | 落實了，但**詮釋有問題**，見必修 2 |
| dev／test 劃分 | 已落實（seed 20261114，400／1,053）。greedy 重現用 split=all，是為了對齊論文的完整題集，可以接受 |
| 成本與重試上限 | 已落實：120 GPU 小時、基礎設施失敗重送 ≤ 2 次、不准換 seed 挑結果 |

## 實作者自己的詮釋
1. **「97.5% CI」解讀成單側 2.5 百分位**：這等於雙側 95%，**沒有做多重比較校正**。56 號在本題列的問題之一是「二擇一沒有校正」，而現行整體規則是「至少一個合格配對 PASS → PASS」，等於 2 個配對擇一。所以 97.5% 應理解為 Bonferroni 後的雙側 97.5%，也就是單側 1.25%。現行版本**稍微偏向 PASS**，要修，見必修 2。
2. **30–85% 套用到 base 與 RL 兩者**：比只套 RL 嚴，但失敗結果是「換配對」，不會把結論推向 PASS 或 FAIL。**同意**。
3. **HF（rl_replace）與 vLLM（base、RL、WiSE）引擎混用**：判準 (i)(ii) 是用不同引擎的結果直接相減。兩個引擎的數值差和取樣實作差的方向無法預知，大小可能接近 1pp 等級，而灰區與 FAIL 的界線就是 1pp。smoke 的 T1–T5 全是 HF 對 HF，**沒有任何 HF 對 vLLM 的檢查**。要修，見必修 3。
4. **token 上限 3000**：Qwen2.5-Math 的 context 只有 4096，這個上限與 Oat-Zero 的評測一致，**同意凍結**。要提醒的是，44 號指出 token 上限本身就能製造或反轉 pass@k 交叉，所以結論只在 3000 這個上限下成立，報告時要寫明。base 模型可能較常撞到上限，讓閘門 0(b) 失敗，那會記為「未能測試」。另外，閘門 0(b) 只比 base 與 RL；建議判準 1 也輸出 rl_replace 的截斷率（目前 per_q 有算，但沒進報表）。
5. **重現閘門 (a) 要求 MATH500、Minerva、Olympiad 三個來源都在 ±2pp 內**：我們的 OlympiadBench 是 581 題的純文字子集，原作是 674 題，**題集不同**，跟已報數字比較沒有效力，很可能因此平白記成「未能測試」，耗掉 3 天。要修，見必修 4。

## 必修
1. **（送任何 job 前）commit**：repo 目前沒有任何 commit。要 commit 的包括 PREREG、criteria.json、pairs.json、scripts、tests、slurm、data/eval.jsonl。
2. **（判準 1 判讀前，但要在 commit 時一起定）多重比較**，二選一，寫進 PREREG：
   - (a) `criteria.json: gate1.ci_one_sided_alpha` 改為 **0.0125**（我建議這個，保留兩配對都跑的選項）；或
   - (b) 在 PREREG 寫明「判準 1 只對第一個合格的配對（A 優先）執行，B 只在 A 不合格時才跑」，此時 m=1，維持 0.025。
   - README 第 61 行「若 A FAIL 就用不到 B」的說法和 (b) 一致，但 PREREG 與 aggregate 的整體規則目前是兩配對擇一。這兩邊必須統一。
3. **（判準 1 送出前）消除引擎差**：在 PREREG 加一條，判準 (i) 的比較對象改為 **HF 版 RL 取樣**：`gen_replace.py --cond rl`，同 test、n=128、同 seed 規則，輸出到 `results/<P>/rl_hf`，aggregate 的 (i) 讀這份。vLLM 版 RL 只用於閘門 0，並另外輸出 rl_hf − rl_vllm 的 pass@1 與 pass@64 當診斷。(ii) 的前緣仍用 vLLM，但要報一個敏感度分析：把 λ=1 的端點換成 rl_hf。成本約為 rl_replace 的一半（每配對約 9–12 GPU 小時），仍在 120 上限內。如果不想加成本，至少要做一個事先登記的子集檢查：HF 對 vLLM 的 RL，test 前 1/4 題 × 128 樣本，pass@64 差的 95% CI 要含 0 且點估計 |·| < 0.5pp，否則改用上述完整方案。
4. **（閘門 0 判讀前）重現來源**：`criteria.json: gate0.reproduce_sources` 改為 `["math500", "minerva"]`，olympiad 改成只記錄。理由見詮釋 5：題集不同，比較沒有效力。
5. **（配對 B 送出前）確認 SimpleRL-Zoo 1.5B 的評測模板與 token 上限**：我記得 SimpleRL-Zoo 對 0.5B、1.5B 這類小模型用的是較簡單的 prompt，不是 Qwen-Math 的 chat 模板，但這點我**沒有離線驗證**。如果真是這樣，用 qwen_math 模板幾乎必然重現失敗。若需要改，請在 PREREG 寫明「每個配對用 RL 訓練時的模板，base 與 RL 用同一模板」，再在 pairs.json 補一個 per-pair 的 template 欄位。

## 建議（非必修）
- 灰區「dev 400 題合併重算」的程式現在就寫好並 commit，符合「加碼事先定好」。
- WiSE 前緣目前是點與點之間的分段線性內插。若某幾個 λ 點不單調，它就不是上包絡。可以另報「上包絡（凸包上緣）」版本作為更嚴格的敏感度分析。
- `00_correctness_gpu.sbatch` 的 `| tee` 不會傳出失敗碼（沒開 pipefail），要人工看 `results/correctness_<P>.log` 的 ALL PASS。可以接受。
- 01 的 base 若真的常撞到 3000 token，單一 job 可能接近 3 小時；6 小時上限足夠。若想縮短等待，可以用 `SHARD/NSHARD=0/2,1/2`。

## 協調者補充事項（B 配對的 dtype 與 lm_head）的查核
- (a) `make_wise.py`：base（tied）沒有 `lm_head`，RL 多存一份。現行程式丟掉 RL 的 lm_head，沿用 base config 的 `tie_word_embeddings: true`，讓 lm_head 等於內插後的 embed。我**實查 SimpleRL-Zoo 的 lm_head 與 embed_tokens 逐元素相同（max|diff| = 0）**，RL 自己的 config 也是 tie=true，所以這個處理在數學上精確：λ=1 時等於 RL。我**已加斷言**：將來任何一側多出的 lm_head 若不等於該側的 embed_tokens，就直接停，不會默默算錯。
- (b) dtype：`gen_vllm.py` 用 `dtype=pairs.json:dtype`（bf16），`gen_replace.py` 預設也是 bf16，兩個模型一致；`make_wise.py` 用 fp32 內插後存成 bf16；正確性閘門 00 兩側都是 fp32。**所有載入點都已明確指定，無須修改。**
- (c) lock-step 兩模型的 dtype 一致：同 (b)，已確認。
- 配對 A 兩側都是 bf16、tensor 名稱一致，不受影響。

## 我直接修了什麼
1. `slurm/00–04*.sbatch`：`--output` 從相對路徑改為絕對路徑。原本若不在本資料夾送出，log 會找不到位置，job 會直接失敗。
2. `scripts/make_wise.py`：加入 tied lm_head 斷言（見上）。
3. `README.md`：更新已過時的「模型權重尚未下載」敘述。
4. `PREREG.md`：在修改紀錄加了上述機械修正，標明不是判準變更。

## 環境與資源檢查
- 使用 `envs/vllm033`，已存在，沒有引用 neon。
- `ext/WhatRLBrings` 存在。
- 4 個模型完整（各約 3.1GB；B 的 RL 約 7GB）。
- partition h200q 沒有時間上限。
- 磁碟推估 ≤ 25GB；WiSE 暫存放在 `/work/.../icml/tmp`，6 個並行時峰值約 19GB，跑完刪除。群組額度剩約 184GB。

## 送出順序（第一個該送的 job：`00_correctness_gpu.sbatch A`）
```bash
cd /work/ChihChungHsu/cactus8603/icml27-topic-search/killtests/rl-sparse-distill
# 先 commit（必修 1；必修 2、4 的 criteria 一併定案）
sbatch slurm/00_correctness_gpu.sbatch A          # 第一個；看 results/correctness_A.log 是否 ALL PASS
sbatch slurm/04_throughput_probe.sbatch A         # 吞吐實測，超過推估 2 倍就重排
# 通過後，閘門 0（配對 A）
for C in base rl base_greedy rl_greedy; do sbatch slurm/01_vllm_sample.sbatch A $C 0 1; done
python -I scripts/aggregate.py --pairs A --gate 0
# 配對 B：必修 5 確認模板後，才送 00 B 與 01 B
# 判準 1：必修 2、3 落實後，只對合格配對送 02（8 片）、03（6 個 λ）與 rl_hf 對照
```
