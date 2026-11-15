# inoculation-lambda：Λ 的第 1–3 天零訓練檢驗

建立：2026-11-14｜狀態：**已備妥，未送出任何 sbatch／srun，未佔 GPU，未 commit。** 判準與門檻的唯一來源見 `PREREG.md` 與 `configs/criteria.yaml`（Fable 重審後版本 2026-11-14-a）。

## 1. 目的與對應回報
- 對應回報：`reports/2026-11-14-54-deepcheck-inoculation-leakage.md` §5–§6 的「第 1–3 天零訓練檢驗」，以及 `2026-11-14-56-killtest-criteria-review.md` 第 2 節（Λ）。
- 檢驗 Λ ≡ ⟨∇L_def, ∇L_τ⟩（step 0、LoRA-B 或指定層參數）：
  1. **閘門 1（程式正確性，有判準）**：Λ 對「實測 1 步 SGD 的 Δτ」的 Pearson ≥ 0.9，否則視為 bug。
  2. **參考（無判準、不停損）**：把 Λ 與兩篇外部論文已報數字對照，輸出 Spearman。
- 依 Fable 審查，原「判準 0（兩組 Spearman ≥ 0.6 才繼續、< 0.3 停）」降為參考；閘門 2（重現）、閘門 3（雜訊）與主判準屬訓練階段（Phase B），本次不執行，判讀程式已備。

## 2. 判準原文
54 號報告原判準 0（已降級為參考）：「兩組外部結果中，至少一組的 Spearman ≥ 0.6。兩組都 < 0.3 就直接停，改走別題。」
56 號 Fable 審查（採用）：
- 判準 0 降為參考，不停損。
- 凍結單一 Λ 版本：step 0，固定層、投影維度、200 筆；k 步版只當次要。
- 相關只算有防禦的 13 個 arm，各任務內算 Spearman，再跨三個任務合併（Fisher z，依任務分層 bootstrap）。PASS：合併 ρ ≥ 0.6、CI 下界 ≥ 0.3、三個任務都 > 0，且拿掉任一防禦家族後仍 ≥ 0.4；灰區 0.35–0.6：只加 1 個 seed 與 6 個 arm 一次；FAIL：< 0.35。
- 「贏過基線」：共同有定義的 arm 上配對 bootstrap，Δρ 的 90% CI 不含 0；基線事先指定一個；可涵蓋的防禦種類另報。
- 匹配 ID：事先固定目標（無防禦最終值的 90% ±5pp），達不到的 arm 標記並列出。判準 2 只描述不判。
- 閘門順序：(1) 有限差分 Pearson ≥ 0.9（幾分鐘）→ (2) 重現無防禦漏出與 IP 降幅（方向對且達已報效果一半）→ (3) 1 任務 5 arm × 3 seed 的 arm 間信度 ≥ 0.8。

## 3. 外部程式與 checkpoint 現況
| 論文 | 公開程式（已 clone） | 模型／資料 | 公開 checkpoint／LoRA |
|---|---|---|---|
| 2611.12864 Is Persona Generalization Predictable? | `ext/persona-generalization`（github.com/olimu/persona-generalization，HEAD 2026-10-25，只讀 ~1.8MB） | Qwen2.5-14B/7B-Instruct、Llama-3-8B-Instruct、OLMo-3-7B-Instruct、Falcon-Mamba-7B；EM 資料需下載上游加密檔（bad_medical_advice、extreme_sports），4 個 OOD 集 repo 內附；評審 Qwen3.6-27B-FP8 | **無**（README 僅說明訓練輸出路徑）。零訓練可行：論文 chunk 實驗是「全參數訓練 3 層」，Λ 在 base 模型上對各層段參數直接算內積，不需 checkpoint |
| 2604.16423 Shifting the Gradient（NeurIPS 2026） | `ext/shifting_the_gradient`（github.com/satchel-goodfire/shifting_the_gradient，MIT，136MB，含訓練資料、IP 提示檔、全部 persona vector、`configs/paper_*.json`） | Qwen2.5-7B-Instruct（主）、14B、Qwen3-4B-Instruct-2507、Llama-3.1-8B；評審 gpt-4.1-mini（需 OpenAI API，本次不需要） | **無**（`sg-paper` 一鍵重訓全部 502 個 run，每個約 1.5–2.5 H100 小時）。零訓練可行：Λ 只需 base 模型 + step 0 |

最便宜的替代（已採用）：兩邊都只用 base 模型在 step 0 計算 Λ，對照論文表格已報的漏出／斜率；唯一「實測」對照是閘門 1 的 1 步有限差分。
已閱讀的程式部分：`finetune_lora.py` 與 `dataset.py`（損失遮罩為全 token、IP 以 prepend_user 插入、基線 system prompt "You are an LLM chatbot."、LoRA r64/α128）、steering hook（層 20 對應 `model.layers[19]`、對所有位置加 ε·v）、`traits_common.py`／`ipablate_common.py`（各 trait 的 n、ε、提示檔）；persona-generalization 的 YAML 與 `download_data.py`／`prepare_data.py`。**未執行兩個 repo 的任何安裝或下載腳本**；EM 資料下載步驟由 `scripts/prepare_data.sh` 照 `download_data.py` 重寫（使用其公開 URL 與上游文件所載密碼），已執行完成。

## 4. 要對照的論文數字（出處）
全部放在 `configs/reference.json`，聚合時讀取。
- **Grant Table 13**（2604.16423，Qwen2.5-7B）：7 個 trait 的 base／無防禦／PPS／IP 表現，例如 evil 0.0/91.6/32.8/31.6、sycophantic 1.6/94.2/37.1/17.8、hallucinating 20.4/99.3/86.0/60.7。比較量：Λ_d/Λ_none 對 expr_d/expr_none（保留比例），14 點。
- **Grant Table 10 prevention 欄 + 附錄 G 文字**：sycophancy 九個單一提示 5.0／5.2／4.6／10.8／2.3／2.6／2.4／14.1／2.9，3/4/5 提示銀行 41.1／52.6／79.5，PPS ε=2 為 37.2、ε=9 為 3.2，無防禦 94.1；及論文自己的 g20。以九個單一提示重算 ρ(g20, expr) = 0.65，與論文所報 0.65 一致，用來驗證抄錄無誤並提供基線。
- **2611.12864 Figure 5**（PDF p.7，人工讀圖）：各 (模型, ID→OOD) 的最底層→最頂層 chunk 斜率 β：Qwen14B med→sports 1.07→0.74、med→cyber 0.88→0.64、sports→med 0.74→0.40；Qwen7B 1.06→0.65、0.77→0.60、0.68→0.32；OLMo-3-7B med→sports 1.15→0.69、med→cyber 1.29→1.36（唯一反例）。每個 chunk 的完整斜率只畫在 Fig. 4 曲線上，HTML 與 PDF 文字層皆無數值，因此只用上述有標註的兩端，並另報「深度 vs. R 的 within-cell 秩相關」（論文：49/50 條曲線對深度為負）。Λ 的對應量：R_c = Λ_OOD(c)/Λ_ID(c)（dz_OOD/dz_ID 的一階代理）。

## 5. 檔案
- `PREREG.md`：預先登記（主指標公式、門檻、資料與 seed、重試次數、成本上限、閘門順序、三態結果、灰區唯一加碼）。
- `configs/criteria.yaml`：所有門檻，集中、易改；`configs/reference.json`：外部論文數字；`configs/*.yaml`：由 `scripts/gen_configs.py` 產生（Grant 七個 trait；2611 的 Qwen7B/14B × ID∈{medical, sports}、OLMo7B × medical）。
- `scripts/compute_lambda.py`：Λ 批次計算（多防禦、多 τ 集、多情境、多 A 種子；`--finite_diff` 為閘門 1；`--smoke` 為 CPU 極小測試）。
- `scripts/aggregate.py`：`--phase a` 輸出閘門 1 PASS/FAIL 與參考 Spearman；`--phase b --table ...` 為主判準（PASS/GRAY/FAIL、基線 Δρ、家族剔除、ID 匹配標記），輸入欄位 task, arm, family, seed, leak, lam, base_pred, id_ok。
- `scripts/setup_env.sh`、`download_models.sh`、`prepare_data.sh`、`smoke_cpu.sh`；`slurm/a1_gate1_fd.sbatch`、`a2_grant_lambda.sbatch`、`a3_pg_lambda.sbatch`、`a4_aggregate.sbatch`。
- `results/`、`logs/`：輸出；`data/em/`：EM 資料（含有害文本，已 .gitignore）。

## 6. 環境、模型、空間
- 環境：既有 envs 皆無 peft，且不宜改動；新建 `/work/ChihChungHsu/cactus8603/icml/envs/inoc`（uv venv：torch 2.8.0+cu128、transformers 4.57.6、scipy、pyyaml、easy-dataset-share；已建好並通過 import 測試）。本實作**不需要 peft**（LoRA 自行實作，與 peft 預設初始化等價）。
- 模型（HF_HOME=/work/ChihChungHsu/cactus8603/icml/hf_home）：本機原有的只有 Qwen2.5-0.5B／1.5B（smoke 用）與 Qwen3 系列，皆不是論文用的；`scripts/download_models.sh` 下載 Qwen2.5-7B-Instruct（約 15GB）與 Qwen2.5-14B-Instruct（約 30GB）；OLMo-3-7B-Instruct（約 15GB，選配，`INCLUDE_OLMO=1`）預設不下載；Llama-3-8B（gated）與 Falcon-Mamba 不跑。
- **預估空間（上限 80GB）**：env 6.5GB + Qwen7B 15GB + Qwen14B 30GB + EM 資料與結果 < 0.5GB ≈ **52GB**；含 OLMo 約 67GB。uv 下載快取在使用者共用的 `.cache/uv`（目前已 12GB，非本題獨佔，未計入；若需要可由使用者清理），本次已改以 `UV_CACHE_DIR` 指向 `icml/tmp`。大檔皆在 `.gitignore`。
- **本題下載狀態（重要）**：Qwen2.5-7B-Instruct 已完整（15GB，可用）。**Qwen2.5-14B-Instruct 未完成**（約 6.7GB 的 `.incomplete`，第一次下載遇 `Disk quota exceeded`，我又重啟一次後依協調者通知（群組額度已滿）立即中止，之後未再重試）。補完需再下載約 25GB：額度恢復後執行 `scripts/download_models.sh`（可續傳）；補完前 a3 腳本會自動只跑 7B（2611 的 14B 格暫缺，Fig.5 的 8 格中 3 格受影響，可用點數 16 → 10）。OLMo 未下載。

## 7. 預估 GPU 小時（推估，須先做 10 分鐘吞吐實測，超過 2 倍就重排）
| 工作 | 內容 | 預估 |
|---|---|---|
| a1 閘門 1 | 7 trait fp32 有限差分（sycophancy 15 個防禦最多） | 1.5–3 h |
| a2 Grant Λ | 7 trait × 3 A 種子 × 4 折，bf16 | 1–2 h |
| a3 層段 Λ | Qwen7B（14B 待下載完成）× 2 個 ID 域，5 層段，全參數 | 1–2 h |
| a4 彙整 | CPU 為主 | ~0 |
合計約 **4–7 GPU 小時**（上限 12，見 criteria.yaml 的 cost）。Phase B（訓練）另計，54 號推估約 60–90 GPU 小時，**本次不做**。

## 8. 送出指令（寫出來，不要執行）
```bash
cd /work/ChihChungHsu/cactus8603/icml27-topic-search/killtests/inoculation-lambda
# 前置（登入節點、不佔 GPU）：環境與資料已完成；模型下載確認 logs/download_models.log 結束後：
sbatch slurm/a1_gate1_fd.sbatch                      # 閘門 1，先跑；FAIL 則停下修程式
sbatch --dependency=afterok:<a1_jobid> slurm/a2_grant_lambda.sbatch
sbatch --dependency=afterok:<a1_jobid> slurm/a3_pg_lambda.sbatch
sbatch --dependency=afterok:<a2_jobid>:<a3_jobid> slurm/a4_aggregate.sbatch
```
CPU smoke（已執行過一次）：`scripts/smoke_cpu.sh`。

## 9. 風險與未確認事項（請審查者特別注意）
1. **Λ 的定義取捨**：τ 用「同資料集 held-out 的特質回應在非誘發情境下的 NLL」，非 judge 判定的特質行為；對 evil 類資料，整段回應 NLL 混有風格與一般語言成分。Grant 自己指出銀行的防禦效果來自跨步一致性而非單步梯度幾何，一階 step-0 的 Λ 預期無法區分銀行與其成員（bank3/4/5 vs 單一提示），這是已知的失敗模式。
2. **有限差分與 LoRA-B**：step 0 時 A 的梯度為 0，Λ 只含 B；η 由 none 防禦校準到 0.01 nats，若二階項顯著則斜率偏離 1。閘門 1 用 ctx=none、seed 0；在 sycophancy 上若 Λ_none ≤ 0 會直接 assert 失敗（資料／實作有誤的訊號）。
3. **2611.12864 的 Λ 對應**：論文做的是全參數 3 層訓練；我們也用全參數梯度，但 Λ 是一階量，且論文斜率是 probit 空間擬合，R_c 只是代理。chunk 斜率只有兩端數字可用（人工讀圖，請核對 Fig. 5）；OOD 資料只有 85–396 筆；訓練與 ID held-out 的洗牌種子為自訂（0），與論文分割不同。無 Llama-3-8B、Falcon-Mamba。
4. **Grant 設定對不上的可能**：未使用 LoRA dropout 0.05（eval 模式求確定性梯度）；allcaps 資料由 evil/normal.jsonl 轉大寫（照其 datasets.py）；Grant 每個 trait 的 n、ε 取自 Table 13，hallucinating 為 PPS ε=9。Table 13 與 Table 10 的數字由 arXiv HTML 抽取，Table 10 已以 ρ=0.65 重算驗證，Table 13 未能獨立驗證。
5. **參考相關的統計力**：Grant Table 13 的 14 點、Table 10 的 14 點與 9 點、2611 的 16 點，ρ 的 95% CI 約 ±0.4，故只當參考。
6. **Phase B 未寫**：主判準的訓練、13 個 arm 的組成（PREREG 載明須先寫入 `configs/phaseB_arms.yaml` 並 commit）、漏出評估尚未實作；`aggregate.py --phase b` 僅以合成資料驗證過邏輯（若有跑，見 logs）。
7. smoke 結果（`logs/smoke_cpu.log`，Qwen2.5-0.5B／CPU／2 筆樣本）：Grant 式 Λ 為正（none 14.3、cos 0.27；IP 13.1；PPS ε2 10.0，皆合理）；有限差分預測 Δτ 與實測（中央差分）none 0.0100/0.0100、ip 0.00915/0.00915、pps 0.00687/0.00687（相符至 4 位，單邊差分差約 1–2%），閘門 1 邏輯可跑通（n=3，不具統計意義）；2611 式 5 層段全參數 Λ 與 R 可算出；主判準 `--phase b` 以合成資料驗證可輸出 PASS/GRAY/FAIL。以上皆只證明程式跑得通。
   另：smoke 發現 pg 設定原會多算一個「全層 all 組」，已改為 `groups: []`（只算 5 個層段，14B 記憶體才夠），smoke 後才改，改後只以 resolve_groups 單元檢查，未再跑模型。
   （原第 7 點說明）smoke 為 0.5B／CPU／2 筆樣本，只證明程式跑得通與符號合理，不代表任何科學結論。
8. 登入節點負載高（load ~50），smoke 很慢；GPU 上 7B／14B 的記憶體與吞吐為推估，14B 全參數層段取 5 層段（約 15 層）以 fp32 累積，預估 < 60GB 額外記憶體，未實測。

## 10. 2026-11-14 代理落差稽核後補充（結果前定案）
- `scripts/phaseb_extras.py`：Phase B 的 L1 信度上限、L2 地平線歸因（S5）、L3 唯一第二探針；`--selftest` 為合成資料測試（`nice $PY scripts/phaseb_extras.py --selftest`）。條文見 `PREREG.md` §3a–§3c。
- `aggregate.py --phase a` 另輸出逐 trait 的 Pearson、斜率、預測 |Δτ| 最小值（只描述）。
- Phase B 訓練腳本（尚未寫）須：每 arm 的 seed 0 存 1.25%、2.5%、5% 步與最終 LoRA；產出 `lambda_seeds.csv`、`horizon.csv`（欄位見 phaseb_extras.py 檔頭）。
- FAIL 的用語：此實作（step 0 原始 Λ）停損，不等於題目淘汰。
