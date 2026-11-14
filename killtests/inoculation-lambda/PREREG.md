# PREREG：inoculation 類防禦漏出預測量 Λ 的停損實驗

建立日期：2026-11-14｜依據：reports/2026-11-14-54（定義）與 2026-11-14-56（Fable 判準審查，第 2 節與通用規則）
狀態：**草案，待 Opus 審查、使用者定案後，送 slurm 前 commit**。門檻的唯一來源是 `configs/criteria.yaml`（version 2026-11-14-a），本文與其一致；兩者不一致時以本文為準並視為 bug。

## 0. 範圍與分期

| 期 | 內容 | 狀態 |
|---|---|---|
| A（本次交付，零訓練） | 閘門 1 有限差分；對照兩篇外部論文已報數字（僅參考） | 程式與 slurm 已備，未送出 |
| B（訓練階段） | 閘門 2 重現、閘門 3 雜訊、主判準 | 判讀程式已備（`aggregate.py --phase b`），訓練與 Λ 批次腳本待 A 通過後另寫、另行審查 |

原「判準 0（兩篇外部論文 Spearman ≥ 0.6 才繼續、兩組 < 0.3 停）」**降為參考，不停損**：論文層段只有 5 個位置、Grant 只有約 9 個設定，且 0.3–0.6 無規則；它不過也可能只是設定對不上。

## 1. 凍結的 Λ 版本（主指標只用這一版）

符號：θ 為可訓練參數；L_def 為防禦情境下的訓練損失（token 平均 NLL）；τ 為非誘發情境下「特質回應」的 token 平均 log-prob，L_τ = −τ。

一步 SGD（學習率 η）：Δτ ≈ −η⟨∇τ, ∇L_def⟩ = +η⟨∇L_τ, ∇L_def⟩。

**Λ ≡ ⟨∇_θ L_def, ∇_θ L_τ⟩（取 η = 1）。** Λ > 0 表示在防禦情境下訓練一步會抬高非誘發情境的特質機率（漏出）。

| 項目 | 凍結值 |
|---|---|
| 時間點 | step 0（基礎模型，LoRA B = 0）；k 步前瞻版只當次要，不入判準 |
| 可訓練參數 | LoRA r=64、alpha=128，Qwen2.5-7B-Instruct 全 28 層、q/k/v/o/gate/up/down；step 0 時 A 的梯度為 0，故只取 B 的梯度；不分層（單一組 "all"） |
| A 初始化 | kaiming_uniform（與 peft 預設等價），種子 {0,1,2} 平均 |
| ∇L_def 資料 | 該任務訓練資料前 200 筆（索引 0–199），損失遮罩 = 全 token（Grant 論文預設）；IP 銀行對每筆樣本取全部提示的等權平均（= 取樣期望）；PPS 以層 20 的 hook 加 ε·v |
| ∇L_τ 資料 | 同資料的第 200–399 筆（與訓練梯度不相交），只算回應 token；在三個非誘發情境（none＝僅基線 system prompt、neutral、random 無關句）各算一次後取平均 |
| 分折 | 訓練 200 筆分 4 折，僅用於報告折間離散，Λ 本身用全部 |
| 反駁 2 的 k 步版 | 次要，不判 |

主判準使用的 Λ = 上述設定下 `lambda_summary.csv` 中 ctx = MEAN、tau_set = trait_heldout、group = all 的 `lam`。**任何其他版本（不同層、不同投影、不同筆數、k 步）均標為探索性，不得用來宣稱通過。**

本實作未使用隨機投影，因為只有 LoRA-B 參數（約 8.9×10⁷）可直接內積；故「投影維度」凍結為「無投影」。

## 2. 閘門順序（前三關失敗記為「未能測試」，限 3 個工作天修；修不好算可行性停損，與此實作停損分開記；要升級為題目淘汰，須另外符合 57 號第 5 節規則 2）

### 閘門 1：有限差分（程式正確性，約幾分鐘～數十分鐘 GPU）
- 做法：`compute_lambda.py --finite_diff`。fp32；η 由 `none` 防禦校準，使 none 的預測 Δτ = 0.01 nats；對每個防禦實際做 B ← B − η·∇L_def（以及 B ← B + η·∇L_def 做中央差分），量測 τ 的實測變化，取中央差分 (nll(+η) − nll(−η))/2 為實測 Δτ；預測 Δτ = η·Λ。
- 量測範圍：Grant 七個 trait 設定的各個防禦（none、IP、PPS；sycophancy 另含 13 個 arm 中的其餘防禦），僅用 ctx = none，LoRA seed 0。
- **PASS：所有 (trait, 防禦) 合併的 Pearson(預測, 實測) ≥ 0.9。** 否則視為程式錯誤，進入 3 個工作天的修復期。
- 附帶報告：實測/預測的迴歸斜率（應接近 1；偏離表示 η 過大、二階項顯著）。
- 2026-11-14（L0）附加報告：每個 trait 的 Pearson、斜率，以及預測 |Δτ| 的最小值（`aggregate.py --phase a` 輸出，只描述，不參與 PASS/FAIL）。
- 2026-11-14（L0）`fd_max_step=0.01` 就此凍結。job 26337 的步長掃描只能拿來診斷，不得用來重新挑上限。26333 若仍 FAIL、要再改上限，須依序：先寫診斷、commit 新值、再重跑；所花時間計入 3 個工作天。

### 閘門 2：重現（需訓練階段資料）
- 無防禦的漏出，以及 IP 的降幅，方向正確且達論文已報效果的一半（Grant Table 13：例如 sycophancy 94.2 → IP 17.8、evil 91.6 → 31.6）。未過 = 「未能測試」。

### 閘門 3：雜訊（1 個任務 × 5 arm × 3 seed）
- arm 間信度 ≥ 0.8（ICC(1,1)，3 個 seed 作重複），否則換任務；最多換 2 次。

## 3. 主判準（訓練階段；閘門 1–3 皆過才判）

**唯一主指標**：在 13 個「有防禦」的 arm 上（不含無防禦 arm），各任務內以 Λ 對最終漏出量計算 Spearman ρ_task，跨三個任務以 Fisher z 合併（權重 n_arm − 3）得合併 ρ。漏出量 = 六族非誘發提示的平均特質表現（以可程式判定任務：MBPP hard-coding、Spanish+ALL-CAPS、GCD 諂媚），arm 內對 seed 取平均。

任務與 arm（沿用 54 號報告 §6，13 個有防禦 arm：IP 6 種提示〔強、弱、無關、改寫等〕、PPS 2 強度、SIP、僅訓練後段層的 IP、簡化版 NIP 2 個 token〔重建過久則以 random-row token 代理，需註明〕，其餘填足 13 的組成在 Phase B 腳本審查時定案，**於送訓練前寫入 `configs/phaseB_arms.yaml` 並 commit**）。每任務另含 1 個無防禦 arm，僅作 ID 匹配與 gate 2 之用，不入主指標。
- 家族標籤：ip_single、ip_bank、pps、sip、nip、layer_restricted。
- seed：每 arm 2 個 seed（0、1）；LoRA r=32、lr 與 epoch 沿用 54 號報告；資料切分固定（訓練 5k 筆、Λ 與漏出評估用的 held-out 另列）。

三態判定（`aggregate.py --phase b`，機械輸出）：
- **PASS**：合併 ρ ≥ 0.6；**且** 依任務分層 bootstrap（arm 內重抽，10000 次，種子 20261114）95% CI 下界 ≥ 0.3；**且** 三個任務的 ρ_task 都 > 0；**且** 拿掉任一防禦家族後合併 ρ 仍 ≥ 0.4。
- **灰區**：合併 ρ ∈ [0.35, 0.6)，或 ρ ≥ 0.6 但上述附帶條件未全過。**唯一加碼**：加 1 個 seed 與 6 個 arm，一次；以同一腳本重判，不得再加碼。
- **FAIL**：合併 ρ < 0.35。
- 「m 個擇一」時顯著水準除以 m：本判準無擇一分支（單一主指標），故 m = 1。

「贏過基線」（獨立於 PASS/FAIL 報告）：基線事先指定為 **Grant 層 20 梯度餘弦**（step 0，同 200 筆）。在基線有定義的共同 arm 上，配對 bootstrap 算 Δρ = ρ(Λ) − ρ(基線)，**90% CI 下界 > 0 才算贏過**。「可涵蓋的防禦種類」另列表報告（基線對 token、adapter、層限制、SIP 無定義）。其他既有預測量（Wichers 誘發強度、Tan loss 下降、PHM persona 距離）只報告、不入判準。

匹配 ID：目標 = 該任務無防禦 arm 最終 ID 學習程度的 90%（±5 個百分點）。達不到的 arm 在 CSV 的 `id_ok` 欄標 False，列於輸出；主指標仍含這些 arm，另報排除後的 ρ 作敏感度分析。

判準 2（IP 提示選擇 top-1／bottom-1 判對率）：**只描述，不判。**

## 3a. 2026-11-14 定案條文 L1–L3（代理落差稽核後、結果前；非探索性）

來源：`reports/2026-11-14-59-proxy-gap-decisions.md` §4–§5 與 `58-proxy-gap-audit.md` §2。實作：`scripts/phaseb_extras.py`（不改 `aggregate.py` 主判準邏輯）、`configs/criteria.yaml` 的 `phase_b_extras`。

**L0 閘門 1 修復**：見 §2 與修訂紀錄（`fd_max_step=0.01` 凍結、加報逐 trait 數字、寫進代理落差）。

**L1 信度上限**
- 在 13 個有防禦的 arm 上分別計算：r_L = 漏出在 seed 0 與 seed 1 之間的 Spearman，用 Spearman-Brown 校正到 2 個 seed；r_Λ = Λ 在 LoRA 初始化 seed {0,1,2} 之間兩兩 Spearman 的平均，用 Spearman-Brown 校正到 3 個 seed。
- C_task = √(max(r_L,0)·max(r_Λ,0))；三個任務用 Fisher z **等權**合併成 C。
- 若 C < 0.75（`ceiling_min`），主判準的 FAIL（ρ<0.35）改記「未能測試（信度上限不足）」。PASS 與灰區照原規則判定。

**L2 地平線歸因**
- Phase B 每個 arm 的 seed 0，存 1.25%、2.5%、5% 步與最終的 LoRA（`checkpoint_fracs`）。
- Δτ_5% = 5% 步時的 τ 減去 step 0 時的 τ。S5 = 各任務 Spearman(Λ, Δτ_5%) 的 Fisher z 合併值（權重沿用主判準的 n_arm−3；條文未另指定，於實作時採此並在此註明，結果前決定）。S5 ≥ 0.8（`horizon_s5_min`）記「短地平線成立」，否則「不成立」。S5 不擋主判準。
- 只描述的分解回報：Λ 對 Δτ_5%、Λ 對 Δτ_final、Δτ_final 對行為漏出。

**L3 唯一的第二探針**
- 觸發：主判準 FAIL，且 L1 的 C ≥ 0.75。
- 版本由 L2 決定：短地平線成立 → 比值版 Λ_τ/Λ_ID（Λ_ID ≤ 0 的 arm 排除；排除超過 3 個則探針記「未能測試」）；不成立 → k 步版（step 0、1.25%、2.5%、5% 四個時點的 Λ 等權平均）。
- 用主判準同一組 PASS 條件判一次（灰區也記「未通過」），上限 5 GPU 小時，不加碼。
- Λ_τ/Λ_ID 作為次要量可照常輸出，但它同時是 L3 的候選探針；**探針結論只由本條產生**。
- 探針也未通過，才可提請淘汰。

輸入表格式（Phase B 的 Λ 批次與評估腳本須產出）見 `scripts/phaseb_extras.py` 檔頭；`--selftest` 以合成資料測過 L1（高信度／信度不足三種原判定）、L2、L3 兩版本、排除 3／4 個 arm、探針通過與未通過。

## 3b. 代理落差（2026-11-14 補；已列＝原 PREREG／審查已寫；未列＝本次新增）

| 落差 | 狀態 |
|---|---|
| 只用 step 0 單點，但理論主張是 Σ_t Λ_t；k 步版原列為次要，現為 L3 的候選版本 | 已列→L3 |
| Λ 未正規化（η=1 的原始內積），卻拿去預測匹配 ID 下的漏出。一階近似下漏出約正比於 Λ_τ/Λ_ID；原始 Λ 會把「防禦讓整體學得慢」也算進來 | 未列→L3 比值版 |
| τ 是固定留出回應的 log-prob，漏出是生成時的行為率；log-prob 到行為率的轉換沒有檢驗 | 未列→L2 分解（Δτ_final 對行為漏出） |
| 漏出在各 arm 間的信度（強防禦 arm 漏出近 0、並列多）與 Λ 對 LoRA seed 的信度限制 Spearman 上限 | 未列→L1 |
| 閘門 1 只是單步正控制，多步下沒有正控制 | 未列→L2 |
| 閘門 1 的步長：Spanish 是 Phase B 的任務；PPS 的梯度範數是 none 的 14 倍（25.6 vs 1.89），而原始內積 Λ 正比於 ‖∇L_def‖。若 Phase B 用 Adam 類最佳化器，更新量不會隨梯度範數縮放，原始內積 Λ 與實際更新的對應會更弱〔推導，未實測〕 | 未列 |

## 3c. 三層判定（FAIL 預設只算到實作層）

| 層 | 主判準 FAIL 最多否證什麼 | 測到了嗎 |
|---|---|---|
| 實作層 | 「step 0、未正規化的 Λ 對最終行為漏出的排序」 | 是（本實驗） |
| 機制層 | 一階轉移在多步下是否成立 | 閘門 1 只驗單步；L2 以 5% 步補部分驗證 |
| 方向層 | 「有一把統一的尺能預測防禦漏出」 | 否；未被否證 |

54 號 §6 的「判準 1 前半失敗 → 停」限縮為「停此實作（step 0 原始 Λ）」；要升級為題目淘汰須 L3 探針也未通過並符合 57 號第 5 節規則 2。

## 4. 可重試次數、成本上限
- 閘門 1–3 失敗：每關最多 3 個工作天修復，修不好記為可行性停損（與此實作停損分開記，不等於題目淘汰）。
- Phase A 成本上限 12 GPU 小時（預估約 4–6，見 README）；Phase B 上限 90 GPU 小時（54 號推估）。每個階段開始前先做 10 分鐘吞吐實測，實測推算超過推估 2 倍就停下重排。
- 灰區加碼：一次。
- L3 第二探針：上限 5 GPU 小時，不加碼；全題 Phase B 上限 90 GPU 小時不變（L2 另存 checkpoint 的額外成本預估 3–5 小時，計入該上限，超出先回報）。

## 5. 原始數字保存
每個 (seed, fold, 防禦, tau_set, ctx, group) 的逐列結果都在 `results/<name>/lambda_raw.json`（含 dot、梯度範數、cos、train loss、τ NLL、有限差分欄位）；config 在 `configs/`、criteria 版本寫入 aggregate 輸出。失敗與重跑不刪，保留於 `results/` 與 `logs/`，重跑須換目錄名並記入 `logs/RUNLOG.md`（job id、git hash、日期、原因）。

## 6. 判讀流程
腳本機械輸出 → Opus 對照本文審 → 使用者定案。實作者與提案者不自行判定。

## 7. 對照外部論文數字（參考，Phase A；不停損）
- 2604.16423（Grant）：Table 13（7 trait × IP/PPS 的保留比例）、Table 10 prevention（sycophancy 13 個防禦 arm + 九個單一提示，並以其自報 g20 重算 ρ=0.65 驗證抄錄與基線）。
- 2611.12864：Fig. 5 的 bottom→top 層段斜率（8 個 (模型, ID→OOD) 格）。Λ 的對應量為層段 c 上 R_c = Λ_OOD(c)/Λ_ID(c)，意義是 dz_OOD/dz_ID 的一階代理。
- 報告「支持」(ρ ≥ 0.6)、「中間」、「弱」(ρ < 0.3) 三種描述；**不觸發停損**。

## 修訂紀錄
- 2026-11-14：初稿（依 Fable 審查）。
- 2026-11-14（Opus 審查，機械修正，非判準變更）：`slurm/a1_gate1_fd.sbatch` 時間上限 4h → 6h（推估 1.5–3h，原上限不足 2 倍餘裕；逾時會讓 a2/a3 的 afterok 相依全取消）。審查意見見 `REVIEW.md`。
- 2026-11-14（閘門 1 於 GPU FAIL 後的程式修正，job 26261：pooled r=-0.579、n=34、斜率 -7.58；門檻 0.9 未動）。原因：有限差分的步長 eta 由 none 防禦校準到預測 Δτ=0.01 nats，對低 cos 的 trait（spanish none cos=0.16、optimistic 0.10，Λ 僅 0.14／0.07）得到很大的 eta（0.073／0.141），步長 eta·‖gL‖ 達 0.14–0.25（其餘 trait 僅 0.003–0.01）；spanish 的 pps_eps2 梯度範數為 none 的 14 倍（25.6 vs 1.89），步長 1.85，落入強非線性區（預測 -0.032，實測中央差分 +0.833），單點即把 pooled r 拉成負值。其餘 5 個 trait（evil、sycophantic、allcaps、pirate、hallucinating）預測與實測相符到 1–3 位有效數字，Λ 的實作與符號慣例無誤。修正：`compute_lambda.py` 的 `fd_apply` 加步長上限 `eta ← min(eta, fd_max_step/‖gL‖)`，`criteria.yaml` 新增 `gate1.fd_max_step: 0.01`（線性區信賴半徑，不是判準門檻；其餘 trait 的步長本來 ≤0.011 不受影響）。判定仍用全部 7 個 trait。
- 2026-11-14（代理落差稽核後、結果前定案，非探索性）：依 59 號報告 §4–§5 新增 L0–L3（§2、§3a）、代理落差與三層判定（§3b、§3c）；`criteria.yaml` 升為 2026-11-14-b，僅新增 `gate1.report_per_trait` 與 `phase_b_extras` 區塊，既有門檻與 `fd_max_step=0.01` 一字未動；`aggregate.py --phase a` 只新增逐 trait 描述輸出，判定邏輯不變；新增 `scripts/phaseb_extras.py`。診斷檔 `DIAGNOSTIC_spanish_fd.md`、`scripts/diagnose_spanish_fd.py`、`slurm/a1_spanish_step_diagnostic.sbatch` 納入版本控管（job 26337，只診斷）。26333–26336 排隊中所用檔案（compute_lambda.py、a1–a4 sbatch、grant_*.yaml）未改。
