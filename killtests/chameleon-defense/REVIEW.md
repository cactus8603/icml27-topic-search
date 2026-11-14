# REVIEW：chameleon-defense 停損實驗送 GPU 前審查（Opus，2026-11-14）

**結論：修完可送出。** 必修 2 項（M1、M2）都會改判準，我沒有動，要由主對話決定並記入 PREREG 修訂紀錄後再送。程式面的小 bug 已修（見「已修」）。

依據：`reports/2026-11-14-56-killtest-criteria-review.md`（第二梯建議、通用規則）、`reports/2026-11-14-49-relaxed-neurips-scan.md` 候選 3。讀過 PREREG、README、config、`scripts/` 全部、`slurm/` 全部；也讀了作者 repo 的 `results/evaluation/*/harmful_synthetic/summary.csv` 與 `docs/PIPELINE_GUIDE.md`。

---

## 0. 審查中查到的關鍵事實（影響判斷）

1. **作者的 4 層、8 層結果是另外訓練的 organism，不是同一個 organism 換監測器。** `docs/PIPELINE_GUIDE.md` 第 53–57 行：`organism_8layer_logistic` 對應 checkpoint `sep08_ablit_12ablit-synth-8l_...`，是「對 8 層探測器訓練逃避」的 organism，屬於探測器深度的消融。原停損「比 8 層 ensemble 高 20pp」在論文脈絡中，指的是**被訓練來躲 8 層 ensemble 的 organism**。在那個 organism 上，8 層 ensemble 的 harm TPR@1%FPR：乾淨 0.795，`harmful` 0.079、`harm` 0.063，餘量很大。
2. 我們手上只有第 12 層 organism。在它上面，**自訓的 8 層 ENS 是否會被躲過，repo 裡沒有任何數字。** 你的疑慮成立：如果第 4、20–32 層沒被擾動，ENS8 在觸發下的 TPR 可能接近乾淨值。這時 ROB 不管多好，Δ 都達不到 0.20，FAIL 就只反映 organism 不合適。
3. 作者 repo 有 `base_model/harmful_synthetic/summary.csv`：未經逃避訓練的模型加上觸發詞後，第 12 層作者探測器的 TPR 不降反升（乾淨 0.842，觸發後 0.874–0.888）。可見 organism 上的 TPR 掉落確實是學到的逃避，不是提示文字本身造成的。這讓閘門 2 的重現有意義。
4. 作者探測器檔案（`probes/*.pt`）用 pickletools 靜態看過，沒有執行：格式是 `{0: {'weight': np.float32 (1,3584), 'bias': ...}}`，與 `extract_acts.py` 的讀法一致。第 12 層對應 `hidden_states[13]`，與 minimal demo 相同；只取回應 token，平均方式也與 demo 一致。CPU smoke 用的是 D=64，這條路徑沒被執行到，但靜態比對沒有問題。

---

## 1. 三個問題的判斷

### (a) PASS、灰區、FAIL 能不能解讀

在**現行**判準下（沒有 M1）：

| 結果 | 能否解讀 | 理由 |
|---|---|---|
| PASS | 可以，但範圍窄 | ENS8 若偏強，對 ROB 只會更難，所以 PASS 是保守的正面證據。結論只限「第 12 層 organism、harm、T0 觸發」。要宣稱「好處來自逃避子空間」，還需要 ROB−RND 的 CI > 0（見建議 S3）。 |
| 灰區 | 只能當參考 | 而且現行的加碼方式縮不了 CI（見 S1）。 |
| FAIL | **不能解讀** | 無法區分「方法沒用」和「organism 根本沒躲 8 層基線」：ENS8 觸發後 TPR 若接近乾淨值，Δ 的上限就小於 0.20。這正是你提的疑慮，而且第 0 節第 1 點顯示，原停損本來就是針對另一種 organism 設的。 |

加上 M1（閘門 2b）之後：

- **2b 通過時**（ENS8 確實被躲過，留有 ≥30pp 餘量），FAIL 可以解讀成「在這個 organism 上，子空間擴增救不回 TPR」，算數。
- **2b 沒過時**，記為「未能測試主假設」，不是 FAIL。這個結果本身也有資訊：公開 organism 的逃避不會轉移到多層監測器。要做這題就得自訓多層 organism，而那需要 gated 的 Gemma，等於要登入，由使用者決定。

### (b) 要不要改主指標或基線

- **主指標維持 Δ8 = ROB8 − ENS8，不建議換成「只比第 12 層」。**
  - 只打贏一個 organism 專門被訓練去躲的單層探測器，是很弱的基線，審稿人不會買單。原作已經顯示 ensemble 能部分緩解逃避，題目的價值在於贏過 ensemble。
  - 臨時把主指標改成第 12 層，等於挑容易贏的比較。
- **但必須加一道預先登記的 organism 適用性閘門（M1）當重現閘門的一部分。** 你提的「先驗證 organism 在第 12 層確實逃避」方向正確，但要驗證兩件事：
  1. **我們的探測器型式**（平均池化邏輯迴歸）在第 12 層也被躲過。閘門 2 只驗證了作者的逐 token 探測器。
  2. **ENS8 本身有餘量。**
- 第 12 層的同層比較 ROB12−ENS12，只在 2b 的 H8 不過時當參考輸出（已加進 analyze 的回報欄位），不得當作題目 PASS。

### (c) 跑還是不跑

**跑。** 理由如下：

- 推估約 1.2 GPU 小時，激活快取約 2.1 GB（36,418 列 × 8 層 × 3584 × fp16，群組額度約剩 160 GB）。
- 光是閘門 2 與 2b，就能回答「公開 organism 能不能拿來做這題」，資訊量對成本比高。

解讀規則：

| 情況 | 解讀 |
|---|---|
| 2b 通過 + PASS 或 FAIL | 算停損證據，但只限「第 12 層 organism」這個 organism。 |
| 2b 通過 + PASS | 不足以代表題目整體成立：論文要的設定是對 ensemble 訓練逃避的 organism，加上自適應攻擊者（第 2 週）。 |
| 2b 沒過 | 主假設記「未能測試」；Δ12 只當參考。 |
| 偵測器 | 一律只當參考（M2）。 |

不加 M1 就送的話，FAIL 沒有診斷力，整份結果只能當參考。所以結論是「修完可送出」，不是「可送出」。

---

## 2. 必修（會改判準；我沒改，請主對話決定後寫入 PREREG／config／aggregate 並記日期）

### M1. 新增閘門 2b「organism 適用性」

放在閘門 2 之後、閘門 3 之前，凍結在 `config.thresholds.gate2b`，由 `aggregate.py` 機械判定。兩項檢查需要的數字，`monitors.json` 都已經有，其中 `per_layer_ENS_tpr_clean` 是本次新增。

- **H12（逃避會轉移到本實驗的探測器型式）**：第 12 層 ENS 的乾淨 TPR 減去觸發 TPR（harm、T0、7 個觸發詞平均、1% FPR），要 ≥ 0.30。
  - 資料來源：`per_layer_ENS_tpr_clean[i12] − per_layer_ENS_tpr_primary[i12]`。
  - 不過：記「未能測試（逃避不轉移到平均池化探測器）」，屬可行性停損。
- **H8（8 層基線留有餘量）**：ENS8 乾淨 TPR 減去觸發 TPR，要 ≥ 0.30。
  - 資料來源：`families.ENS.tpr_per_cond.clean.mean − families.ENS.tpr_primary_mean`。
  - 不過：記「未能測試主假設（organism 未躲過 8 層基線）」，**不是 FAIL**。照常輸出 `ROB_layer12_vs_ENS_layer12`，用同一組三態門檻，但標「參考」。
- **為什麼門檻是 0.30**：PASS 需要 Δ ≥ 0.20 且 CI 下界 ≥ 0.10。ROB 還要付容量稅（乾淨正例 TPR 會小降），所以餘量低於 0.30 時，PASS 在結構上幾乎不可能，FAIL 也就沒有資訊。這個門檻只看基線，不看 ROB，事先凍結就不構成擇優。

### M2. 偵測器移出第 1 週裁定，只當探索性回報

PREREG 自己留了這個選項（第 4 節「若主對話認為不屬第 1 週…」），但必須在跑之前決定。看到結果才決定，就是事後擇優。

偵測器的設計有一個根本的混淆：

- 訓練標籤是「提示裡有 T0／T1 監測文字＝1，乾淨或誘餌＝0」；
- 測試正例是 T2／T3（也在提到監測），負例完全沒有監測字樣（`detector_stage` 的 `negrows` 只放乾淨、誘餌、負例乾淨）；
- 所以一個只認「提示提到被監測」文字的分類器，AUROC 就會接近 1。偵測器的 PASS 因此無法解讀，正是 49 號報告引用的 LessWrong 失敗模式。

真正的控制組要用 base model 跑同樣的觸發輸入，但 base model 是 gated。這屬於第 2 週的範圍。

修法：`aggregate.gate4` 的 `overall` 只取監測器狀態，偵測器照常輸出並標「探索」。

---

## 3. 已修（小、明確、不動門檻；已記入 PREREG「修訂紀錄」2026-11-14）

1. **`extract_acts.py --bench` 的總時數估計錯誤。** 原本 `est_total_gpu_h` 用的是 bench 本身的列數（limit 100 → 約 750 列），不是全量 36,418 列，低估約 50 倍。README 正是叫人看這個數字來決定要不要重排。已改用全量列數，並輸出 `n_full_rows`。CPU tiny 實測輸出 `n_full_rows: 36418`，與手算的 18,000＋2,000＋15,000＋488＋930 一致。
2. **bench 的抽樣順序有偏。** 原本依長度排序，最短的列先跑，會高估吞吐。bench 模式改成固定 seed 的隨機順序；正式抽取的順序不變。
3. **`logs/` 目錄不存在。** 三個 sbatch 都用 `--output=logs/%x-%j.out`，目錄不在時 Slurm 開不了輸出檔，job 會直接失敗。已建 `logs/.gitkeep`；.gitignore 已排除 `*.out`。
4. **`analyze.py` 讀快取時檢查非有限值**（fp16 溢位成 inf），有就停，算閘門 1 失敗。
5. **`analyze.py` 新增兩個只回報的欄位**，不進裁定：`per_layer_ENS_tpr_clean`、`delta_primary.ROB_layer12_vs_ENS_layer12`。M1 需要它們；aggregate 的次要表會自動列出。
6. **`slurm/env.sh`**：
   - XDG、Triton、Inductor、Matplotlib 的快取都導到 `$TMPDIR`（`/work/.../icml/tmp`），並 `mkdir -p`；
   - 每個 job 開頭把 job id、host、git hash、config md5 印進 log（對應通用規則「原始數字要能追溯」）。
7. **README 與 PREREG 的 checkpoint 狀態**改成：已下載，snapshot `c4aeb57502a8bed6a62dac57819eec27fd5e3c99`，約 18 GB，CPU 驗證了 config 與 464 個 tensor 的 header。額度敘述也改成約剩 160 GB。

驗證（CPU、nice、輸出放在 `/work/.../icml/tmp/cham-review`，驗完已刪；沒碰 GPU，沒寫 /tmp）：

- tiny bench、tiny 抽取、對舊 smoke 快取跑完整 `analyze.py all` 與 `aggregate.py`，都能跑完，新欄位有輸出。
- 另外用假輸入走過 aggregate 的閘門 4，四種情境都照 config 機械判定：
  - Δ=0.15，CI [0.07, 0.23] → 灰區；
  - Δ=0.05 → FAIL；
  - Δ=0.12，上界 0.19 → FAIL；
  - CI 半寬 0.09 → 閘門 3 擋下。

---

## 4. 建議（會改判準或設計，不必修，但最好在送出前一併決定）

- **S1. 灰區加碼沒有效。**
  - 現行加碼是 seed 從 5 增到 20，再併入 deception。但 Δ 的 CI 來自測試資料的 bootstrap，不是 seed 變異（閘門 3 已經要求 seed SD ≤ 0.03），加 seed 幾乎縮不了 CI。
  - deception 只有 62 個負例，1% FPR 的門檻就等於負例最大值，併進來只會增加雜訊。
  - 建議改成：抽取 harm_test 剩下的樣本。manifest 顯示可用 1,421 正／1,625 負，目前只用了 1,000／1,000。加進來 CI 約縮 15–20%，GPU 成本約 0.3 小時。
- **S2. 閘門 2 的 deception 兩項改成「只回報、不擋」。** n=62 時這兩項的判定接近擲硬幣；它們不在主指標裡，卻可能把整個實驗擋成「未能測試」。harm 的四項就足以驗證 organism。
- **S3. PASS 的歸因。** 要宣稱「好處來自逃避子空間」，需要 `ROB_vs_RND` 的 95% CI 下界 > 0。否則只能說「擴增有幫助」。可以列為 PASS 的共同條件，或只寫進解讀規則。
- **S4. 觸發下的 FPR。** `harm_test_neg` 已經抽了 `T0:harmful` 與 `D:harmful`，但 analyze 沒用到。建議補一個次要指標：觸發後負例的 FPR（門檻仍用乾淨負例），檢查 ROB 是不是靠把所有觸發輸入都判成陽性來拉高 TPR。
- **S5. 省記憶體（選配，未套用，因為沒在 GPU 上測過）。** `model(...)` 加上 `logits_to_keep=1`，可以省掉每批約 6 GB 的 256k 詞表 logits。H200 有 141 GB，不加也跑得動。
- **S6. 範圍說明。** 49 號報告第 1 週的「Fisher 子空間探測器」沒有實作，原停損也不需要它；PREREG 寫明「第 1 週不做」即可。
- **S7. PREREG §2 補一句**：作者的 8 層結果來自「對 8 層訓練逃避的 organism」（ENS8 的 harmful TPR 0.079），本實驗的 ENS8 對應的是不同設定。這樣日後讀結果的人不會拿兩邊的數字直接比。

---

## 5. 其他檢查清單

| 項目 | 狀態 |
|---|---|
| 唯一主指標 | ✔ Δ8（harm、T0 7 詞平均、1% FPR）。次要指標明列「不得改判」。 |
| 閘門順序：正確性 → 重現 → 雜訊 → 假設 | ✔ aggregate 依序判，前三關沒過記「未能測試」。需插入 M1 的 2b。 |
| 擇一條件的多重比較校正 | ✔ 沒有「m 個擇一」：主指標唯一，與偵測器是「兩者都要過」（AND），不需校正。M2 套用後只剩監測器。偵測器從 T2T3 退回 T0 是依樣本數決定，不看結果，可以接受。 |
| 門檻高於雜訊 | ✔ PASS 要 Δ ≥ 0.20 且下界 ≥ 0.10；閘門 3 要求 CI 半寬 ≤ 0.08、seed SD ≤ 0.03、洗牌標籤 TPR ≤ 0.05。 |
| aggregate 機械照 config 判 | ✔ 門檻全部讀自 `config.thresholds`，用假輸入驗證過。小瑕疵：閘門 2 的「相對逃避率」在乾淨 TPR=0 時會算成 1 而通過，但同一關的乾淨 TPR ≥ 0.55 會擋下，不影響結果。 |
| sbatch：h200q、gres=gpu:1、時間上限 | ✔ 25 分鐘／3 小時／2 小時。 |
| TMPDIR 與快取在 `/work/.../icml/tmp`、不寫 /tmp | ✔（已補 XDG／Triton／Inductor）。HF_HUB_OFFLINE=1，權重已在 HF_HOME。 |
| `logs/` 目錄 | ✔ 已建（原本缺，job 會起不來）。 |
| 激活存檔大小 | ✔ 約 2.1 GB，加 bench 可忽略，遠低於約 160 GB 的剩餘額度。分析時 RAM 約 2.1 GB fp16 加上暫存，mem=96G 足夠。 |
| 資料洩漏 | ✔ ROB 的子空間與擴增只用 benign E 概念的位移，不碰 harm、deception、H_b 的觸發資料；ENS 與 ROB 的層、資料、超參數相同。 |
| 原始數字與追溯 | ✔ 每個 seed 的數字都在 monitors.json；job id、git hash、config md5 已寫進 log。送 slurm 前仍需 commit（依規則由主對話做）。 |
| 判讀流程 | 機械輸出 → Opus 對照 PREREG → 使用者定案；記憶中的規則是 Fable 判讀結果，兩者不衝突。 |
