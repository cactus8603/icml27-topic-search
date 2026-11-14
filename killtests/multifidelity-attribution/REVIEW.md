# 送出前審查：多保真度資料歸因停損實驗

審查者：Opus。日期：2026-11-14。
依據：回報 56 第 4 節與「通用規則」；題目背景見回報 45。
範圍：PREREG.md、configs/criteria.json、README.md，以及 scripts/ 下所有 .py 與 .sbatch。
執行方式：沒有送 sbatch、沒有用 GPU、沒有 commit、沒有寫 /tmp。CPU 測試一律用 nice，TMPDIR 設在 `icml/tmp`。

## 結論：修完可送出

- 判準與程式整體正確。
- 審查中找到的程式錯誤都已修好（見 §2），而且沒有動到任何門檻數值或判讀邏輯。
- 必修三項（§1）都是**文件澄清加 commit**，不需要改程式，大約半小時可以完成。完成後可以從 `s0_bench` 開始送出。

---

## 1. 必修（送 slurm 前完成；會動到判準文字，所以審查者沒有自己改）

**M1. 先 commit 再送出。**
- 通用規則要求 PREREG、判準檔與判讀腳本在送 slurm 前 commit。目前全部都是 untracked。
- 本次修改後，`aggregate.py` 會把 `git_head`、`git_dirty` 與 `criteria_sha256` 寫進 verdict。若 commit 後 `git_dirty` 仍為 true，就代表判讀時用的不是凍結版。

**M2. 灰區 (a) 與跨種子條件的先後關係要寫清楚。**
- 衝突在哪裡：
  - PREREG §6 的 FAIL 寫「其餘情形（含……跨種子輸給便宜方法）」。
  - 但灰區 (a)（增益 ≥ 0.08、CI 下界 ≤ 0.03）沒有提到跨種子。
  - 程式目前在 (a) 成立時一律輸出 GREY，不看跨種子結果。
- 為什麼要改：如果跨種子已經輸了，加碼的 200 個同種子重訓救不回跨種子條件，這次加碼就白花了。
- 建議寫法：「跨種子差 < 0 時直接 FAIL，優先於灰區 (a)/(b)」。
- 由使用者決定寫法；決定後 `aggregate.py` 要跟著改一行。

**M3. 加碼之後若仍落在灰區，結論是什麼，要寫清楚。**
- 現況：PREREG 只寫「以相同規則重判一次，之後不得再加碼」，沒有說重判後仍是 GREY 時怎麼算。`aggregate.py --M 400` 也可能再次輸出 GREY。
- 建議：在 PREREG 加一句「加碼後仍為 GREY 記為 FAIL」，並讓 `aggregate.py` 在 `--M 400` 時把 GREY 映射成 FAIL。

> M2 與 M3 只影響階段 1。但凍結規則要求送任何工作之前就把全文定稿，所以列為必修。

---

## 2. 已修項目（小而明確的錯誤，不動門檻；PREREG「修改紀錄」已記 2026-11-14）

| # | 檔案 | 問題 | 修法與驗證 |
|---|---|---|---|
| F1 | aggregate.py | **NaN 會導致錯誤的 CONTINUE。** 若某便宜方法的分數含 NaN，它的 LDS 是 NaN；`max(..., key=)` 可能把它選成「最佳」。接著 α、κ_res 與增益都變成 NaN，三個 STOP 條件的比較都為假，結果輸出 CONTINUE。 | 所有判讀量都先檢查是否為有限值，否則輸出 `INVALID(程式錯誤，非題目結論)` 並 exit 4，與「題目不成立」分開記錄。已在 smoke 副本中注入 NaN 驗證。 |
| F2 | aggregate.py | **Y 的區塊會被靜默錯接。** 程式用 glob 抓 `Y_*_*.npy` 後依起點排序接起來。如果目錄裡多了抽查檔（例如 README 建議的 `--qend 2` 產生的 `Y_0_2.npy`），欄位會錯位，而且沒有任何報錯。 | 只接受從 0 起、首尾相接、互不重疊的區塊，並檢查欄數；否則 exit 2。已驗證。 |
| F3 | aggregate.py | **便宜方法缺檔時，會在子集合上選「最佳」。** PREREG 固定 7 個變體。 | 缺任何一個就不判讀（exit 2）。已驗證。 |
| F4 | aggregate.py | **閘門直接信任子腳本寫的 `pass` 旗標。** 匯出保真度的門檻 5e-3 寫死在 cheap_torch.py，不在 criteria.json。 | criteria.json 加入 `gates.export_fidelity_max_abs_logit_diff: 5e-3`（數值不變）。MLP 與保真度兩個閘門改由 aggregate 依 criteria.json 重新判定。 |
| F5 | aggregate.py | **缺跨種子重訓時，可靠度路線輸出 FAIL。** 主路線在同樣情況下輸出「未完成」，兩者不一致。 | 改為輸出「未完成」。另外，階段 1 判讀必須先有 `verdict_stage0` 為 CONTINUE。 |
| F6 | cheap_torch.py、jax_common.py、common.sh | **torch 看不到 CUDA 時會靜默改用 CPU。** 這會在 3 小時的 GPU 配額裡跑 CPU。JAX 的訓練與重訓也沒有檢查是否真的在 GPU 上。 | `common.sh` 設定 `REQUIRE_GPU=1`。JAX 的 `setup()` 與 cheap_torch 都會印出裝置，看不到 GPU 就中止。已在 CPU 上確認會中止。 |
| F7 | exact_y.py `--bench` | **吞吐外推會系統性高估。** 舊版把 JIT 編譯和 49 步段重算都攤進 20 個反向步，可能誤觸「超過推估 2 倍就重排」的規則。 | 先暖機，再用 t(21 步) − t(1 步) 算出純反向步的耗時；段重算成本另外加上。 |
| F8 | s0_bench.sbatch | **bench 沒有檢查 torch cu124 在 H200 上能否運作，也沒檢查 42 GB 的 TRAK 投影矩陣放不放得下。** | 新增 torch 檢查：印出 CUDA 狀態、配置 10.3M×2048 的 bf16 矩陣，並做一批投影。 |
| F9 | check_qchunk.py | **README 寫的 `--qend 2` 用法一定失敗。** 程式去找 `Y_0_2.npy`，但這個檔案不存在。 | 改為找出涵蓋 [qstart, qend) 的已存區塊，再從中切片。smoke 測試：[0,2) 與 [3,5) 都通過，相對誤差約 7e-8。 |
| F10 | 新增 check_resnet_fd.py（非判準） | **gate_mlp 用的 MLP 沒有 BN 緩衝，也沒有資料增強。** ResNet 上 BN running stats 與增強 key 的反向路徑沒有被已知答案驗證過。 | 寬度 4 的 ResNet9、64 筆、float64，與中央有限差分比對：**最大相對誤差 1.8e-8**。BN、增強與 momentum 的反向都正確。 |
| F11 | aggregate.py | 沒有記錄 git hash（通用規則要求）。 | 新增 `git_head`、`git_dirty`、`criteria_sha256`。階段 0 另外輸出**僅供描述、不參與判讀**的增益配對 bootstrap CI，以及折半相關。 |

修完後重跑了 smoke 的 `aggregate --stage 0`，流程正常。原始檔案備份在 scratchpad，不在 repo 內。

---

## 3. 逐項審查

### 3.1 判準本身

**階段 0 的兩個停點：正確。**
- PREREG §5 與 criteria.json 一致：
  - 最佳便宜方法 ≥ 0.8 → STOP
  - κ_res/κ < 0.5 → STOP
  - gain < 0.04 → STOP
- aggregate 依序判斷，第一個成立的條件即為結論。
- gain 定義為 LDS(混合) − max(LDS(Spell), LDS(最佳便宜))，等價於「兩個差取較小者」，與階段 1 的定義一致。

**是否固定 B = 0.2k，有沒有無限擇優的空間：已封住。**
- **B**：只從 `B_frac` 計算（k=50 → 10，k=200 → 40）。
- **U**：固定為 G_V 的前 B 個特徵向量。隨機 U 只當次要報告。
- **α**：固定為量測子空間上的純量最小平方，不收縮。
- **便宜方法**：7 個變體固定。階段 1 沿用階段 0 選出的方法，程式也強制如此。
- **還剩的自由度**：只有 7 選 1 的「最佳便宜」。這個選擇同時抬高了比較基線，方向是保守的；PREREG 已註明。
- **另一個設計取捨**：純量 α 與回報 45 的逐查詢 α_q 不同。純量對混合法不利（保守），但 FAIL 的解讀要因此打折，見建議 S3。

**配對 bootstrap 的單位：正確。**
- 單位是查詢；`hl − sl` 與 `hl − cl[best]` 在同一個查詢上配對。
- 重抽 10,000 次，seed 0，取雙尾 95% 百分位數 CI。
- 限制：U 與 α 是用全部 k 個查詢估出來的，重抽時沒有重新估計，所以 CI 只描述「這批查詢」，可能偏窄。這是回報 56 指定的單位，不算錯誤，見建議 S4。

**門檻與雜訊相比：合理。**
- 階段 0（k=50）：逐查詢 LDS 差的標準差若約 0.1，標準誤約 0.014，門檻 0.04 約等於 3 個標準誤。真實增益若是 0.06，被誤判為 STOP 的機率約一成。對停損點來說可以接受。
- 本次加入的描述性 CI，可以讓判讀者看出結果是否剛好落在門檻邊緣。
- 階段 1（k=200）：0.08 加上「CI 下界 > 0.03」，大約是 4 到 5 個標準誤，相當穩。
- 雜訊底線閘門（同種子無移除的重複）只量 GPU 的非確定性，很可能接近 0 而輕易通過。它和「同軌跡 LDS」的主指標一致，但不反映逐查詢 LDS 真值本身的雜訊。後者由折半信度處理：階段 1 用於判讀，階段 0 現在也會描述性輸出。

**重現失敗與程式錯誤，是否和「題目不成立」分開：是（加上 F1 之後更完整）。**
- 前三關失敗 → `UNTESTABLE`，exit 3
- 資料不全 → exit 2，不輸出 verdict
- NaN/inf → `INVALID`，exit 4
- 三者都不會產生 STOP、PASS 或 FAIL。

**三種結果能否解讀：可以，除了 M2 與 M3 兩處要補。**

| 結果 | 怎麼解讀 |
|---|---|
| STOP（≥ 0.8） | CIFAR 不是混合法的戰場。 |
| STOP（κ_res/κ） | 殘差沒有可以低秩捕捉的結構。 |
| STOP（gain） | 有結構，但 B=0.2k 時增益不足。 |
| PASS | 混合法在 CIFAR 上同時贏過簡化版 Spell 和最佳便宜方法 ≥ 0.08，跨種子也不輸。這已足以寫成方法論文的主結果（正面貢獻），可再加上「僅可靠度路線」作為第二組件。 |

- PASS 的注意事項：
  - Spell 是簡化重現，所以「贏 Spell」的說服力不如「贏便宜方法」。
  - 論文最後仍需要 GPT-2 或 UGN 的對照（回報 45 的原判準 2）。
  - PASS 代表「值得繼續」，不代表論文已經完整。

### 3.2 aggregate.py 的邏輯

- **門檻**：數值全部取自 criteria.json，判讀字串裡的 0.8 等數字只是標籤。F4 之後，保真度門檻也不再寫死。
- **邊界條件**：
  - `≥ 0.8` → STOP
  - `< 0.5` → STOP
  - `< 0.04` → STOP
  - 階段 1：`≥ 0.08` 且 `> 0.03` → PASS；`[0.04, 0.08)` → 可靠度路線；可靠度要求 `≥ 0.7`、`≥ 0.6`、並嚴格大於 max|平凡預測量|；灰區為 `[0.45, 0.6)`
  - 以上都與 PREREG 的文字一致。
- **NaN 處理**：見 F1。另外，LDS 在常數列時的分母有 `+1e-30`，結果是 0 而不是 NaN，屬於保守方向。
- **小提醒**：`boot_ci` 用 `nanmean`，NaN 查詢會被略過。因為 F1 已攔下所有判讀量，實際上不會發生。

### 3.3 程式正確性

**混合估計的三個恆等式：通過。**
- gate_hybrid 檢查：α=0 時等於 Spell；B=k 時等於 Y（α 取 None、0、3 都試）；C=Y 時輸出 Y 且 α̂=1；投影冪等；α̂ 為最小平方解。
- `hybrid.py` 的公式和 PREREG 一字不差：Ŷ = YUUᵀ + α(C − CUUᵀ)。κ 與 κ_res 的定義也一致。

**LDS 的計算：正確。**
- 符號：預測值 = −masks@S，對 j 取 Spearman，再對查詢平均。
- 各便宜方法的符號約定都正確：
  - grad_dot 與 tracin 是 −g_i·g_q
  - TRAK 是 −τ
  - kronfluence 是 −score（kronfluence 正值代表有幫助）
- smoke 測試中各方法的 LDS 都是正值，符號正確。

**MAGIC 精確 Y：合理，並經已知答案驗證。**
- 實作：
  - 整個訓練步（含 BN 緩衝更新）一次做 `jax.vjp`。
  - 狀態的餘切向量是 (∂L/∂params, 0, ∂L/∂bufs)，因為查詢損失在 eval 模式下依賴 running stats。
  - 段內狀態由檢查點重算，`fold_in(key0, t)` 與 `lr(t)` 都和前向一致。
  - `vmap(vjp_fn)` 只做一次 primal 重算。
- 驗證：
  - MLP 有限差分 4.7e-10，留一重訓 Pearson 0.998
  - ResNet9 有限差分 1.8e-8（F10）
  - qchunk=1 與 vmap 版本相差約 7e-8
- 尚未驗證：GPU 預設精度（可能是 TF32）下的準確度，只能靠重現閘門（LDS ≥ 0.88）間接把關。

**sbatch 參數：符合要求。**
- 全部是 `--partition=h200q`、`--gres=gpu:1`。
- 時間上限：bench 40 分、train 1 小時、retrain、crossseed 與 addon 4 小時、exact 3 小時、cheap 3 小時。
- `common.sh` 把 TMPDIR、PIP、UV、XDG 快取都設到 `/work/ChihChungHsu/cactus8603/icml/tmp`，並 `cd` 進去。程式沒有任何地方寫 /tmp。
- 小瑕疵：非 array 工作的 log 檔名 `%a` 會變成 4294967294，只影響外觀。

**輸出位置與磁碟用量：沒問題。**
- 群組額度：1377G / 1536G，剩約 159 GB。
- 大檔放在 `icml/runs/multifid/stage0`。repo 內只放 json 和小 npy，`.gitignore` 已排除 npz 等檔案。
- 階段 0 的推估用量：
  - 模型 npz：10 個 × 約 41 MB ≈ 0.4 GB
  - kronfluence factors：最大層 r2a 的 5761² 協方差加特徵向量約 0.27 GB，全部約 1 GB
  - 精確 Y（k=200）：40 MB
  - 重訓：每個模型 201 個浮點數
  - **合計 < 2 GB。**
- **TRAK 的 42 GB 投影矩陣只放在 GPU 記憶體，從不寫入磁碟**（`R` 在 `cheap_torch` 內，用完 `del` 釋放），不吃群組額度。
- 唯一的磁碟風險是可選的 `optional_bergson_gpt2.sbatch`：100 個 GPT-2 約 50 GB 以上，在額度只剩 159 GB 時，不要在階段 0 期間跑。

### 3.4 GPU 未驗證點與 s0_bench 的覆蓋情況

| 未驗證點 | 是否由 s0_bench 先檢查 |
|---|---|
| JAX cuda13 偵測 GPU（需驅動 ≥ 580） | 是（`REQUIRE_GPU=1` 加印出 `jax.devices()`）。F6 之後所有 JAX 工作都會檢查。 |
| 精確 Y 的 vmap 記憶體（qchunk=10，41 個檢查點留在 GPU） | 是。bench 用正式 cfg 做真正的反向步，OOM 會在 bench 階段就出現。 |
| 吞吐 / 單次訓練時間 | 是（F7 修正了外推高估的問題）。重訓 2 個模型也有實測。 |
| torch cu124 在 H200 上能否運作 | **原本沒有** → F8 已加入。 |
| TRAK 42 GB bf16 投影矩陣能否配置 | **原本沒有** → F8 已加入。 |
| EK-FAC（kronfluence）在寬度 80 時的耗時與記憶體 | **沒有**，見建議 S1。 |
| TF32 對 metagradient 準確度的影響 | 沒有，只能由重現閘門間接把關，見建議 S5。 |

---

## 4. 建議（不阻擋送出；有些會動判準，要改請記日期）

- **S1. s0_cheap 的 3 小時可能不夠。**
  - 這個工作要跑：TRAK 5 個模型的逐樣本梯度，EK-FAC 的 factor 擬合，以及 3 次 50000×200 的 pairwise score。kronfluence 在寬度 80 時的耗時完全沒測過。
  - 建議：時間上限放寬到 6 小時，或者把 ekfac 拆成第二個工作（`--methods ekfac`，前面的輸出會保留）。
- **S2. 跨種子的 seed 1 和 TRAK 集成的 seed 1 重複了。**
  - 若 trak5 是最佳便宜方法，它在跨種子評測上會帶點優勢。這是保守方向，但會混淆解讀。
  - 建議凍結前把跨種子改成 seed 5，或在報告中註明。
- **S3. 增加一個探索性報告：逐查詢 α_q（加收縮）。**
  - 回報 45 的設計是逐查詢的 α_q。PREREG 為了去掉自由度改用純量 α，這對混合法偏保守。
  - 若結果是 STOP(gain) 或 FAIL，報告時要說明「這是純量 α 加簡化版 Spell 的結果」。
  - α_q 版本只當探索性報告，不參與判讀。
- **S4. 增加一個次要穩健性檢查：查詢折半。**
  - 把 200 個查詢分成兩半，各自獨立估 U 與 α，再看增益是否一致。這能補足配對 bootstrap 沒有重估 U 與 α 的問題。
- **S5. 用 bench 加 check_qchunk 抽查 TF32 的影響。**
  - 若重現閘門不過，第一步先把 `matmul_precision` 改成 "highest" 重跑。這屬於閘門修正，不是調超參數，但要記入修改紀錄。
- **S6. 灰區 (a) 的唯一加碼方式效果有限。**
  - 灰區 (a) 的 CI 是對查詢重抽，加 200 個重訓只會降低逐查詢 LDS 的雜訊，CI 寬度大概縮不了多少。
  - 若要讓加碼真正有用，應改成加查詢（精確 Y 再加 100 個）。這會改動判準，只能在凍結前決定。
- **S7. 增加 GPU 上的匯出保真度實測。**
  - smoke 在 CPU 上的保真度是 1.5e-8，但 JAX 在 GPU 上的 TF32 對 torch（關閉 TF32）的差距沒有實測。
  - cheap 工作一開始就會檢查，不過會擋下也只浪費幾分鐘，影響小。

---

## 5. 審查時實際執行的 CPU 測試（全部 `nice -n 19`，TMPDIR 設在 `icml/tmp`）

- `exact_y.py --bench`（smoke）：新的計時流程跑通。
- `check_qchunk.py`：[0,2) 與 [3,5) 都 PASS，相對誤差 6.3e-8 與 7.0e-8。
- `REQUIRE_GPU=1` 下：`retrain.py` 與 `import cheap_torch` 在 CPU 上都正確中止。
- 在 smoke 副本（`icml/tmp/reviewtest`，測完已刪除）中：
  - 多放一個重疊的 Y 區塊 → exit 2
  - 注入 NaN → INVALID
  - 刪掉 trak5 → exit 2
  - 正常資料 → STOP（smoke 的數字沒有統計意義）
- `check_resnet_fd.py`：最大相對誤差 1.8e-8。
- `aggregate --stage 0 --smoke`：正常，verdict 含 git hash 與 criteria 的 sha256。
