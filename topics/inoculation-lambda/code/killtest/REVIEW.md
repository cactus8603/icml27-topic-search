# REVIEW：inoculation-lambda（Phase A，零訓練 a1–a4）

審查者：Opus｜日期：2026-11-14｜範圍：PREREG.md、configs/、scripts/compute_lambda.py、scripts/aggregate.py、slurm/a1–a4、README｜依據：reports 54、56 第 2 節與通用規則。
只做了閱讀與 CPU 上的極小檢查，沒有送出任何 job，也沒有 commit。

## 結論：**修改後可送**
唯一必修是先 commit。Phase A 只有閘門 1 一個判準，而且失敗只會記成「程式錯誤」，不會讓題目死，所以 Phase A 本身不會把結論推向 PASS 或 FAIL。

## 這次 Phase A 怎麼判（給使用者）
- **閘門 1（唯一有判準的項目）**：所有 (trait, 防禦) 點合併計算 Pearson(預測 Δτ = η·Λ, 實測中央差分 Δτ)，≥ 0.9 為 PASS。FAIL 的意思是程式有錯，給 3 個工作天修；修不好記為可行性停損，不記為「題目不成立」。a1 的 aggregate 遇 FAIL 會回傳 exit 1，a2、a3 的 afterok 相依會自動取消，不會白跑。
- **對照論文數字的 Spearman**：只描述（支持／中間／弱），**不停損，也不能拿來宣稱支持**。真正決定題目生死的是 Phase B 的主判準，它的訓練腳本與 13 個 arm 必須另寫、另審，並在送出前 commit。
- 不論 Phase A 的參考數字好壞，Phase B 都不得改動凍結的 Λ 版本（step 0、LoRA-B、200 筆、trait_heldout、ctx=MEAN）。改了就算探索性。

## 對照 56 號修正的落實情況
| 項目 | 狀態 |
|---|---|
| 判準 0 降為參考、不停損 | 已落實（PREREG §0、§7；aggregate 不依參考值改變 exit code） |
| 凍結單一 Λ 版本 | 已落實（§1）。投影維度寫成「無投影」，有交代理由，可以接受 |
| 13 個有防禦 arm、各任務內 Spearman、Fisher z、依任務分層 bootstrap | 程式已備（phase b）。我用合成資料檢查：有訊號時判 PASS（ρ=0.97），無訊號時判 FAIL（ρ=−0.05） |
| PASS／灰區（0.35–0.6，唯一加碼 1 seed + 6 arm）／FAIL < 0.35 | 已落實。「ρ ≥ 0.6 但附帶條件沒全過」歸為灰區，這是合理的詮釋 |
| 贏過基線：事先指定一個、配對 bootstrap、90% CI | 已落實（Grant g20 餘弦；實作為下界 > 0） |
| 匹配 ID 90% ±5pp，標記並列出 | 已落實（id_ok，另報排除後的 ρ） |
| 判準 2 只描述 | 已落實 |
| 閘門順序：有限差分 → 重現 → 雜訊 | 已落實。失敗記為「未能測試」，修復期 3 個工作天 |
| 成本上限、吞吐實測 | Phase A 上限 12 GPU 小時；Phase B 上限 90 |

## 實作者自己的詮釋
1. **τ 用 NLL，不用 judge**：這和 54 號的原始定義一致（τ_e 本來就是 log-prob），不算偏離。NLL 會混入風格與一般語言成分，這會讓 Λ 在 Phase B 更**難**通過，偏向 FAIL 而不是 PASS。可以接受，但請見建議 1。
2. **論文圖讀數（Fig.5 人工讀圖、Table 13 未獨立驗證）**：只用於參考相關，不影響任何判定。我沒有核對 PDF。
3. **只做零訓練**：符合「先做便宜的檢查」。要說清楚的是，Phase A 做完只能抓 bug，不能回答題目該不該停。
4. **用中央差分當閘門 1 的實測值**：這是對梯度正確性更乾淨的檢查，但也幾乎保證通過，只要 autograd 一致就會過。它抓不到「兩邊用同一個錯的損失函數」這類共模錯誤，例如遮罩錯或情境錯。請見建議 2。

## 必修（送出前）
1. **commit**：repo 目前沒有任何 commit，`a*.sbatch` 寫進 RUNLOG 的 git hash 會是 `none`，違反通用規則「送 slurm 前 commit」。要 commit 的包括 PREREG、criteria.yaml、reference.json、configs、scripts、slurm。

## 建議（非必修）
1. **對比式 τ（次要、探索性）**：Grant 的資料有對應的 normal 回應。可以另加一個 tau_set「normal_heldout」，報告 Λ_trait − Λ_normal，把一般語言成分扣掉。在 config 加一個 tau_set 即可，成本約 +30%。要在 Phase B 定案前決定，不得事後替換主版本。
2. **閘門 1 補兩個附帶輸出**（不改判準）：
   - 單邊差分 `dtau_true` 的 Pearson 與斜率。這比較接近 Fable 說的「1 步 Δτ」。
   - a2 的 bf16 Λ（seed 0）對 a1 的 fp32 Λ 的相關。閘門 1 只驗證了 fp32，主判準用的卻是 bf16。
3. **參考相關 G1 的保留比例**：改用 (expr_d − base)/(none − base)。optimistic 的 base 是 77.7、hallucinating 是 20.4，不扣掉 base 會扭曲比例。另外 pirate、spanish、allcaps 有多個 0 值並列。
4. **Phase B 寫腳本時一併處理**：
   - Λ 用 LoRA r=64，Phase B 訓練用 r=32，兩者應一致。
   - τ 的 3 個非誘發情境應與漏出評估的 6 族一致。
   - phase b 的 bootstrap 會因 NaN（常數輸入）丟掉樣本，造成 CI 有偏。應改成計數並報告，或在 spearman 遇常數時回傳 0。
   - phase b 每次判讀要跑好幾分鐘，可以接受。
5. `.gitignore` 排除了 `results/*/lambda_raw.json`。正式 run 的 raw 檔很小（約數百列），建議 commit 進 repo，以符合「原始數字全留」。
6. a4 只用 CPU 卻掛 1 GPU，只跑 20 分鐘，影響很小。若 h200q 允許不帶 gres，可以拿掉。

## 我直接修了什麼
- `slurm/a1_gate1_fd.sbatch`：`--time` 從 04:00:00 改為 06:00:00。推估 1.5–3 小時，原本的上限連 2 倍餘裕都沒有；一旦逾時，a2、a3 的 afterok 會被全部取消。已記入 PREREG 修訂紀錄。

## 環境與資源檢查（皆通過）
- 不再引用 `envs/neon`：使用 `envs/inoc`，已存在。
- 7 份 grant config、5 份 pg config 的資料檔、提示檔、steering 向量全部存在。
- Qwen2.5-7B 完整。**Qwen2.5-14B 現在已完整**（8 個分片都在，沒有 .incomplete），所以 a3 會跑 14B，README §6、§9 的「14B 未完成」已過時。
- partition h200q 沒有時間上限；記憶體推估 7B fp32 約 31GB、14B 5 個層段全參數約 75GB，都 < 141GB。
- 群組額度剩約 184GB。

## 送出順序（第一個該送的 job：a1）
```bash
cd /work/ChihChungHsu/cactus8603/icml27-topic-search/killtests/inoculation-lambda
# 先 commit（必修 1）
A1=$(sbatch --parsable slurm/a1_gate1_fd.sbatch)
A2=$(sbatch --parsable --dependency=afterok:$A1 slurm/a2_grant_lambda.sbatch)
A3=$(sbatch --parsable --dependency=afterok:$A1 slurm/a3_pg_lambda.sbatch)
sbatch --dependency=afterok:$A2:$A3 slurm/a4_aggregate.sbatch
```
a1 開跑後 10 分鐘看 log：用 sycophantic 每個防禦的耗時推算總時間，超過推估的 2 倍就停下重排。
