# Attention 寫入與 optimizer 壓力校準

[codex] 2026-11-15 台北時間。這是一次校準，決策是是否值得用新 seed 驗證這個特定現象。沿用 33M 模型、資料來源、強基線與單步損傷定義。首輪的機械結果與限制見 [Astra 審查](../../../astra-reviews/rebuttal-pilot-gate-optimizer-result.json) · [首輪結果報告](../../../reports/2026-11-15-rebuttal-pilot-results-codex.md#attention-gate--optimizer)。

## 配對與選擇規則

訓練、驗證及探針使用同一份 SHA-256 固定的 OpenWebText shard 與 tokenizer、同樣的文檔 hash 切分。第一輪的 final test 桶不在本輪評估中。校準 seeds 為 1865、2713、3907、5411，與首輪 dev seed 812、三個正式種子及下輪保留 seed 都不重疊。模型、優化器、資料順序、原始 hyperparameter、probe 定義沿用 10792d3，三個 base LR 鎖定首輪 dev 搜索選擇的 StableAdamW 0.003、Muon 0.004、AdamW + sandwich RMSNorm 0.0015。

每個 optimizer × seed 重建初始 checkpoint，正常 LR 先跑 100 步，保留完整模型和 optimizer state。從這同一狀態復原四個分叉，共用後續 400 步批次。正常參照以 cosine 從 1× 降到 0.1×。壓力臂各以 20 步把倍率升到事前固定的 **3×、4×、6×** 之一，之後維持不變。每 50 步固定用 dev 桶評估。三個壓力設定的探針在相同步數使用同一批資料，signal 和 outcome window 互斥。每 10 步測一筆，選擇規則只用 ramp 結束之後的步數。正常參照全程可用且無測試桶調參。每個倍率先走完所有預定 cell，然後機械選最低合格倍率。

1. **排程資格。** 一個倍率必須讓全部 3 臂 × 4 seeds 皆有對應配對正常參照，雙方訓練、評估、預定 stress probe 均已完成且有限。以每一 checkpoint 的 dev NLL 減同 cell 的正常參照，任一 cell 若末步差值 >0.5 nats，或連續三次評估差值 >0.5 nats，視為該倍率全局品質崩退。明確數值發散和已知 prefix 失敗也使該倍率失格。時間不足、缺檔或缺排程記錄則整網格記為「不可判」，不退回挑完的格子。
2. **現象門檻。** 在上述合格倍率，獨立 outcome batch 更新後的 NLL 減更新前 NLL ≥0.05 nats 算一筆損傷。至少 4 筆、跨至少 2 seeds 且至少 2 種強基線，即為可重現校準現象。在三個倍率中取合格且達門檻的最小值。每格都測完且無符合倍率，則此次校準不建議進入正式驗證。資料不全才判不可判。每個倍率、seed、optimizer 都報完成與損傷計數，不以成功的子集替代主要母體。
3. **訊號僅描述。** 使用第一輪的整 seed 留出 ridge logistic，比較相同的 QK max、attention 寫入 activation max、grad norm、更新 RMS、optimizer 臂和時間步，再加入固定輸入下的 attention 寫入變化 p95。兩個模型都只從留出以外的 seeds 擬合 z-score 和係數，固定 ridge 0.1。合格倍率上至少有 12 次損傷且至少 3 個留出 seed 可算 AUROC 時，才報每 seed 的配對 AUROC 增量及 seed-level 描述性區間。否則明示不足，列訓練與留出類別數。這個校準訊號不參與選倍率或 go/no-go。
4. **改善空間為同 checkpoint 的局部診斷。** 對每個實際損傷步，用相同的 pre/post checkpoint 與相同 outcome batch，固定其他權重，只比較最強 signal 方向所選那層的 output projection 更新。替代案是將該方向響應減半、同 Frobenius 更新量的均勻縮步、同修改量的正交循環位移方向，另量完全回退該層 output projection。選層和方向的一切計算都只看 signal batch，在原本軌跡上回復實際權重。報可用配對數、兩個 norm 誤差與平均有號 NLL 回收。整步完全回退可以消去已觀察到的正損傷，所以該步「回到更新前 NLL」的缺口恰為觀察到的損傷，對各替代案的回收分數截在 0 到該缺口。此上限只限制這個回收定義，排除額外改善與持續訓練回饋。局部介入與 rollback 都不是實際優化器或部署上限，不作升級門檻。

## 獨立驗證計畫

本作業結束於校準。若機械判讀選出倍率，先由獨立審查核對選擇、有限步數與品質淘汰，再用新 commit 凍結該倍率與驗證。保留 seeds 6007、6833、7019、7841、8581、9157、10069、11113、12379、13309、14821、15809。正式驗證使用同一倍率、同三臂、同 guard 和 probe，至少 30 次損傷及至少 6 個正負類皆存在的留出 seed 才算可判。候選等權平均 AUROC 增量須至少 0.03 才支持人工壓力診斷；沒達事件覆蓋記不可判，不轉回其他倍率。自然 spike 與持續更新介入是否縮短達同 NLL 的時間，仍需獨立實驗。沒有看到正常排程、等品質的端到端收益前，部署改善空間維持未量。

## 執行

一張 H200，Slurm 上限兩小時，程式限時 6,500 秒（包含資料準備）。共 12 個配對 prefix ×（100 + 4×400）步，約 20,400 步、8,356 萬訓練 tokens，另加測量。凍結後可先以 `bash calibrate.sbatch FULL_COMMIT_SHA --check-only` 驗證工作區和兩輪依賴，再按 [協調規則](../../../../COORDINATION.md) 開作業。執行唯讀快照與 corpus 只寫共用 icml/tmp，結果寫父層 `results/round2-<job-id>/`。CPU 合成測試輸出可用父層已忽略的 `.smoke`。本配置沒有直接執行獨立驗證的入口。
