# 用少量精確計算改善訓練資料歸因｜目前結果

**目前現象與空間。** 50 個固定查詢、64 個新移除批次的同一訓練軌跡，EK-FAC 是 **0.15040 LDS**，完整精確參考是 **0.95259**，差 **0.80218**。換到另一個初始化，精確法的優勢縮到 **0.01878 LDS**，95% 區間 **−0.02617 至 +0.06533**，尚未確立可轉移。 [跨初始化量測](code/experiments/multifidelity-transfer-lds/PREREG.md#讀取紀錄)

**目前訊號。** 另一批 50 個新查詢使用既有 200 個移除批次，剩餘方向餘弦 **0.20609**，通過門檻和置換控制。當批混合法比最佳比較基線 Spell 高 **0.04074 LDS**，距完整精確仍差 **0.67642**。先前在另一批新移除資料上，混合法低於 EK-FAC。 [50 個查詢的確認](code/experiments/multifidelity-signal-confirm/PREREG.md#讀取紀錄) · [新移除批次試驗](code/experiments/multifidelity-heldout-lds/PREREG.md)

**正在驗證。** `26866` 已送出。它固定訊號確認的 50 個查詢，加上 64 個全新的移除批次，檢查同一套分數能否預測沒看過的重訓結果。 [新移除批次確認](code/experiments/multifidelity-new-mask-confirm/PREREG.md)

## 信心、潛力與資源

- **現有證據 3/5，條件式論文潛力 4/5。** 同一訓練軌跡的訊號與精確上限都有實測。最有價值的是證明少算精確方向仍能穩定改進重訓排序。
- **下一步資源估計。** `26866` 以 **1 張 H200** 跑 64 個新移除批次，估 **32–40 分鐘**，約 **0.53–0.67 GPU 小時**，另加 CPU 彙整，不含排隊。申請 **64 GiB RAM**，新增保留檔案估 **小於 1 MiB**。先前相同訓練形狀的重訓顯存約 **126.4 GiB**，含 JAX 預留。 [新移除批次資源](code/experiments/multifidelity-new-mask-confirm/PREREG.md)

## 深入調查

MAGIC 沿整段訓練追蹤資料對預測的影響，計算成本較高。SPELL 只量少數精確方向來節省成本，EK-FAC 則能快速給出近似分數。

我們要驗證的是，低成本分數能否補上 SPELL 沒量到的部分，在相同精確計算預算下，比單用 SPELL 或 EK-FAC 更準確地預測移除資料後的重訓結果。 [資料歸因文獻深查](../../reports/2026-11-14-45-deepcheck-multifidelity-attribution.md)

[回題目首頁](README.md)
