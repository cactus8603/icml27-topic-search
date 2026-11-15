# 用少量精確計算改善訓練資料歸因

## 介紹卡

**要解的問題。** 我們想知道訓練資料中的哪些例子，對模型在某張測試圖上的判斷幫助最大。每次都移除一批資料重訓，花費很高。便宜的估計能快速打分，但排序可能和重訓結果差很多。

**例子。** 模型把一張卡車圖看成汽車。若想比較「移除第 A 批」和「移除第 B 批」對這張圖的影響，完整實驗要分別重訓。這題想在只做少量精確計算時，就能常常排對哪一批影響較大。

**做法。** 先用少量精確歸因量出幾個重要的查詢組合，再讓低成本方法 EK-FAC 補上未量到的部分。此處的精確歸因是沿原本的訓練步驟，反算資料權重小幅改變會如何影響測試損失。Spell 式基線只使用前面算過的精確部分。 [原方法與基線](../killtests/multifidelity-attribution/PREREG.md)

**成功的樣子。** 在沒用過的移除批次上，同樣精確計算預算的混合方法，能比 EK-FAC 和 Spell 更準確地排序真實重訓效果。

## 設計卡

1. **量現象。** 在 CIFAR-10 上，每次移除約 1% 訓練資料再重訓，記錄每張查詢圖的損失。LDS 衡量各種移除造成的損失排序有多接近實測。1 代表排序完全一致，接近 0 代表排序幾乎無關。 [正式量測](../killtests/multifidelity-attribution/PREREG.md)
2. **量訊號。** 比較精確歸因和 EK-FAC 在尚未精確量到的方向是否一致。主要量是這兩者剩餘部分的餘弦，越接近 1 代表方向越一致。把 EK-FAC 對應的訓練例打亂，檢查這個對齊是否消失。 [訊號確認設計](../codex/experiments/multifidelity-signal-confirm/PREREG.md)
3. **量空間。** 在同一批重訓比較最佳低成本基線與完整精確參考，再換訓練初始化，檢查同一份預測還有多少用。 [跨初始化設計](../codex/experiments/multifidelity-transfer-lds/PREREG.md)

**目前現象與空間。** 50 個固定查詢、64 個新移除批次的同一訓練軌跡，EK-FAC 是 **0.15040 LDS**，完整精確參考是 **0.95259**，差 **0.80218**。換到另一個初始化，精確法的優勢縮到 **0.01878 LDS**，95% 區間 **−0.02617 至 +0.06533**，尚未確立可轉移。 [跨初始化量測](../codex/experiments/multifidelity-transfer-lds/PREREG.md#讀取紀錄)

**目前訊號。** 另一批 50 個新查詢使用既有 200 個移除批次，剩餘方向餘弦 **0.20609**，通過門檻和置換控制。當批混合法比最佳比較基線 Spell 高 **0.04074 LDS**，距完整精確仍差 **0.67642**。先前在另一批新移除資料上，混合法低於 EK-FAC。 [50 個查詢的確認](../codex/experiments/multifidelity-signal-confirm/PREREG.md#讀取紀錄) · [新移除批次試驗](../codex/experiments/multifidelity-heldout-lds/PREREG.md)

**正在驗證。** `26866` 已送出。它固定訊號確認的 50 個查詢，加上 64 個全新的移除批次，檢查同一套分數能否預測沒看過的重訓結果。 [新移除批次確認](../codex/experiments/multifidelity-new-mask-confirm/PREREG.md)

**下一個判定。** 若固定預算的混合法能在新移除批次穩定贏過兩個基線，同一已訓練模型上的資料歸因就有可做的切入點。跨初始化另有驗證門檻。 [跨初始化判準](../killtests/multifidelity-attribution/PREREG.md)

### 信心、潛力與資源

- **現有證據 3/5，條件式論文潛力 4/5。** 同一訓練軌跡的訊號與精確上限都有實測。最有價值的是證明少算精確方向仍能穩定改進重訓排序。
- **下一步資源估計。** `26866` 以 **1 張 H200** 跑 64 個新移除批次，估 **32–40 分鐘**，約 **0.53–0.67 GPU 小時**，另加 CPU 彙整，不含排隊。申請 **64 GiB RAM**，新增保留檔案估 **小於 1 MiB**。先前相同訓練形狀的重訓顯存約 **126.4 GiB**，含 JAX 預留。 [新移除批次資源](../codex/experiments/multifidelity-new-mask-confirm/PREREG.md)

### 程式入口

[新遮罩驗證作業](../codex/experiments/multifidelity-new-mask-confirm/run.sbatch) · [作業設定與判準](../codex/experiments/multifidelity-new-mask-confirm/PREREG.md)

### 深入調查

MAGIC 沿整段訓練追蹤資料對預測的影響，計算成本較高。SPELL 只量少數精確方向來節省成本，EK-FAC 則能快速給出近似分數。

我們要驗證的是，低成本分數能否補上 SPELL 沒量到的部分，在相同精確計算預算下，比單用 SPELL 或 EK-FAC 更準確地預測移除資料後的重訓結果。 [資料歸因文獻深查](../reports/2026-11-14-45-deepcheck-multifidelity-attribution.md)
