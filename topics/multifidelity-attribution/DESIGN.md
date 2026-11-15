# 用少量精確計算改善訓練資料歸因｜設計卡

1. **量現象。** 在 CIFAR-10 上，每次移除約 1% 訓練資料再重訓，記錄每張查詢圖的損失。LDS 衡量各種移除造成的損失排序有多接近實測。1 代表排序完全一致，接近 0 代表排序幾乎無關。 [正式量測](code/killtest/PREREG.md)
2. **量訊號。** 比較精確歸因和 EK-FAC 在尚未精確量到的方向是否一致。主要量是這兩者剩餘部分的餘弦，越接近 1 代表方向越一致。把 EK-FAC 對應的訓練例打亂，檢查這個對齊是否消失。 [訊號確認設計](code/experiments/multifidelity-signal-confirm/PREREG.md)
3. **量空間。** 在同一批重訓比較最佳低成本基線與完整精確參考，再換訓練初始化，檢查同一份預測還有多少用。 [跨初始化設計](code/experiments/multifidelity-transfer-lds/PREREG.md)

**下一個判定。** 若固定預算的混合法能在新移除批次穩定贏過兩個基線，同一已訓練模型上的資料歸因就有可做的切入點。跨初始化另有驗證門檻。 [跨初始化判準](code/killtest/PREREG.md)

[目前結果與資源](VERDICT.md) · [執行方式](RUN.md) · [介紹卡](README.md)
