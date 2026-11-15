# 只移植少數 RL 推理決策的蒸餾｜設計卡

1. **量現象。** 固定題組、提示和 3,000 token 預算，找一對截斷率相差小於 2 點的模型。要求 RL 的 `pass@1` 高至少 5 點，而基礎模型的 `pass@64` 高至少 2 點。
2. **量訊號。** 在兩個模型偏好的下一個 token 不同時測少量替換。正式要求相對同引擎 RL，`pass@64` 增加至少 2 點且信賴下界大於零，同時守住 `pass@1` 保留比例。另用同一前綴、同一續寫模型的單 token 分叉測局部效果。 [正式訊號判準](code/killtest/PREREG.md) · [局部分叉](code/experiments/rl-fork-budget/PREREG.md)
3. **量空間。** 在相同 `pass@1` 下，比較 token 替換、回應混合與 WiSE-FT。另用已知答對與否後逐題選模型，估計固定候選內的事後上限。 [原取樣的上限診斷](code/experiments/rl-gate0-censoring/PREREG.md)

**下一個判定。** 先取得截斷率可比的新配對。用同樣算力看少量 token 替換能否帶來多次作答的增益，再決定是否訓練學生模型。 [正式決策條件](code/killtest/PREREG.md)

[目前結果與資源](VERDICT.md) · [執行方式](RUN.md) · [介紹卡](README.md)
