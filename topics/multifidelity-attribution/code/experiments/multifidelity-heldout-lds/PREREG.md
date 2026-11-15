# 新遮罩的資料歸因排序

本診斷檢查補空間的幾何對齊能否改善固定 12 個查詢的實際移除重訓排序。使用和補空間試驗相同的查詢及預測輸入，另建獨立的遮罩重訓真值。原 Stage 0 與保留 seed 5 的 H_x 按原登記。 [補空間試驗](../multifidelity-complement/PREREG.md) · [正式 Stage 0 預登記](../../killtest/PREREG.md)

## 固定輸入

- 封存原訓練和純量擬合程式於 `599469e3faf5dec7aae854cc929c6d265bdd706f`，交叉 Gram 投影於 `1432b5dffa9527677f3cce697f138bbe437c40cb`。沿用 CIFAR-10 全訓練集、ResNet-9 W=80、seed 0 原順序、SGD＋momentum、原學習率與權重衰減、20 epochs、batch 500、原增強與 BN、原 JAX 精度。以權重歸零實現樣本移除。
- 測試排列位置從零起算，固定取 `[436,448)`。輸入沿用舊前 50 欄 Gram／精確 Y／固定 EK-FAC 及 `26643` 的新 12 欄精確 Y／固定 EK-FAC／交叉 Gram。所有檔案、程式、題目順序、兩份產物及彙總收據的 SHA-256 記於 `inputs.json`，另核對產物與收據互相指認。全精確 Y 的新 12 欄只供同一遮罩上的參考方法。
- 同一個舊 RNG、`mask_seed=1234`、每列以 1% 機率移除，固定獨立 test 遮罩 j=328–391。Stage 0／舊新查詢沿用 j=0–199，方向覆蓋用 j=200–327。CPU 預檢核對 64 個新遮罩的內容互異、和全部 328 個歷史遮罩互斥，並核對事先固定的 packed SHA-256。
- 正式新遮罩之前，重播固定歷史 j=100／101，保持原 400 查詢前向形狀，僅比對原 loss。各遮罩的 RMSE 除以 `max(std(reference), 0.1)` 必須小於 0.02，沿用原精確查詢的技術對照界線。新 64 個遮罩在同一個 448 查詢順序前向，只保留最後 12 欄交叉熵。歷史遮罩只作技術核對。 [原重訓程式](../../killtest/scripts/retrain.py) · [精確重播技術允差](../../killtest/exact_probe/PROTOCOL.md)

## 評分與判讀

- 固定從舊 Gram 選十方向 U。投影座標 D 沿用補空間試驗的同一計算。Spell 是 `(Y_c U)D`，固定 EK-FAC 為 `C_t`。沿用原量測子空間純量 `α=⟨Y_c U,C_c U⟩/⟨C_c U,C_c U⟩`，Hybrid 為 `Spell+α[C_t−(C_c U)D]`。兩個預測法的固定方向及係數都從舊校準欄取得。
- 負對照以 `default_rng(20261114)` 同列置換 `C_c`、`C_t`，如原 Stage 0 重算置換後的 α，保留同一 Spell。比較獨立遮罩上的置換增益 `LDS(置換 Hybrid)−max(LDS(Spell), LDS(置換 EK-FAC))`，`≤0.01` 為診斷參考。置換補空間餘弦由已封存的幾何對照提供。
- 每個遮罩的預測為移除列分數總和的負值。沿用逐查詢、跨遮罩的 Spearman，並列各方法 12 查詢平均 LDS。其分數與全精確 Y 同遮罩比較。固定以整組平均 LDS 計較強基線 `max(Spell, EK-FAC)`，不逐題挑方法。
- Hybrid 對兩條固定基線及對較強基線的差、精確法距較強基線的差與置換增益，均報 10,000 次成對查詢重抽的描述性 95% 區間。沿用 seed 0，逐次取兩種基線均值的最大值。報兩個原六查詢區塊均值。Hybrid 對較強基線的差值 `+0.04` 是實用性診斷參考。常數損失或任一必要預測為常數時，記錄不可評查詢及共同可評數量，整組比較留空。結論限單一 seed 及這組查詢和新遮罩。

[codex] 2026-11-15 12:55（台北）。Astra 獨立核對固定投影、置換後重新擬合、原 LDS 方向及共同基線。這 12 個查詢已用於幾何對照，留出樣本為 64 個遮罩。查詢重抽區間描述此面板，未估遮罩或訓練種子的不確定性。

## 執行及硬體

一張 H200，64 GiB 主記憶體、1 小時 Slurm 上限。舊重訓同設定穩態約 29 秒／遮罩，顯存約 126.4 GiB 含 JAX 90% 預留。新 64 個遮罩加二個歷史技術重播、前向與 JIT，估 **32–40 GPU 分鐘**。24 GB 卡需要同配置的未預留顯存實測。CPU 預檢和最後排名可在無 GPU 的環境執行。

結果在 `icml/runs/multifid/heldout-lds/<job-id>/`。只保留 64×12 的 loss、遮罩和查詢 ID、技術與結果收據，持久化估低於 1 MiB。不輸出模型權重，重訓日誌僅有遮罩 ID、耗時及歷史技術誤差。

~~~bash
stage=codex/experiments/multifidelity-heldout-lds
freeze=FULL_FREEZE_COMMIT
bash "$stage/run.sbatch" "$freeze" --check-only
# 完成指定審查及佇列核對後執行
sbatch --parsable "$stage/run.sbatch" "$freeze"
~~~

## 讀取紀錄

- [codex] 2026-11-15 13:29（台北，13:29:45）。已讀 `icml/runs/multifid/heldout-lds/26654/summary.json`，凍結 `c7aa443`。 [研究進度報告](../../../../../codex/reports/2026-11-15-topic-status-codex.md#多保真度資料歸因)
- [codex] 2026-11-15 13:29（台北，13:29:50）。Astra 獨立讀取同一份凍結彙總。
