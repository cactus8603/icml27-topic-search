# 多保真度歸因的第二個訓練 seed

固定採用未曾跑過的 seed 6，從相同的 CIFAR-10 規格重新初始化與訓練 ResNet-9。原 Stage 0 的 seed 0、集成 seed 1–4 與正式跨種子 seed 5 沿用原用途。這次要看另一條訓練軌跡上，精確 metagradient 與事前固定的強 EK-FAC 是否仍有可辨識的同軌跡 LDS 差。沒有幾何選擇、混合模型、超參數搜尋或重算舊結果。 [原預登記](../../../killtests/multifidelity-attribution/PREREG.md) · [Stage 0 對照摘要](../../../killtests/multifidelity-attribution/VERDICT_stage0.md)

完全重用 26448／26449 的 12 個查詢及其 100 個評估遮罩。查詢為原池位置 `[118,124)`、`[268,274)`，遮罩為 `j=100..199`，相同 mask RNG 與 1% 移除。凍結連接 CIFAR 程式、CIFAR 批次 SHA-256、查詢索引與遮罩 packed SHA-256。同一組 SGD、momentum、BN 及資料增強流程重訓，權重遮罩只改變損失權重，維持原 BN 定義。同 seed 額外兩次未移除訓練量測雜訊，作輔助診斷。 [12 查詢原協定](../../../killtests/multifidelity-attribution/exact_probe/PROTOCOL.md)

精確方法只訓練一次全資料 seed 6 的主模型，在相同狀態上兩次反向重播各六個查詢。檢查精確重播 loss 與同一 checkpoint 的標準化 RMSE 小於 0.02。EK-FAC 使用同一個最終 checkpoint 及相同 50,000 個訓練例估 factors。版號、FP32 與 JAX 高精度參照的匯出保真度沿用原程式。事前固定 `damping=1e-8` 的 Stage 0 最強方法，只算 12 個查詢對的分數。對 curvature 抽樣另外固定 torch seed，避免依賴先前方法的 RNG 消耗。

各方法對同一 100 組遮罩取 `−sum(歸因分數)`。逐查詢與真實重訓交叉熵算 Spearman，再對查詢取平均。列 exact、EK-FAC、兩者差與共用查詢的配對 95% bootstrap 區間。兩個查詢區段共享訓練軌跡、資料與遮罩，所以區間只是描述性。若任一必要欄恆定，嚴格狀態記未定，兩方法一律使用共同可評分欄；少於九欄只給覆蓋。結果只針對固定查詢與新 seed 的同軌跡比較。原正式跨種子 `H_x` 的核准前提、seed 5、遮罩和 50 查詢維持原預登記。

## 操作與資源

一張 H200，依序執行 JAX 軌跡／重訓、PyTorch EK-FAC 和 CPU 彙總。前一次兩個六查詢區塊各約 7:20，Stage 0 重訓 25 個遮罩約 12:30，原全套便宜方法 10:13。以同機型估計本輪約 65–80 GPU 分鐘，Slurm 上限兩小時。24 GB 版須先量既有六欄重播和 EK-FAC 峰值，這次排程使用 H200。重用原 CIFAR 與環境。新軌跡與模型梯度約 45 MB，EK-FAC factors 依舊版約 675 MiB，作業目錄預估 0.8 GiB，保留 2 GiB 磁碟餘裕。

以下預檢不載入模型、不讀歷史結果、不產生工作產物。由 repo 根目錄執行。

```bash
shared=/work/ChihChungHsu/cactus8603/icml
pilot=codex/experiments/multifidelity-second-seed
export TMPDIR="$shared/tmp" CIFAR_DIR="$shared/data/cifar10/cifar-10-batches-py"
"$shared/envs/attrib-jax/bin/python" -B "$pilot/test_pilot.py"
"$shared/envs/attrib-jax/bin/python" -B "$pilot/pilot.py" --stage check --source killtests/multifidelity-attribution
bash -n "$pilot/run.sbatch"
```

複核並凍結後，依共用規則檢查佇列和登記。作業從已凍結的 Git 版本擷取程式，另固定 Stage 0 與 Kronfluence 來源 commit。所有產物寫在 `icml/runs/multifid/second_seed/<job-id>/`，正式彙總為該目錄的 `summary.json`。

```bash
freeze=$(git rev-parse --verify HEAD)
sbatch "$pilot/run.sbatch" "$freeze"
```
