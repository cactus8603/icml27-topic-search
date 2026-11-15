# 獨立查詢的補空間對齊

Stage 0 的補空間餘弦 `ρ⊥` 落灰區。方向覆蓋 `26517` 量的是精確方向在新遮罩上的擬合能力。種子 6 `26535` 與跨軌跡四格 `26574` 量的是 LDS。本次固定原十方向與最強廉價 EK-FAC，直接量它們在全新 12 條查詢上的補空間餘弦。 [正式 Stage 0 規格](../../../killtests/multifidelity-attribution/PREREG.md)

## 固定輸入

- 使用 seed 0、原 CIFAR-10 SGD／BN／增強與 ResNet-9。校準為原查詢排列位置 `[0,50)`，新目標固定取 `[436,448)`，避開原 `[0,400)` 及方向覆蓋 `[400,436)`。26535 和 26574 的 12 欄都在原 `[0,400)`。新欄位按位置固定，兩區塊各六欄。所有資料、程式、舊權重與舊因子 SHA 必須通過。
- 原校準精確矩陣 `Y_c`、Gram `G_cc`、最強 C 的 `C_c` 已存在。新目標各算 `Y_t` 及 `C_t`。C 固定 EK-FAC `damping=1e-8`，以鎖版 Kronfluence 讀原 seed 0 factors。重算位置 0–3 的四條舊 C 作技術核對，逐欄相對 L2 最大允差 `2e-4`。輸出原 checkpoint 的 PyTorch query 輸出須對齊 JAX 高精度參考，沿用 `5e-3` logit 差上限。JAX 歷史校準 loss 與新目標精確重播對當趟 checkpoint 的標準化 RMSE 都須低於 `0.02`。

## 投影、對照與量測

- 令 `J_c`、`J_t` 為同一最終模型的全參數 query loss 梯度列，`G_cc=J_c J_cᵀ` 取前 10 個特徵向量 `U`、特徵值 `Λ`。只用校準 Gram 選方向。新座標 `D=Λ⁻¹ Uᵀ (J_c J_tᵀ)`。令 `Q=J_cᵀ U Λ⁻¹/²`，則 `QᵀQ=I`，且 `QQᵀJ_tᵀ=J_cᵀ U D`。在校準欄 `D=Uᵀ`，精確還原 Stage 0 的右乘 `UUᵀ`。正交性位於最終模型的參數梯度空間，映射成訓練樣本權重梯度後的殘差仍可在該輸出空間相關。
- 主量是每個新目標的 `cos(Y_t−Y_c U D, C_t−C_c U D)` 的平均，固定 10,000 次查詢重抽、種子 `2026111546`，中央 95% 區間，另列兩個六欄區塊。任何必需向量零範數、非有限或 eigengap 不明時中止。核對 `UᵀU=I` 及校準 `D=Uᵀ`。重新算出的校準 Gram 相對 Frobenius 差不得超過 `2e-4`。
- 負控制依原 seed `20261114` 同時置換 `C_c`、`C_t` 的訓練樣本列。它與線性投影可交換，故可直接對 C 殘差做同一置換。若 `|ρ_perm|>0.05` 記參考控制失敗。另報未投影 `cos(Y_t,C_t)`、同預算 Spell 項 `cos(Y_t,Y_cUD)`，以及兩個殘差對各自原向量的範數比例。此步只量幾何，未執行原 Stage 0 的置換 LDS 增益控制或新增遮罩重訓。
- 與原 `0.20` 點值及 `0.10` 區間參考比對，結果限於這個 held-out 投影延伸。正式 `ρ⊥`、Stage 0 現象／空間與 seed 5 的 `H_x` 按原規約。

## 執行與成本

依序跑 JAX 12 個新精確查詢、PyTorch 新 query Gram 及同 factors 的 EK-FAC、CPU 聚合。一張 H200，JAX 含預留約 126 GiB。原兩組六欄精確回播約 13 分鐘；本次預估總 15–25 分鐘，作業上限 45 分鐘。EK-FAC 讀既有 675 MiB factors，不複製、不重擬。新 Y、C 各約 2.4 MB；含庫自身的 pairwise 檔及收據估低於 50 MiB。CPU 合成、資產 SHA 與 NPY header 預檢使用 NumPy，2 CPU、8 GiB RAM 足夠。

```bash
stage=codex/experiments/multifidelity-complement
python=/work/ChihChungHsu/cactus8603/icml/envs/attrib-jax/bin/python
"$python" -IB "$stage/test_pilot.py"
"$python" -IB "$stage/pilot.py" preflight
bash "$stage/run.sbatch" FULL_FREEZE_COMMIT --check-only
```

主執行在 `icml/runs/multifid/complement/<job-id>/` 寫技術收據及唯一正式彙總 `summary.json`。獨立判讀使用凍結後完成的彙總。
