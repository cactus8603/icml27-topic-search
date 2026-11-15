# 固定多保真度歸因的跨種子盲測

量測 seed 0 的精確歸因相對固定 EK-FAC 的排序優勢，有多少能轉移到另一條訓練軌跡。先前轉移比較採用 seed 0／6 和 12 個查詢。這次重用前 50 個查詢的固定預測，用同一批新遮罩分別量 seed 0 的同軌跡表現及 seed 7 的轉移表現。推論範圍限於這兩條軌跡、固定查詢以及 1% 遮罩。正式 Stage 0 與預留 seed 5 的 `H_x` 按原協定判讀。 [先前轉移試驗](../multifidelity-cross-seed/PROTOCOL.md) · [正式規格](../../../killtests/multifidelity-attribution/PREREG.md)

## 固定輸入與重訓

- 預測用已封存的 seed 0 精確矩陣五塊和同 seed 的 EK-FAC `damping=1e-8`。兩者固定取原 CIFAR-10 測試排列的前 50 個位置 `[0,50)`，不另選方法、方向或係數。以 `inputs.json` 核對全部來源、資料、既有預測與技術對照檔案雜湊，並核對 50 個查詢 ID 的固定順序。
- 固定原 `mask_seed=1234` 的 64 個新遮罩 `j=392..455`，每筆訓練例以 1% 機率移除，兩個目標 seed 共用完全相同的遮罩。CPU 預檢核對新舊遮罩內容無重疊、兩批各自互異，以及新遮罩的 packed SHA-256。已用於 Stage 0、方向覆蓋與補空間 LDS 的遮罩範圍合計為 `j=0..391`。 [遮罩留出規格](../multifidelity-heldout-lds/PREREG.md)
- 封存訓練程式和設定於 `599469e3faf5dec7aae854cc929c6d265bdd706f`，沿用 CIFAR-10 全 50,000 筆、ResNet-9 W=80、SGD＋momentum、20 epochs、batch 500、原學習率排程、原增強／BN 與原 JAX 精度。移除以權重歸零實現。seed 7 同時更換初始化、排序與增強隨機序列。原 400 查詢的前向形狀、順序與精度維持一致，只保存前 50 個交叉熵。
- 正式新遮罩前，先重播歷史 `j=100/101` 的 seed 0 全 400 查詢，用封存 loss 做技術核對。每個遮罩的 RMSE 除以 `max(std(reference), 0.1)` 須小於 0.02。兩次重播僅作技術核對。 [精確重播允差](../../../killtests/multifidelity-attribution/exact_probe/PROTOCOL.md)

## 評分與判讀

- 每個方法、遮罩及查詢的預測損失變化為移除列分數總和的負值。每個查詢跨 64 個遮罩對該目標種子的取樣重訓交叉熵算 Spearman，再按 50 查詢等權平均得到 LDS。沿用 `c7aa443a3d284ec1accbd841d7d383a4b0f52058` 的秩計算與平手處理。
- 主量為 seed 7 目標的「精確 LDS 減 EK-FAC LDS」。另報 seed 0 的同軌跡差、兩差相減（跨軌跡優勢減同軌跡優勢）、四個個別 LDS，以及兩條目標軌跡的遮罩真值排序一致度。以上共用固定查詢順序，以同一組 10,000 次查詢配對重抽（PCG64 seed 0）報描述性中央 95% 區間。區間不估遮罩或訓練種子的變異。
- 主量點估至少 `+0.05` 且區間下界大於零時，記為 `positive_transfer_at_reference`。區間上界低於 `+0.05` 時記為 `below_reference`，其餘記為 `inconclusive`。`+0.05` 沿用正式跨種子門檻作描述性參考。任一必要預測或真值有查詢為常數時，整體記為 `untestable_incomplete_fixed_panel`，並列不可評位置，其他正式統計留空。技術閘門、指紋、非有限值或執行故障直接中止。

## 執行與硬體

一張 H200、8 CPU、64 GiB 主記憶體、2 小時 Slurm 上限。沿用原 64 遮罩及兩次歷史重播的實測吞吐，兩條軌跡估約 65 GPU 分鐘。GPU 階段只輸出持久化約 25.6 kB 的 float32 loss 和小型收據，無訓練權重或因子。程序完成後於同一作業呼叫 CPU 階段，核對作業、版本、裝置、兩次技術誤差、陣列順序及產物雜湊才評分。唯一正式評分輸出在 `icml/runs/multifid/transfer-lds/<job-id>/summary.json`。 [同設定實測](../../HARDWARE.md)

```bash
stage=codex/experiments/multifidelity-transfer-lds
freeze=FULL_FREEZE_COMMIT
bash "$stage/run.sbatch" "$freeze" --check-only
sbatch --parsable "$stage/run.sbatch" "$freeze"
```

CPU 預檢從指定 commit 和兩份歷史來源封存檔執行測試，確認執行環境、資料和預測檔指紋及新遮罩計畫。正式作業依共用規則在判準凍結及審查通過後提交。

## 讀取紀錄

- [codex] 2026-11-15 16:58（台北）。核對 `26703` 為 `COMPLETED / 0:0`。`runs/multifid/transfer-lds/26703/summary.json` 的 SHA-256 為 `b833dd6abc83e67df59d29a753b4b01a2cd7c148ed5d60c3282368e240a5c2d4`，與日誌、兩份重訓產物雜湊、作業與凍結 `a4c1ce1f182f1fdda26325704a4489abe9f7546a` 均吻合。兩次歷史重播的標準化 RMSE 為 0.002454／0.002008，50 個查詢全可評。同軌跡精確優勢 +0.80218，轉移優勢 +0.01878（描述性 95% 區間 [−0.02617, +0.06533]），機械判為未定。Slurm 實測 1 小時 4 分 18 秒、126.44 GiB VRAM、6.93 GiB RAM，三份產物內容共 40,485 bytes。
