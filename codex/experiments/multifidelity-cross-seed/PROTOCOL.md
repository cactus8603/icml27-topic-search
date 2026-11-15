# 兩條訓練軌跡的固定歸因轉移

`26535` 可量 seed 6 的同軌跡 LDS。這個 CPU 比較再沿用 seed 0 和 seed 6 完全相同的 12 個 CIFAR 查詢、100 個 1% 移除遮罩 `j=100..199`，交叉評估兩條軌跡的歸因預測。正式 Stage 0 及 `H_x` 仍依原規約使用前 50 查詢、先前 100 個遮罩、預留 seed 5 和三項通過前提。 [原預登記](../../../killtests/multifidelity-attribution/PREREG.md) · [種子 6 協定](../multifidelity-second-seed/PROTOCOL.md)

- 每個預測 seed 的精確 Y 及事先固定 `damping=1e-8` EK-FAC，各以 float64 彙整 `−sum(被移除樣本的歸因分數)`。對每個目標 seed 的實測重訓 loss 逐題算 Spearman，產生 `0→0`、`0→6`、`6→0`、`6→6` 四格。正向探索重點為原 seed 0 的兩種凍結預測能否在新 seed 6 保留精確法優勢。
- 每格報精確法、EK-FAC 及精確法減 EK-FAC。固定同一目標種子，另報跨軌跡減同軌跡的優勢差，並計兩套實測重訓 loss 在遮罩上的逐題排序一致度。這裡不重選便宜方法也不重訓；EK-FAC 使用各自已凍結的 factor，數值差可以包含估計隨機性。
- 四格和真值一致度共用可評查詢。任何一格遇常數即將全體標為未定，至少九題才給共同子集描述。使用同一組查詢重抽 10,000 次、種子 `2026111538`，報成對的中央 95% 區間與各六題區塊均值。兩區塊共享固定遮罩及兩條訓練軌跡，區間只描述這個小面板，不用作跨種子母體信賴區間或停損。

## 執行

預檢核對 seed 0 歷史收據、源碼、查詢、遮罩和 103 個既有檔案雜湊，並執行合成測試。正式描述須先凍結此檔與程式，再等 `26535` Slurm 成功。程式核對新收據、交叉綁定、輸入雜湊及技術門檻，在獨立作業目錄 `icml/runs/multifid/cross_seed/<job-id>/summary.json` 寫一份聚合結果，附實際打開輸入的台北時間。用原 JAX 環境的 NumPy/SciPy，無新 GPU、重新訓練或資產下載。

```bash
shared=/work/ChihChungHsu/cactus8603/icml
stage=codex/experiments/multifidelity-cross-seed
export CIFAR_DIR="$shared/data/cifar10/cifar-10-batches-py"
export TMPDIR="$shared/tmp" OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1
"$shared/envs/attrib-jax/bin/python" -IB "$stage/test_compare.py"
"$shared/envs/attrib-jax/bin/python" -IB "$stage/compare.py" preflight
bash "$stage/run.sbatch" FULL_FREEZE_COMMIT --check-only
```
