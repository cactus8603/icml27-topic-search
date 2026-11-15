# 兩個時鐘的同輸入區塊差距

第 2 週正式 `G3` 比較共同留出與自身訓練區塊的 CFG3 精確分數誤差。這裡用同一份封存結果中的完整模型 × 區塊 MSE 矩陣，拆出同輸入區塊下可見的成員差，以及共同留出與其他訓練區塊的差。第 2 週正式判讀與 CIFAR 閘門沿用原規格。 [第 2 週預登記](../../../killtests/biased-gen-cfg/PREREG.md) · [原正式複核](../../reports/2026-11-15-three-frozen-readbacks-codex.md#兩個時鐘)

## 固定資料與估量

- 驗證來源 Slurm `26383` 任務名、成功退出、提交紀錄與原凍結 `9e92804a7bfded3c5cafd4fccd4315df2034eca5`。來源 `hier_oracle_w2_k6.npz` SHA-256 `4f33e24a7a332358cf8d098d3877f517ae1e5be5687008222b77bbdee1f9aaea`，原 `week2_verdict.json` SHA-256 `b2af1ce8a25513a15ad5b7aacb811336b533a40dd6b594cc33743af0a5689551`。使用原 manifest 驗六個 K=6 pick、完整時間網格、每區塊前 1,000 筆與留出區塊 7。
- 用共同留出的條件 CE 於八個 t 與六模型取平均、三點平滑後選原 `e†`。若在第一格則不判讀，末格照用並標記。核對原摘要的 `e†`、t 遮罩、G3、原 95% 區間、分子分母與機械狀態。
- 固定該 checkpoint 與原六個 t 的 CFG3。以 `Mij` 表示模型 `i` 在區塊 `j` 的 x0 oracle MSE 加總，`Ti` 表示該模型的共同留出 MSE 加總。各對 `i<j` 取 `A=(Mii+Mjj)/2`、`B=(Mij+Mji)/2`、`C=(Ti+Tj)/2`。再對 15 對等權算平均 `a,b,c`。同輸入差為 `h=(b−a)/b`。用同一原分母拆為成員差項 `(b−a)/c` 和留出區塊差項 `(c−b)/c`，兩者恰加至原 G3 `(c−a)/c`。同一對兼具兩個模型與兩個輸入區塊，可消去加性的模型主效應與區塊難度。
- 區間以六個既有 pick 重抽 2,000 次，seed 0。配對權重為兩個 pick 的重抽次數乘積，四個比率在同一重抽用同權重。分母非正或沒有相異配對時跳過，至少需 95% 有效重抽。配對區間為固定模型與資料的描述性敏感度。原 G3 區間由原獨立模型重抽重算比對。`h` 的點估與區間另依 10%／5% 原幅度線標示參考位置。
- 另一模型可能在自己的區塊見過與此區塊相同的葉序列，且模型與區塊仍可能有交互作用。此實驗只從固定六模型的舊資料重用成員對照。這個設計可指出原灰區的組成。獨立重現需重新抽樣與訓練模型。

## 執行與磁碟

單 CPU 核、4 GiB 主記憶體，作業上限五分鐘。輸入沿用約 4.8 MiB NPZ 和約 12 KiB 原摘要，永久產物為數 KiB 彙總。`--check-only` 只核對兩個指紋、封存入口與合成資料。正式執行完成後才讀新結果並登記。

```bash
stage=codex/experiments/two-clocks-input-block
bash "$stage/run.sbatch" FULL_FREEZE_COMMIT --check-only
sbatch --parsable "$stage/run.sbatch" FULL_FREEZE_COMMIT
bash "$stage/run.sbatch" FULL_FREEZE_COMMIT --local
```

## 已知結果與讀取紀錄

- [codex] 2026-11-15 17:16（台北），制定診斷前已閱讀原正式複核與十題報告中的 G3 彙總。僅以雜湊核對舊 NPZ 和判讀檔，本步未打開同輸入矩陣或其中各格數值。
