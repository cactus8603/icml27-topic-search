# 文生圖罕見事件估計的同分合成診斷

先前用 seed 0–3 探索二元分數。子代理在提交前執行了四臂種子 `[2026111550, 2026111582)` 並讀取 stdout，這批屬已揭露的探索。此確認批固定 32 個不同的種子 `[2026111600, 2026111632)`，沿用原四臂、事件和估計器，每臂執行全部種子。每個種子在四臂重設同一個 NumPy RNG，並核對初始 1,000 個高斯樣本逐位元相同。各臂的接受與初始鏈選擇可能不同，因此後續鏈的位置可分歧。

所有分數的事件均為 `Z ≥ c`，其中 `Z ~ N(0, 1)`，`c = Φ⁻¹(0.999)`，真率為 `0.001`。唯讀預檢用高斯解析尾率及臨界點兩側的浮點鄰值核對真率與事件等價。

| 分數 | 定義 | 命中門檻 |
|---|---|---|
| 連續 | `Z` | `c` |
| 單調量化 | `floor((Z − c)/0.25)` | `0` |
| 大量零分 | `max(0, Z − 2)` | `c − 2` |
| 二元 | `1[Z ≥ c]` | `1` |

沿用固定來源 `b54a86f` 的 `subset_simulation_v2`。本診斷固定一維、`N=1000`、`p0=0.1`、每次記錄前 4 次 MCMC、初始步長 0.5、上限 30 層。回報有限估計與耗盡階段的次數，並以所有種子為分母回報呼叫數和層數。有限估計的偏差與配對連續分數差另列，任一種子失敗時整批的偏差列為缺值。輸出聚合 JSON。這次只量已知真值的分數同分敏感度。原高斯／多峰 (a′)、代理 Gate B、原事件及正式 SD 判讀各依原登記。 [原估計器](../../../killtests/t2i-rare-failure/scripts/estimators.py) · [原預登記](../../../killtests/t2i-rare-failure/PREREG.md)

```bash
shared=/work/ChihChungHsu/cactus8603/icml
stage=codex/experiments/t2i-tied-scores
"$shared/envs/t2irare/bin/python" -IB "$stage/run.py" --check-only
bash "$stage/run.sbatch" FULL_FREEZE_COMMIT --check-only
sbatch --parsable "$stage/run.sbatch" FULL_FREEZE_COMMIT
```

## 執行及讀取紀錄

- [codex] 2026-11-15 10:14（台北）在 Git 凍結前已讀種子 `[2026111550, 2026111582)` 的 stdout 彙總。確認批只用凍結的另一段種子。
