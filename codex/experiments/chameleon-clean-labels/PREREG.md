# Neural Chameleons 清潔面板的固定門檻標籤對照

沿用 `5bff580` 公開面板稽核的完整精確交集。僅使用 `cache-d42f49d` 的既有 `harm_test`、`clean`、第 12 層公開 harmful probe 分數。兩臂共用列、分數與門檻，只換本地及公開 medium 標籤。固定 1,174 列，聯合標籤 `(本地,公開)` 的 `(0,0)、(0,1)、(1,0)、(1,1)` 為 605、57、0、512 筆。排除各面板獨有的列，避免混入選樣效應。

以 605 筆同標負例第 `floor(0.99*n)` 個排序值（從 0 起算）固定共同門檻，嚴格 `score > threshold` 算超閾。回報兩標籤定義的 TPR / FPR、公開減本地的兩項配對差及兩種相異標籤方向的超閾率與例數。空組超閾率及區間保持缺值。固定種子 `202611151156`，對四個聯合標籤組分層、有放回重抽 4,000 次。兩臂每次沿用同一列索引和同一數值門檻。95% 百分位區間以固定門檻及固定各組大小為條件。

同一門檻下 `ΔTPR = (57/569) × (異標超閾率 − 同標正例超閾率)`，`ΔFPR = (57/662) × (同標負例超閾率 − 異標超閾率)`。任一共同門檻的 TPR 標籤效應絕對上界由 `57/569` 給出。26572 的共享樣本比較對兩套標籤各自重設門檻，與本次固定門檻比較有不同的估計量。兩次工作使用同一公開模型版本和 probe 權重，讀取的評分來源各異。

`inputs.json` 固定來源、原面板標籤 placement、cache receipt、成對原始列與 shard offset 綁定及 NumPy 版本。預檢核對 18 個 shard 的檔案中繼資料、`row_idx` 成員及完整 permutation，不載入 probe 分數與 activation。正式診斷只使用已驗證 offset 上的乾淨分數和指定 probe 欄。結果僅保留彙總，存於 `/work/ChihChungHsu/cactus8603/icml/runs/chameleon-clean-labels/<job>/summary.json`。對照原 [來源稽核](../chameleon-source-audit/PROTOCOL.md) 與 [共用面板預登記](../chameleon-clean-panel/PREREG.md)。

此對照只量現有面板標籤差。原 Gate 2 的歷史 `merged_probe.pkl`、clean 評估實際選樣清單及 checkpoint 到訓練 metadata 的對應仍待驗證。

## CPU 入口

合成測試與來源預檢在 repo 根目錄執行。凍結後的 Slurm 腳本從 Git 讀取程式，資源上限為 1 CPU、2 GB、5 分鐘。無模型執行、下載或新權重。新增持久空間為單份 JSON。

```bash
/work/ChihChungHsu/cactus8603/icml/envs/inoc/bin/python -IB codex/experiments/chameleon-clean-labels/test_analyze.py
/work/ChihChungHsu/cactus8603/icml/envs/inoc/bin/python -IB codex/experiments/chameleon-clean-labels/analyze.py --check-only
bash codex/experiments/chameleon-clean-labels/run.sbatch "$freeze" --check-only
sbatch codex/experiments/chameleon-clean-labels/run.sbatch "$freeze" --run
```
