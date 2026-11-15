# Neural Chameleons 公開資料來源稽核

`26505` 核對了公開 demo 與本地乾淨資料的少量配對。本稽核重建公開完整 pipeline 的評估候選，量兩套資料的精確同列交集和標籤差。沿用原第 12 層 organism 的重現閘門。

來源鎖在作者 Git `261e6547f2042385c95c47ad25860aacedb6aa73`，包含合成 TEST、資料讀取／篩選程式與歷史 `harmful_synthetic/summary.csv`。照公開 pipeline 採 topical response、移除拒答與缺評分，依明列正負類別與 `0.75`／`0.0` 參數分類，保留原順序，分別記完整平衡池和 `medium` 每類前 1,000。公開 CSV 可提供類別大小，歷史 run 的逐列 ID 與探針 metadata 仍缺。匹配以共同公開 JSON 的 0-based 原列索引及精確 prompt／response 配對為準。重複或找不到原列時計數，這類情況不產生有序列 ID digest。 [公開 pipeline](https://github.com/mgm52/neural-chameleons/blob/261e6547f2042385c95c47ad25860aacedb6aa73/experiments/pipeline/run_probe_pipeline.py) · [作者發布範圍](https://github.com/mgm52/neural-chameleons/blob/261e6547f2042385c95c47ad25860aacedb6aa73/docs/PIPELINE_GUIDE.md)

主入口只對 stdout 寫一份 JSON 聚合來源收據。無模型、網路、快取分數、逐列 ID 或原文字輸出。CPU 本機執行即可。

```bash
stage=codex/experiments/chameleon-source-audit
../icml/envs/inoc/bin/python -IB "$stage/test_audit.py"
../icml/envs/inoc/bin/python -IB "$stage/audit.py"
bash "$stage/run.sh" FULL_FREEZE_COMMIT
```

## 來源讀取紀錄

* [codex] 2026-11-15 09:35（台北）讀取作者固定版本的 `harmful_synthetic/summary.csv`，只核對發布的每類分母。尚未以兩個面板執行模型打分。
