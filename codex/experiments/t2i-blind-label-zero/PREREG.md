# 文生圖判準 B 的盲標解析修復

[codex] 2026-11-15（台北）。正式盲標尚未產生時，核對原凍結準則與評分程式。準則允許 `0`、`1`、`2`、`?`，解析器少列 `0`。接受原準則的 `0` 並和原邏輯一樣只把 `1` 算作真陽性。保留其他標籤、對照門檻、樣本、主指標、原生成和選格流程。[原標註準則](../../../killtests/t2i-rare-failure/configs/review_rubric.md) · [判準 B](../../../killtests/t2i-rare-failure/PREREG.md)

使用獨立凍結入口，依雜湊核對 `4351ebb70f2e7ad72e0ec8b99658d9ed41c06567` 的啟動檔與 `b54a86fb05e845e2c11139822beac8846c63c77a` 的解析器。只在封存暫存副本中更正標籤集合。Stage 1、2a、2b、分析前的合格事件與盲標清單沿用原正式入口。人類按原準則填寫盲標後，使用此入口跑 `review-stats` 及 `aggregate`。

```bash
repair=codex/experiments/t2i-blind-label-zero/run.sh
bash "$repair" "$fix_freeze" "$formal_freeze" --check-only
bash "$repair" "$fix_freeze" "$formal_freeze" review-stats "$receipt_sha"
bash "$repair" "$fix_freeze" "$formal_freeze" aggregate "$receipt_sha"
```

預檢只開啟已凍結程式和合成資料，測試四種合法標籤、空白、未知標籤及重複編號。原合成測試曾以 `0` 代表非法值，暫存副本改用 `99` 檢查同一個錯誤條件。正式判讀仍須核對樣本完整、盲標名單、來源和兩組對照。
