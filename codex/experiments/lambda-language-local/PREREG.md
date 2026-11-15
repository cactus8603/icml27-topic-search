# Λ 全樣本語言診斷的本機重播

[codex] 2026-11-15（台北）。在群組 CPU 配額滿載時，以登入節點單執行緒重播已預登記的 600 筆語言診斷。直接封存並呼叫 `ac12f76c30adb0b491a598af7b20272bb8439113` 的分析函式和測試。題庫、評分封包、fastText、盲辨識、缺測界與 20 題配對重抽依原預登記。預檢只跑合成測試和逐檔完整性核對。 [全樣本診斷預登記](../lambda-language-panel/PREREG.md)

正式執行先寫入盲 ID 計數，關閉檔案後才讀取分臂映射。分臂結果和原 60 筆比對以固定函式彙總。計算程序在 29 秒收到終止訊號、30 秒強制結束。

執行收據記錄本機主機、執行 ID、台北時間、入口與判讀程式的封存版本及輸入指紋。只有完整彙總才發佈 `summary.json`。盲計數和彙總只存在 `icml/runs/inoculation-lambda/language-panel-local-<run-id>/`，日誌輸出執行身分、總筆數和彙總雜湊。

```bash
stage=codex/experiments/lambda-language-local
bash "$stage/run.sh" "$freeze" --check-only
bash "$stage/run.sh" "$freeze" --run
```
