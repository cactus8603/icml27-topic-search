# Λ 西語 Gate 2 的離線 Batch 轉接

工具核對封包 `26567` 的收據、既有評分匯入器的鎖版及 600 筆請求網格。輸出每筆請求的 Batch 外層欄位及原始 Chat Completions 請求物件，並用盲 ID 與請求 SHA-256 的前 32 個十六進位字元組合成 `custom_id`。不加 `--out` 時只回報列數及 SHA-256。 [既有匯入與正式 Gate 2](../lambda-author-import/README.md) · [OpenAI Batch 格式](https://developers.openai.com/api/docs/guides/batch)

```bash
lambda_batch=codex/experiments/lambda-author-batch/batch.py
lambda_packet=../icml/runs/inoculation-lambda/author-review-26567
lambda_python=../icml/envs/inoc/bin/python
"$lambda_python" -IB "$lambda_batch" --packet "$lambda_packet" export
"$lambda_python" -IB "$lambda_batch" --packet "$lambda_packet" export --out "$batch_input"
```

使用已核對來源的完整 Batch 成功回覆檔。保留原始上傳檔、Batch ID、輸入和輸出檔 ID、請求狀態與原始下載檔供獨立審查。轉接器比對上傳檔的完整固定內容及兩檔雜湊，按 `custom_id` 核對所有 HTTP 200 回覆與日期模型，並拒絕缺列、重複的識別碼和錯版。每列原生 `response.body` 逐位元組保留於既有匯入器所需的外層格式中。輸出只包含新檔的識別資訊，分數由既有匯入器按凍結協定計算。

```bash
"$lambda_python" -IB "$lambda_batch" --packet "$lambda_packet" wrap \
  --batch-input "$batch_input" --batch-input-sha256 "$batch_input_sha256" \
  --batch-output "$batch_output" --batch-output-sha256 "$batch_output_sha256" \
  --out "$native_responses"
"$lambda_python" -IB codex/experiments/lambda-author-batch/test_batch.py -v
```
