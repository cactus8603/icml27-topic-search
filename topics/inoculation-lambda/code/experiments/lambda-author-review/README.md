# Λ 西語重現的離線複核材料

等待生成作業 `26566` 成功後，依凍結版本 `639bf144b1df0b1cb29ded89b5f5cad68213a606` 核對三臂共 600 筆回覆及訓練、生成收據。依作者原題庫重建提示雜湊，檢查題目及樣本 ID、來源版本、模型、採樣和解碼後的空白回覆數。在三臂所有樣本齊全時建立下列檔案。[作者生成](../lambda-author-generation/README.md) · [Table 13 與評分方法](https://arxiv.org/html/2604.16423v2)

- `pilot.json` 供獨立診斷。每題依固定雜湊選一個樣本序，三臂同序，得到 60 份完整回覆。只存一份共用 rubric、20 個題目和盲化短 ID。所有回覆依盲化 ID 打散，對照表另存。這個小樣本只檢查語言漏出與判讀方式。
- `author_requests.jsonl` 留完整 600 筆、按作者原規格序列化的 Spanish 判分輸入。保存指定版本、單 token、top 20 logprobs 與其餘已固定欄位。CPU 預檢以記憶體替身攔截作者鎖版 judge，檢查請求內容一致。
- `mapping.json` 與完成 `receipt.json` 留在工作目錄。盲化 ID 由本機 salt 產生，所有檔案的雜湊記在收據中。資料包保留評分輸入，正式證據還須有原指定評審的逐筆回應與獨立來源核對。Astra 的診斷另列，原 Gate 2 門檻沿用。

作業使用 CPU，產物在 `icml/runs/inoculation-lambda/author-review-<job>/`，估小於 10 MiB。程式不發出網路請求，日誌只含覆蓋及檔案摘要。只向獨立審查者提供 `pilot.json`。

```bash
stage=codex/experiments/lambda-author-review
shared=/work/ChihChungHsu/cactus8603/icml
TMPDIR="$shared/tmp" "$shared/envs/inoc-phaseb-author-6964328/env/bin/python" -IB "$stage/test_bundle.py"
freeze=$(git rev-parse HEAD)
bash "$stage/run.sbatch" "$freeze" --check-only
sbatch --parsable "$stage/run.sbatch" "$freeze"
```
