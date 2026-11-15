# GSM8K 作者候選解的獨立留出集

採用 OpenAI GSM8K 官方發布的 1,319 題候選資料。每題保留原作的四種模型設定及其候選解，依序為 `6b_finetuning`、`6b_verification`、`175b_finetuning`、`175b_verification`。候選來源、資料版本和標籤都來自發表者，與第一、二輪的候選來源分開。這個面板涵蓋留出的官方 GSM8K 題目與固定四種解題設定。 [官方釋出說明](https://github.com/openai/grade-school-math#view-model-solutions) · [固定候選檔](https://github.com/openai/grade-school-math/blob/b0bb162abedc65e1fdd8e93ed090fd7598ee68bc/grade_school_math/data/example_model_solutions.jsonl) · [授權](https://github.com/openai/grade-school-math/blob/b0bb162abedc65e1fdd8e93ed090fd7598ee68bc/LICENSE)

## 資料與分割

- 使用第二輪的 NFKC、負號轉換、casefold 及移除空白題目鍵，核對發布檔與快取的官方 GSM8K test 完整重合。排除第一輪四個固定分片中的所有題目，以及第二輪原凍結的 600 個入選題。官方候選集因此移除 22 和 300 題，留下 997 題、每題四個候選。先前 OpenMathInstruct 開發預覽只取 GSM8K train。
- 使用固定種子 `2026111519` 對題目雜湊排序，前 250 題供開發，剩餘 747 題為單次留出。每題四個候選同屬一側。以題目雜湊排序後寫入固定候選順序，輸出為 UTF-8 的正規 JSON Lines。正式測試前保存並核對候選、金標檔雜湊。
- 原作候選以最後一行的 `A:` 提供答案，官方 test 以 `####` 提供標答。程式用第二輪 Decimal 解析器比對兩份標答，再獨立核對候選最終數值是否與發布的 `is_correct` 一致。保留原標籤；缺少可讀最終行、數值標籤衝突、發布參考不符或空候選時標為 unknown 並計數，所有列保留。
- 候選檔只包含題目、候選、出處及分割鍵。金標檔另存相同 ID、發布的布林標籤、核對後標籤和狀態。方法比較、prompt 選擇與正式統計另行由獨立開發資料決定。發布候選多以 `A:` 收尾，兩種判決 token 的敏感度需在另行凍結的推論規格處理。

## CPU 預檢及輸出

程式驗證原候選檔、官方 test Parquet、原輪次設定和來源分片的 SHA-256，預檢只印計數及輸出內容雜湊。候選原檔 2,563,523 bytes，存於 `icml/data/fallible-verifier-gsm8k/`。

```bash
shared=/work/ChihChungHsu/cactus8603/icml
export PYTHONPATH="$shared/envs/t2irare_extra_cscomp"
"$shared/envs/vllm033/bin/python" -B codex/experiments/fallible-verifier/real_holdout/prepare.py --check-only
"$shared/envs/vllm033/bin/python" -B codex/experiments/fallible-verifier/real_holdout/test_prepare.py
```

正式將 `--check-only` 改為 `--output-dir "$shared/runs/fallible-verifier/gsm8k-holdout/<new-id>"`。只使用新的私有目錄，成功時最後寫 `coverage.json`。題目、候選及逐列標籤保留在叢集。

## 留出彙總讀取紀錄

- [codex] 2026-11-15 11:51（台北），Codex 與 Astra 子代理獨立讀取 `gsm8k-policy-evaluation/26599/summary.json`，凍結 `ac3950820be8c7f4f1528ecc553b9cf51aad894b`。原判讀門檻沿用。
