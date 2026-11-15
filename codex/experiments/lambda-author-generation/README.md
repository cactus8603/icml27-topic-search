# Lambda Phase B 作者 Spanish 生成

只準備 Gate 2 的 Base、none 和 IP 三臂回覆。作者的 `paper_run.sbatch` 與 Spanish Base 分析皆使用 `eval_trait.py` 的預設 Hugging Face backend。依 `6964328` 的題目載入器、HF 模型載入器、聊天模板和 `generate_responses` 生成，從評審前的步驟取得回覆。直接使用已安裝的 Python 3.10.20 鎖版環境與本地 Qwen2.5-7B snapshot `a09a35458c702b33eeacc393d103063234e8bc28`。LoRA 在推論時套用至原模型，保留作者的 PEFT 路徑，不合併權重。此階段不用 vLLM、API、憑證或評審。[作者評估入口](https://github.com/satchel-goodfire/shifting_the_gradient/blob/6964328411bf0ae49de47fa1bdfcfe8f94a75cce/scripts/eval_trait.py) · [作者生成與評審流程](https://github.com/satchel-goodfire/shifting_the_gradient/blob/6964328411bf0ae49de47fa1bdfcfe8f94a75cce/grad_shift/evaluation/pipeline.py)

使用 `extract` 的 20 題，按原順序每題 10 樣本，固定 system prompt `You are an LLM chatbot.`、seed 42、batch 20、最多 500 新 tokens、BF16、temperature 1、top-p 1，沿用底層 Qwen 的 top-k 20、repetition penalty 1.05 及 EOS。CPU 預檢核對來源和 snapshot 的小型檔雜湊、權重分片指紋及大小，並以空白回覆 CPU 替身走完整作者 tokenization 及十批呼叫。實際 GPU 文字及敘述性吞吐需要首次執行才有。作者原始 Hub revision 及歷史 GPU 數值環境未發布，因此這是對當前固定版設定的重跑。

生成作業以 `afterok:26526:26527` 等兩個訓練成功，再讀完整輸入憑證確認同一訓練版本與模型。三臂以獨立 Python process 依序執行，生成前驗證該臂已完成的 adapter 檔案雜湊。每臂的 `responses.jsonl` 保存短 ID、作者問題 ID、樣本序、提示雜湊和未修改回覆；文字不進 log。遇到解碼後空白照常留在固定格內，並記數。各 `receipt.json` 記錄 200 筆覆蓋、採樣參數、回覆與原訓練憑證的雜湊。整套後續只接受三臂完整收據，評分及閘門判讀另走原協定。

一張 H200、8 CPU、64 GiB RAM，Slurm 上限 90 分鐘。推估三臂共 15–45 分鐘，尚未量測，最多 300,000 個生成 tokens。共用輸出在 `/work/ChihChungHsu/cactus8603/icml/runs/inoculation-lambda/author-generation-<job>/`，估小於 10 MiB。沿用已有模型、adapter 和環境空間。

```bash
stage=codex/experiments/lambda-author-generation
freeze=$(git rev-parse HEAD)
bash "$stage/run.sbatch" "$freeze" --check-only
# 完成審查後送入佇列，兩個訓練作業均成功後才執行
sbatch --parsable "$stage/run.sbatch" "$freeze"
```
