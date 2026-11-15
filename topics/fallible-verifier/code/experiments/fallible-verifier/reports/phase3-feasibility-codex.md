# Fallible Verifier 第三輪開發可行性

這次只檢查公開開發資料的答案標籤，以及以固定 tokenizer 接收單一判決的格式。以 GSM8K 正式發布的 **train** 檔核對 OpenMathInstruct-1 的 **validation** 檔。OpenMathInstruct-1 的 train 和 validation 均來自原始 GSM8K 和 MATH 訓練題。候選解由 Mixtral-8x7B 產生，本次僅用 GSM8K 和 `without_reference_solution`，作為格式與標籤開發資料。 [OpenMathInstruct-1 固定版本資料卡](https://huggingface.co/datasets/nvidia/OpenMathInstruct-1/blob/4627efae6bd2ddcebb8acac00d513ffd8e00775c/README.md) · [原版 GSM8K train](https://github.com/openai/grade-school-math/blob/b0bb162abedc65e1fdd8e93ed090fd7598ee68bc/grade_school_math/data/train.jsonl)

兩個類別檔各讀取開頭 1 MiB，只處理有完整換行的 JSONL 資料列。以題文正規化後對回 GSM8K train，用原問題 `####` 後的數字核對 `expected_answer`，再檢查發布的 `predicted_answer` 精確數值是否與二元 `is_correct` 一致。分別得到 324 筆正候選、264 個正題群，以及 48 筆負候選、48 個負題群。兩類共享 15 個題群。範圍內另有 13 筆缺少發布的候選答案、4 筆不可按數值處理、2 筆 publisher 判定與數值比對衝突，這些資料列保留為未知原因計數。

來源給出的 `predicted_answer` 仍由發布者從候選解抽出。此次核實標答來源和可辨識資料的標籤一致性，未核驗抽出的數字是否忠實反映每條候選解。檔案依發布類別分存，前綴也受檔案排序影響。現有樣本缺乏自然類別比例及對目標部署族群的代表性。新來源若要用於獨立評估，須先固定目標資料與標註定義、按題群隔離開發資料與先前實驗題、保留未知與衝突的數量，並抽查候選解與其答案欄是否一致。資料變更也改變 solver 和題源，應另外註冊。

## 單一判決格式

保留 Qwen 無思考 chat template，使用 `A` 表達答案正確和 `B` 表達答案錯誤。兩端都以同一個合成題測試。指定 vLLM `allowed_token_ids`、`max_tokens=1`、`min_tokens=1` 及 greedy，以回傳 token ID 驗證格式。此設定下正常輸出由長度上限終止，對應 `finish_reason="length"`。空輸出、多 token、其他 token 和其他結束原因均為未知。

Qwen3-0.6B 和 Qwen3-32B-AWQ 的快取 tokenizer 通過 SHA-256 核對，`A`／`B` 都對應到 token ID 32／33。兩端均成功建立 vLLM 取樣參數，各自通過 6 個合成回傳案例。

固定選項會強制 verifier 選邊，增加格式覆蓋率本身不能回答誤收或判決品質。後續實際推論若獲授權，可在同一個合成或開發題上以交換 `A`／`B` 對應做配對敏感度檢查，再固定一個規格。這次未載入模型權重、未跑 GPU、未估模型正確率或策略效益。

## 重跑開發檢查

以下命令將原始公開資料前綴和參考集放在共用的 `icml/tmp/`。腳本核對固定前綴 SHA-256 和參考檔 SHA-256，只輸出彙總數。兩個前綴共 2 MiB，參考檔約 4 MiB。 [儲存與暫存規則](../../../../../../COORDINATION.md#6-磁碟與暫存)

```bash
shared=/work/ChihChungHsu/cactus8603/icml
export TMPDIR="$shared/tmp" HF_HOME="$shared/hf_home"
sample_dir=$(mktemp -d "$TMPDIR/fallible-verifier-phase3.XXXXXX")
revision=4627efae6bd2ddcebb8acac00d513ffd8e00775c
for kind in correct incorrect; do
  curl --fail --silent --show-error --location --max-time 30 \
    --range 0-1048575 --max-filesize 1048576 \
    --output "$sample_dir/$kind.jsonl" \
    "https://huggingface.co/datasets/nvidia/OpenMathInstruct-1/resolve/$revision/${kind}_solutions/validation.jsonl"
done
curl --fail --silent --show-error --location --max-time 25 --max-filesize 6291456 \
  --output "$sample_dir/gsm8k-train.jsonl" \
  https://raw.githubusercontent.com/openai/grade-school-math/b0bb162abedc65e1fdd8e93ed090fd7598ee68bc/grade_school_math/data/train.jsonl

strong="$shared/hf_home/hub/models--Qwen--Qwen3-32B-AWQ/snapshots/0499c3ac83fdef8810b907a23894ba91e95eddd8"
HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 OMP_NUM_THREADS=1 \
  "$shared/envs/vllm033/bin/python" -B \
  codex/experiments/fallible-verifier/phase3_feasibility.py \
  --dev-prefix-dir "$sample_dir" \
  --weak-model-dir "$shared/models/Qwen3-0.6B" --strong-model-dir "$strong"
```
