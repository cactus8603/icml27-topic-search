# Fallible Verifier 兩種標籤對應的似然診斷

開發資料固定為找零、平分、經過時間、分數四類，每類六個新編問題。每題用可直接驗算的正確數字搭配一個錯誤數字。每類的正確數字位置和錯誤偏差方向各半。題文和候選數字放在 [cases](phase3_likelihood_cases.json)，獨立標答放在 [gold](phase3_likelihood_gold.json)。程式用精確算術再驗算標答。實際推論完成後才載入 gold，所有提示由 cases 建立，模型輸入沒有候選真類、標答欄或內部 ID。

每個候選用兩份只有 A／B 對應互換的提示。同一題的對應定義順序固定，各類正確定義在前或在後各半。每題另有一個相同格式的候選隱藏對照，把數字換成 `[withheld]`，並跑兩種對應。對照保留同一問題，因此測量模型在缺少候選答案時的偏好。它在同一題的正負候選上使用相同分數，因此無法區分兩者。每個模型共 144 個診斷提示，另有 2 個 GPU 介面測試提示。

每次由 `allowed_token_ids` 固定 A／B，使用 vLLM 0.33.0 原生的 `logprob_token_ids` 索取同一步的兩個值。要求 `processed_logprobs`，檢查兩個 logprob 皆有限，且指數化後加總為 1。以輸入 token IDs 核對回傳和請求相等，核對 request IDs 未重複且 greedy 的單一輸出和較大 logprob 相符。GPU 介面測試必須完整通過。個別請求缺輸出或缺任一分數時記缺測。回傳身分不符、出現其他 token、分數無法驗證或 smoke 缺測時，該模型執行失敗。 [vLLM SamplingParams](https://docs.vllm.ai/en/stable/api/vllm/sampling_params/) · [vLLM 取樣設定](https://docs.vllm.ai/en/stable/configuration/engine_args/)

令 `d = log P(A) − log P(B)`，按「A 為正確」及「B 為正確」的順序取得 `d₁`、`d₂`。語意帶符號分數為 `(d₁ − d₂) / 2`，標籤偏好為 `(d₁ + d₂) / 2`。這會抵銷兩份提示共享的加性 A／B 偏好。隱藏答案對照用相同公式及門檻。有足夠分數且語意分數絕對值至多 `log(1.25)` 時棄權。任一候選或對照的任一對應缺測時，兩臂在該題同時記缺測。

輸出各類和整體的配對排序勝／平／負、固定門檻後判決正誤與棄權，以及缺測數。排序對照以同題正確候選高於錯誤候選計勝，差異在 `1e-5` 以內計平。完整題組另記語意與標籤偏好絕對值的中位數。這些未經校準的開發分數只用來判斷合成題能否測到候選訊號，以及辨認對應偏好和覆蓋率。資料及現象判讀獨立於 Round 2。

## 執行

兩個模型各用下列命令做 CPU 預檢。此步驗算 gold、檢查模型設定與 tokenizer SHA、確認權重在快取、建構 144 個唯一請求與原生取樣參數。

```bash
shared=/work/ChihChungHsu/cactus8603/icml
runner=codex/experiments/fallible-verifier/phase3_likelihood.py
strong="$shared/hf_home/hub/models--Qwen--Qwen3-32B-AWQ/snapshots/0499c3ac83fdef8810b907a23894ba91e95eddd8"
export TMPDIR="$shared/tmp" HF_HOME="$shared/hf_home"
export HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 OMP_NUM_THREADS=1
"$shared/envs/vllm033/bin/python" -B "$runner" --check-only --role weak --model-dir "$shared/models/Qwen3-0.6B"
"$shared/envs/vllm033/bin/python" -B "$runner" --check-only --role strong --model-dir "$strong"
TORCH_COMPILE_DISABLE=1 VLLM_USE_FLASHINFER_SAMPLER=0 "$shared/envs/vllm033/bin/python" -B \
  codex/experiments/fallible-verifier/test_phase3_likelihood.py
```

凍結題文、標答、程式及本說明後，以一張 H200 執行，作業上限 24 分鐘。沿用 Round 2 權重版本、載入限制及 PyTorch sampler，循序驗證兩個模型；保留 eager 路徑，避免小型診斷為 logprob 排名額外編譯。作業出入口會丟棄模型進程的 stdout／stderr，以避免 vLLM 故障時的請求 dump。Slurm 日誌只記角色進度；結果目錄各角色只寫一個彙總 JSON。

```bash
freeze=$(git rev-parse --verify HEAD)
sbatch codex/experiments/fallible-verifier/phase3_likelihood.sbatch "$freeze"
```

## 讀取紀錄

- [codex] 2026-11-15 08:09–08:10 主代理與獨立子代理已讀 `results/phase3-likelihood/26520/{weak,strong}.json` 的機械彙總，沿用 `b6023ce`。此為合成開發診斷，與真實候選留出分開。
