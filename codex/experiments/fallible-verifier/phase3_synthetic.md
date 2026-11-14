# Fallible Verifier 單 token GPU 格式診斷

用同一算式模板產生八道 GSM 風格算術題，每題搭配一個正確數字和一個偏移 1 或 2 的錯誤數字，共 16 個候選。兩個 verifier 各用兩種 A／B 對應判斷同一批候選。每種對應固定指定一個 token 給答案正確，另一個給答案錯誤。依題號交替兩種提示的處理順序。兩端使用 Qwen 無思考模板、greedy 和單 token allowlist。

每個 verifier 只記錄兩種對應各自的有效回傳比例、兩種都有效的比例，以及兩種都有效時解碼成同一語意後的判決不一致比例。有效輸出是單一 A 或 B token ID 且由一 token 上限終止。記錄比例時同時保留分子和分母。單一題型只用於驗證 GPU 跑通及檢查選項對應敏感度。程式不評估答案正確率、誤收風險或策略效果，也不讀取過去或公開資料的題目與標籤。

## CPU 送件前檢查

```bash
shared=/work/ChihChungHsu/cactus8603/icml
runner=codex/experiments/fallible-verifier/phase3_synthetic.py
strong="$shared/hf_home/hub/models--Qwen--Qwen3-32B-AWQ/snapshots/0499c3ac83fdef8810b907a23894ba91e95eddd8"
export TMPDIR="$shared/tmp" HF_HOME="$shared/hf_home"
export HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 OMP_NUM_THREADS=1
"$shared/envs/vllm033/bin/python" -B "$runner" --check-only --role weak --model-dir "$shared/models/Qwen3-0.6B"
"$shared/envs/vllm033/bin/python" -B "$runner" --check-only --role strong --model-dir "$strong"
```

CPU 檢查驗證合成提示、固定模型的 config 和 tokenizer SHA-256、模型權重檔是否存在、vLLM 版本與合成的彙總案例。完整的權重 SHA-256 核對在 GPU 路徑執行。

## GPU 作業

一張 H200、20 分鐘，分兩個進程先後載入兩端。提交前凍結相關程式、配置和本文件，並核對佇列與登記作業。執行時核對凍結版本與工作樹，輸出放在 `results/phase3-synthetic/<jobid>/`。每端一個彙總 JSON，Slurm 日誌與快取放在 `icml/tmp/`。

```bash
freeze=$(git rev-parse --verify HEAD)
sbatch codex/experiments/fallible-verifier/phase3_synthetic.sbatch "$freeze"
```
