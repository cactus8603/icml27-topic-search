# 真實開發集的驗證器量測診斷

強驗證器在真實候選解上，能否比只看同題的模型更常把正確候選排在錯答前？使用已凍結的 GSM8K 面板 `37a8a78`，限 250 題開發集及四種已發布候選。747 題測試的預測、結果與標註判讀留給另行凍結的評估。開發程式先比對底層發布檔雜湊，然後按已固定的列尾 `split=train` 解析。候選請求先執行，完成後才載入同批開發標籤。 [資料面板](PREREG.md)

## 固定量測

- 使用原 Qwen3-0.6B 及 Qwen3-32B-AWQ 權重。兩端同用已固定的 vLLM 0.33.0 原生單步兩 token processed logprobs 與完整性檢查，並先過 GPU 介面測試。沿用原 A/B 雙對應定義與固定 prompt 定義順序交替。若 `d=log P(A)-log P(B)`，語意分數為 `(d₁−d₂)/2`，字母偏好為 `(d₁+d₂)/2`。兩種對應共享的加性字母偏好在語意分數中抵消。 [原生介面定義](../phase3_likelihood.md)
- 每題一次候選盲對照，保留題文及相同提示，只用 `[withheld]`，兩種對應分開測。每個真實候選用原文和中性末行各測兩種對應。中性版只把最後非空白行開頭的 `A:` 變為 `Result:`，保留答案、空白與其餘候選文字。缺少該末行格式時兩版共用請求。相同 token 序列只推論一次並回填原對應。末行標記與判決字母共用 A，直接配對能量到標記是否改變字母偏好或語意排序。
- 每個角色最多 4,500 個邏輯請求加兩個合成介面測試。模型看不到標籤、標準答案、candidate ID 或 solver 變體名稱。覆蓋率、缺測與未知 gold 分開計數。GPU 產物只保留彙總 JSON。

## 判讀

- 主比較為 strong 的中性候選相對候選盲同題排序差。在每題已核對的一正一負所有配對，分數較高算 1，差不超過 `1e-5` 算 0.5，其餘算 0。候選盲在同題兩候選共用分數，所以其同題排序固定 0.5。先在題內平均，再等權平均題目。報百分點差、完整題數與配對重抽 2,000 次的雙尾 95% percentile CI，固定種子 `2026111523`。
- 主要與次要排序差都用同一批正／負均有金標且兩版候選與候選盲分數完整的題。任一已核對候選少一個對應，整題在兩項排序比較記缺測。無已核對雙類的題分開計數。未知 gold 仍可用於不需金標的標記敏感度。
- 次要描述包括同題中性版相對原文的排序差。只在實際替換了末行、雙版兩種對應都完整的候選中，先按題目平均，再報帶符號和絕對字母偏好變化、語意分數變化幅度與語意正負改變比例。標記比較量到的是這一次文字替換的影響，包含文字形式或長度的變化。Weak 同樣輸出，作描述對照。這一步提供開發量測，沒有判斷門檻或測試集規則選擇。

## 執行與資源

一張 H200、上限 60 分鐘。兩端各用獨立程序依序載入，沿用原權重與 sampler；GPU 腳本丟棄可能含提示的原生程式輸出。彙總存於 `icml/runs/fallible-verifier/gsm8k-dev-likelihood/<job>/`。

```bash
shared=/work/ChihChungHsu/cactus8603/icml
experiment=codex/experiments/fallible-verifier/real_holdout
export HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 TOKENIZERS_PARALLELISM=false OMP_NUM_THREADS=1
export TMPDIR="$shared/tmp" HF_HOME="$shared/hf_home" VLLM_NO_USAGE_STATS=1
"$shared/envs/vllm033/bin/python" -IB "$experiment/dev_likelihood.py" --check-only --role weak
"$shared/envs/vllm033/bin/python" -IB "$experiment/dev_likelihood.py" --check-only --role strong
"$shared/envs/vllm033/bin/python" -IB "$experiment/test_dev_likelihood.py"
sbatch "$experiment/dev_likelihood.sbatch" FULL_FREEZE_COMMIT
```

## 讀取紀錄

- [codex] 2026-11-15 11:06（台北），Codex 與獨立子代理已讀 `26565/strong.json` 及 `26565/weak.json`。凍結版本 `2bc9d7e`。
