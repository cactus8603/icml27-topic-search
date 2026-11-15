# 配對 A 短預算停止診斷

此診斷量化 Base、RL 在原 3,000-token 取樣規格下的截斷差，以及額外停止 token 對差距的影響。以 0.7.2 推論環境固定生成兩模型，保持同題、同模板、同抽樣設定。正式配對資格、pass@64 與稀疏替換依原 Gate 0 流程。 [原規格](../../../killtests/rl-sparse-distill/PREREG.md) · [首版公開推論環境](../rl-oat-v1-engine/PROTOCOL.md)

## 固定輸入

- 封存 `1432b5dffa9527677f3cce697f138bbe437c40cb` 的 `pairs.json` 與 `eval.jsonl`。Base 為 `4a83ca6e4526a4f2da3aa259ec36c259f66b2ab2`，RL 為 `a98e477854071157a450e57dd45fd0684b6fa38a`，離線核對權重與模型／tokenizer SHA。沿用配對 A 的 `qwen_math` 提示，兩端對所有原 test 提示的 token 序列須一致。
- 原 1,053 個 test 題中，提示長度 ≤1,096 的共同可用 1,052 題構成抽樣框。按 `SHA256("20261115-rl-a-short-stop-v1:panel:" + 題 ID)` 並以 ID 作次排序，取最前 128 題。固定題目順序、每題 4 筆及題目 ID 陣列 SHA-256 `4f36ef324b3bb77c046c96d13e99b25c61223e143baaffd67643cd585aa05fd7`。同一題的抽樣 seed 取 `SHA256("20261115-rl-a-short-stop-v1:draw:" + 題 ID)` 前四位元組作無符號大端整數，再對 `2^31−4` 取餘。兩個模型使用相同計算，依同題做配對。
- 沿用 BF16、temperature 0.6、top-p 0.95、4,096 context 與 3,000 新 token。沿用已核對的 vLLM 0.7.2 V0 堆疊，其 package inventory SHA 為 `c49872e678c0d7cdfbd3d17f3d2521b40b28ce036a445872a97598830d00f2eb`。一次只載一個模型，GPU 記憶體配額 0.90，使用原程式的 prefix cache。

## 配對量測

- 實際生成只使用兩模型皆有的原生 EOS `151643`。影子停止規則沿用舊程式的 `151645`，在原生生成的 token ID 序列中視首個 `151645` 為額外的停止位置。輸出足 3,000 且原生未停，才計為原生截斷。該輸出內如果也有 `151645`，影子規則計為正常停止。停止檢查早於同一步長度限制。
- 每模型回報兩種規則的截斷率、額外停止 token 出現率，以及其對截斷率的改變。回報共同題目上的兩個 Base 減 RL 差及差之差。用全臂共用的 10,000 次題目重抽（seed `2026111511`）求配對 95% 區間，保留每題四筆為一群。各規則的點差按 `|Base−RL|<2` 個百分點列參考欄，抽樣設計只提供停止與截斷診斷。
- 影子配對量的是在同批固定路徑套入停止規則的直接影響。引擎變動、真實提早停止對排程的影響及 128 筆正式抽樣資格依其各自規格另查。正式短預算交叉與數學正確性仍待相應試驗。

## 執行與交付

CPU 預檢含 4 個合成測試、權重與資產指紋、固定題組與相容性。GPU smoke 每模型跑同兩筆合成題，各四樣本、16 新 token，核 EOS 與 prompt 邊界，僅留技術收據。正式診斷依賴同一凍結版的 smoke。每次只佔一張 H200。試跑估 3–5 分鐘，診斷估 5–10 分鐘，Slurm 上限 25 分鐘。共用空間保留 <1 MiB 彙總，逐題計數暫存在作業隔離目錄，生成文字留在單次程序記憶體。

```bash
stage=codex/experiments/rl-a-short-stop
bash "$stage/run.sbatch" "$freeze" --check-only
sbatch --parsable "$stage/run.sbatch" "$freeze" smoke
sbatch --dependency="afterok:$smoke_job" "$stage/run.sbatch" "$freeze" diagnostic "$smoke_job"
```

收據位於共用 `icml/runs/rsd-pair-a-short-stop/`。判讀前先核對凍結與 Slurm 完成。

[codex] 2026-11-15 13:10（台北）。合成試跑 `26655` 在運算節點比對權重掛載路徑時停止。依已驗證的首版診斷做法，先解析共用目錄別名再比對固定位置。原符號連結與檔案大小維持核對，正式載入前再核該模型的內容 SHA。取消無法滿足相依的 `26656`，按原抽樣與判讀重送。
