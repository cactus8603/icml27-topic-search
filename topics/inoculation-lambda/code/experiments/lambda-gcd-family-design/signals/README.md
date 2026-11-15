# GCD 整家族留出的訓練前分數

在每臂、每 seed 啟動全新的作者 `LoRATrainer`，用同一 PEFT 參數座標、初始化及凍結訓練設定。先嚴格核對來源作業、資料清冊、GPU 技術憑證、完整訓練臂指紋和相同 seed 的 recipe。只用每臂凍結順序前 200 列、另 200 筆錯誤主張目標及三種非誘發情境。訓練梯度使用實際會送入作者 loader 的前綴。PPS hook 只用於訓練來源梯度。目標與無新增防禦前綴的校正 loss 都用撤除 hook 的模型。每個 loss 對受監督 token 等權；算完每個情境的分數後等情境平均。前向採 BF16、關閉 dropout 與 KV cache，累積梯度用 FP32。GPU 短測、設定指紋與正式 cell 憑證各自核對，正式判讀須把每個 `initial_A_B_sha256` 與同臂、同 seed 的訓練前憑證逐一匹配。

所有 `scores` 欄位方向固定為分數較小預測較低漏出。方法與欄位如下。

| 分數 | 定義 |
| --- | --- |
| `lambda` | 訓練梯度與錯答目標 NLL 梯度的內積。正值表示一步 SGD 會提高錯答目標 log-prob。 |
| `gradient_cosine` | 同一組梯度在同一 PEFT 座標的 cosine。任一范數為零時保留缺值。 |
| `negative_target_suffix_nll` | 錯答目標回應 NLL 的負值，量訓練前模型已給此回應多少支持。 |
| `train_suffix_nll_delta` | 同樣 200 個助理答案，防禦輸入的 NLL 減無新增防禦前綴及無 PPS 的 NLL。固定把越容易在防禦輸入下提供訓練答案，解讀為越低的事前漏出預測。 |
| `train_gradient_rms` | 訓練梯度 L2 範數除以全部可訓練 A/B 純量數的平方根。零 LoRA-A 梯度也計入分母。 |
| `trainable_parameter_count` | 可訓練的 A/B 純量數。主要供區分 layer mask 大小，同家族常數依共同支撐評估規則處理。 |

這些是 GCD 面板專用基線。PHM 由獨立入口計算。`lambda` 為訓練前的 SGD 線性量。實際一輪訓練仍依原作者設定執行。

```bash
stage=codex/experiments/lambda-gcd-family-design/signals
bash "$stage/run.sbatch" "$freeze" --check-only
array=$(sbatch --parsable --dependency="afterok:$data_job:$ready_job" --array=0-50%1 \
    "$stage/run.sbatch" "$freeze" cell "$data_job" "$manifest_sha" "$ready_job" receipt)
array=${array%%;*}
sbatch --parsable --job-name=lambda-gcd-family-step0-join --gres=none --cpus-per-task=1 --mem=4G --time=00:05:00 \
    --output="$shared/tmp/lambda-gcd-family-step0-join-%j.log" \
    --dependency="afterok:$array" "$stage/run.sbatch" "$freeze" join "$data_job" "$manifest_sha" "$ready_job" receipt "$array"
```

啟動先向 Slurm 確認資料和 GPU 短測皆成功，從短測作業日誌的最後一筆 JSON 取指定產物的 SHA-256，再校驗產物本身。`receipt` 可換成手動指定 SHA，仍核對日誌及檔案。此機制可讓作業先排相依。

一格一張 H200、上限 15 分鐘。依先前五臂梯度作業粗估每格 1–3 分鐘，含獨立載入模型及三個目標情境重新計算。實際吞吐及 VRAM 待 GPU 試跑，沿用既有模型快取。每格只保存 `icml/runs/lambda-gcd-family-step0/<array>/cell-<task>.json` 的分數和指紋，無逐題文字。全 51 格成功後，CPU 匯總器重查所有工作與憑證並產出 `summary.json`，其中 `cells["<arm>:<seed>"]` 記錄初始化、資料與 recipe SHA、明定方向的分數及來源 cell SHA。已事先凍結的信號整批完成、綁定訓練前初始化後，才適用於對應行為結果的比較。
