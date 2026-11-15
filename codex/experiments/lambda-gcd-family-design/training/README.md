# GCD 四家族訓練技術檢查

從同一封存版本載入父層家族設計、既有 GCD 模型和作者訓練程式。CPU 前置先以小型 Qwen 測真實 PEFT 與作者 hook，再以四題合成 GCD 驗證 17 臂、3 種子的實際 CLI 設定，交作者資料載入器檢查各族的 system prefix、銀行交替、右 padding 與助理回答標籤。成功輸出每臂的合成輸入 SHA-256。

GPU 模式用同一份 Qwen 模型和作者 LoRA 設定，對每族的第 0 臂、seed 0 驗證 rank 32、LoRA-A 非零、LoRA-B 全零以及實際的可訓練參數。另用 seed 1 和 2 重驗完整與後 14 層兩種設定的初始化。重用在記憶體中的基座，並以作者 `_setup_lora` 裝入各臂設定。以獨立的合成微批次測 step-0 梯度與作者 accelerator 的反向梯度時，檢查兩者從同一份初始 A/B 位元與遮罩開始。PPS 透過作者 hook 加上向量，在第 20 個含 embedding 的索引核對逐位元相等的位移與撤除後回復。

GPU 收據包含 CPU 設計清冊雜湊、模型與 steering 資產、合成資料與 token 批次 SHA-256、每個代表格的初始化 SHA-256、投影數、啟用的梯度數及 PPS 診斷。預先固定清冊的 SHA-256 可由 CPU 設計的 `--check-only` 取得。GPU 只讀完整清冊驗證家族配置，實際微批次仍由合成題組成。只在完整通過後寫入 `icml/runs/lambda-gcd-family-train-ready/<job>/summary.json`。不做優化器更新、不讀正式資料或漏出結果、不寫權重。正式的分開作業要再比對每格實際傳遞給預測與訓練的初始化和輸入憑證。

```bash
stage=codex/experiments/lambda-gcd-family-design/training
bash "$stage/run.sbatch" "$freeze" --check-only
sbatch --parsable --dependency="afterok:$data_job" "$stage/run.sbatch" "$freeze" readiness "$manifest" "$manifest_sha"
```
