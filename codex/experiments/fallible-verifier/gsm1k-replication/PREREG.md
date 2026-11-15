# GSM1k 近分布複驗

Scale AI 的 GSM1k 是另行撰寫、刻意貼近 GSM8K 的數學題群。用它複驗強驗證器的共同誤收、條件訊號與等暖機策略省費。前一輪的 747 題固定策略有已驗證的成本紀錄，本輪以新的作答模型產生候選，並用新暖機資料重建原方法的政策。公開題庫可能出現在模型既有訓練資料，獨立性指本專案的題群與政策訓練。 [作者論文](https://arxiv.org/abs/2405.00332) · [作者公開資料固定版](https://huggingface.co/datasets/ScaleAI/gsm1k/tree/bc09569d09a614b9b530edc7f076fb214ac10493) · [原留出協議](../holdout-evaluation/EVALUATION_INTERFACE.md)

## 題群與候選

- 凍結作者版 `bc09569d09a614b9b530edc7f076fb214ac10493` 的 Arrow（389,520 bytes，SHA-256 `bf7b848f2badbdcb138d6fbc1f246a1902337ce4c00cc079e4f35baf8ed10764`）。1,205 題的作者金標均為整數。按原題文正規化排除完整 GSM8K test 與前兩輪。近重複使用正規化後連續五字元 shingles 的 Jaccard ≥0.80，先對舊資料，再按固定順序對新資料去重。固定 seed 202611151410 的雜湊順序取 997 題，前 250 暖機、後 747 留出。準備器只看題目和作者金標。
- 每題固定四個真實候選。Qwen2.5-Math-1.5B Base 與 Oat-Zero 各生成兩次，沿用封存的 Qwen Math boxed 模板、vLLM 0.7.2、BF16、temperature 0.6、top-p 0.95、每筆最多 768 新 token，同題 seed 從題 ID 固定衍生。作答器只讀題文，保留原生完成原因。候選名冊為 `base-0`、`base-1`、`oat-0`、`oat-1`。
- 以官方整數和候選最後完整 `\boxed` 的精確 Decimal 比對，空白、未正常完成和不能解析的候選列為未知。金標檔只包含候選身分與判定，另外記錄該判定使用的規則。公開來源提供每題標準答案，候選標籤由此規則計算。逐題內容與收據留在叢集。

## 固定驗證與判讀

- 凍結原 `ac3950820be8c7f4f1528ecc553b9cf51aad894b` 的 Qwen3-0.6B 和 Qwen3-32B-AWQ、vLLM 0.33.0、雙對應中性 prompt、2048 token 輸入界線及弱端 0／強端 0.25 棄權界。兩端各自用同一組候選，完整覆蓋兩種 A／B 對應才判決。缺測保留，評分程式只讀候選及來源收據。
- 驗證原 16 個暖機籤，由 1,000 個暖機候選重擬合。兩臂的共同攤銷為每候選 0.135，強判決對金標價格 1:10。留出報原共錯、條件 Brier、相同成本的雙端策略、直接決策和全暖機資訊參考。原金標覆蓋 ≥85%、兩真類至少 50 及六項風險與 20% 省費門檻沿用。按題群共同重抽與原多重檢定。GSM1k 的原方法複驗與 GSM8K 固定策略族的事後上限各自解讀。 [原門檻與成本](../round2/PREREG.md) · [事後上限協議](../empirical-policy-ceiling/PREREG.md)

## 執行

同一項 Slurm 作業依次準備題群、兩個作答器、候選與金標組裝、兩種驗證器及 CPU 彙總。GPU 程序各自結束才載入下一個模型。任何輸入或收據核對失敗即中止。保留每項局部收據，等 Slurm 完成後才讀封存彙總。使用單張 H200，預估單次 **15–40 GPU 分鐘**，作業上限 2 小時，沿用快取權重。產物預估低於 **40 MiB**，僅留在 `icml/runs/fallible-verifier/gsm1k-replication/<job>/`。

```bash
stage=codex/experiments/fallible-verifier/gsm1k-replication
bash "$stage/run.sbatch" FULL_FREEZE_COMMIT --check-only
sbatch --parsable "$stage/run.sbatch" FULL_FREEZE_COMMIT
```
