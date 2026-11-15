# DeepScaleR 公開計數的 32k 重現

[codex] 使用作者公開的成對成功計數，重現同一批題目的 pass@k 交叉，量化兩端回應層級混合與事後逐題選端的改善空間。研究範圍是作者的 32,768-token 面板。專案 3,000-token 正式資格另外需要本專案的共同取樣面板、原作雙面板重現、截斷與判分憑證。 [v1 論文](https://arxiv.org/abs/2609.22547v1) · [正式 3k 預登記](../../killtest/PREREG.md)

## 固定資料與方法

- 只用作者 `ac7c3fc58d5e325447607e778bdcbc897182cf7e` 釋出的 clean 32k 計數。兩臂為 R1-Distill-Qwen-1.5B 與 DeepScaleR-1.5B-Preview。同一批 1,060 題，每題每臂 128 樣本，OlympiadBench 633、Gaokao-2023-EN 358、AMC23 39、AIME24 30。作者記錄溫度 0.6、top-p 0.95、Math-Verify 五秒逾時計錯。公開計數的生成 seed、權重 revision 與完整 runtime 未知。 [作者資料來源](https://github.com/cafferychen777/passk-inference/blob/ac7c3fc58d5e325447607e778bdcbc897182cf7e/docs/PAPER.md) · [作者 metadata](https://github.com/cafferychen777/passk-inference/blob/ac7c3fc58d5e325447607e778bdcbc897182cf7e/examples/deepscaler32k/metadata.json)
- 固定 `base.jsonl` SHA-256 `184d3c7446d8f2ff389195909ca408a2b2bb0984ad726e4549c48adee8af1fc4`，`rl.jsonl` SHA-256 `9da1ffef9e04f951427112a7ee88d3105956952ab30229886a901e2864f5bb45`。只接受逐題 ID 互相對齊、同分層、來源分布與每題樣本量均吻合的計數。資料進作業後核全檔雜湊。
- 沿用本專案凍結的 `pass_at_k_vec` 與 R2 `mix_curve`。作者 v1 的事後稠密網格固定每一個整數 `k=1..128`，4,000 次 Gaussian 題目乘子重抽（seed 0，95% sup-t 同時帶）。必須重現正向區間 1–10、負向區間 61–128 和首次負向的信賴集合 `[11,61]`。輸出 `k=1/64/128` 的兩臂與配對差。這個區間是原作者事後稠密網格，信賴帶只對這批題目與同一組模型有效。 [作者重現程式](https://github.com/cafferychen777/passk-inference/blob/ac7c3fc58d5e325447607e778bdcbc897182cf7e/examples/deepscaler32k/reproduce.py)
- 對照原 `criteria.json` 的 +5 點 pass@1、−2 點 pass@64、兩端 pass@64 都在 30%–85% 及截斷率差 <2 點。前兩者使用新彙總值，截斷只引用同一 clean 32k 面板在論文 v3 表 1 的 Base 7.8%、RL 0.7%（差 7.1 點，超出參考）。公開計數移除了每題截斷分布，故截斷引用是一個獨立欄。互相可比的短預算取樣與 WiSE-FT 同 pass@1 前緣仍待現行專案的正式證據。 [v3 表 1 與評估說明](https://arxiv.org/html/2609.22547v3#S3.T1) · [固定原判準](https://github.com/cactus8603/icml27-topic-search/blob/5808cb676c8fc5ab52b74d7adb0601caa4cfedd6/killtests/rl-sparse-distill/configs/criteria.json)
- R2 固定 `k=64`、`π=0,0.05,…,1`，每次回應獨立選 RL 的機率為 π。用有限樣本超幾何比估計全錯機率。對照在每個題目只抽一次模型並由它生成整批的策略，兩者在相同 π 的平均 pass@1 相同。這個比較限兩個端點的 pass@1 範圍。逐題選兩端中已觀測 pass@k 較高者，僅量固定題目、固定 `k` 的事後選端參考，平手取 Base。另列事後策略的 pass@1，並保留其與實際方法未對齊的限制。

## 執行

使用已準備的 `icml/data/rl-external-passk/ac7c3fc58d5e325447607e778bdcbc897182cf7e/`。凍結預檢只驗雜湊、載入固定程式與執行六項合成測試。正式作業離線，在 CPU 上只輸出一份彙總 `icml/runs/rsd-external-passk/<job>.json`。預估 1 CPU、低於 2 GiB RAM，兩份輸入及彙總共小於 1 MiB。完成後依規則登記讀取並獨立審查。

```bash
stage=codex/experiments/rl-external-passk
bash "$stage/run.sbatch" "$freeze" --check-only
sbatch --parsable "$stage/run.sbatch" "$freeze" --run
```
