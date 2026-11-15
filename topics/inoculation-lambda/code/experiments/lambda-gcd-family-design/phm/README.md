# GCD 的 PHM 文字前綴基線

PHM 先在未微調模型上，對同一批輸入平均每層最後輸入 token 的 residual。接著逐層算各前綴和無新增前綴的平均向量之間的 `1−cos`，最後等權平均各層。原實驗使用 10,000 筆 FineWeb，PHM 所報關聯的目標為 matched 與 off-prefix 行為率差。此實作用固定 10,000 筆無主張、與訓練、Λ 及評估完全切開的 GCD 問句，測同一距離公式能否為本任務的 off-prefix 漏出排序提供可用基線。 [PHM 第 4 節與附錄 B](https://arxiv.org/html/2610.09384v1#S4)

固定 Qwen2.5-7B-Instruct 原始 checkpoint 與其原生模板，對應原 Qwen 層帶的相對位置，選用 28 層中的 0-based 第 9–15 個 block。每個 block 取其輸出的 residual。無新增前綴使用只有 user message 的原生渲染，保留模型模板本身的預設系統文字。所有前綴均取沒有 assistant generation header 的完整輸入最後一個未 padding token。模型以 BF16 前向，逐層累加及 cosine 使用 float64。若輸入超過 512 tokens，該批的資料預檢停止。

正式分數為距離的負值，較小代表預測漏出較低。固定文字臂直接使用其前綴距離，預設有固定前綴但同時使用 PPS 或限制層的臂使用相同公式的 prefix-only 對照。Bank 依入組前已凍結的文字前綴混合權重平均各前綴距離，是本試驗的延伸。未指定可比較前綴的臂輸出 `null`。評分器分開標示文字原公式、prefix-only 診斷和 bank 延伸的覆蓋範圍。

`--check-only` 執行單測，確認快取模型與 tokenizer 指紋、原生無新增前綴渲染；給定清冊與校準檔時，同時離線 tokenize 該固定 panel。`readiness` 短程 GPU 試跑僅用三筆合成數學問題與兩種前綴，核對 residual hook 對應 block index，並比較右側 padding batch 和逐筆 BF16 推論。它只保存最大相對 L2 誤差與技術證明。`score` 在同一程序中先重做 readiness，再以固定校準資料和設定評分。輸出保留 SHA-256、層距離、前綴距離與逐臂分數。正式模式的清冊與題目檔均需提供凍結 SHA-256。

```bash
stage=codex/experiments/lambda-gcd-family-design/phm
bash "$stage/run.sbatch" FULL_FREEZE_COMMIT --check-only
bash "$stage/run.sbatch" FULL_FREEZE_COMMIT --check-only DESIGN_JSON CALIBRATION_JSONL DESIGN_SHA CALIBRATION_SHA
sbatch "$stage/run.sbatch" FULL_FREEZE_COMMIT readiness
sbatch "$stage/run.sbatch" FULL_FREEZE_COMMIT score DESIGN_JSON CALIBRATION_JSONL DESIGN_SHA CALIBRATION_SHA
```

設計 JSON 的 `phm_interface` 需列 `prefixes`（`no_prefix` 為 `null`）、`calibration_rows` 和 `arms`（每列的 `id` 和 `phm_prefix_weights`，可附 `scope`）。校準 JSONL 每列包含不重複的 `id` 及無主張 GCD 問句 `text`。依主設計清冊驗證無序題對互斥。每個 Slurm 作業輸出到 `icml/runs/lambda-gcd-family-phm/<job>/`。
