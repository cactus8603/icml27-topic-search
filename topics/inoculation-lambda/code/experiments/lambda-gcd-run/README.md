# GCD 訓練與生成入口

此程式保存五臂、三 seed 的候選執行入口。現階段的 CPU 預檢不需要 Gate 2 結果，可檢查所有 15 個作者 CLI 設定、五份 5,000 列資料指紋及 300 題的合成生成與逐列 schema。沿用候選 GCD 設計，正式替代 Spanish＋ALL-CAPS 仍須原流程的獨立審查及凍結。[GCD 方案](../lambda-gcd/GATE3.md)

```bash
work=/work/ChihChungHsu/cactus8603/icml
stage=codex/experiments/lambda-gcd-run
"$work/envs/inoc-phaseb-author-6964328/env/bin/python" -IB "$stage/run_cell.py" --source "$work/ext/shifting_the_gradient"
```

候選設定是同一 Qwen 7B BF16 snapshot、LoRA `r=32`、`α=64`、一輪、AdamW 的作者訓練路徑、LR `1e-5`、batch 4 × 累積 8、response-only。依已檢查的五臂資料轉換將提示附加在使用者回合，保留 assistant 文字。seed 按 0、1、2 配對。正式開始前由審查者確認 CPU 計畫的 `design_sha256` 和共同超參數。每個陣列子作業申請一張 H200、8 CPU、64 GiB RAM、兩小時上限。沿用作者訓練日誌做十分鐘的實際吞吐核對，45 分鐘為規劃估計，首筆預估超過 90 分鐘即停該格。估計每臂 seed 的最終 adapter 約 0.30 GiB，全組約 4.5 GiB，另外有小型生成與收據。GPU 耗時、記憶體和解析率均待實測。

正式 launcher 會從指定 Git 凍結點讀取 `AUTHORIZATION.json`。此檔只在獨立核定後加入，記錄 `schema: 1`、`decision: "activate_gcd_reliability"`、`design_sha256`、`gcd_reviewed_by`、`gate2_reviewed_by` 與 `gate2_evidence_sha256`。兩個審查欄指向各自負責核定的人或同一條外部審查紀錄。啟動前用固定 `1280b1f` 的原 Gate 2 規則重算該 SHA 對應的經審查 Spanish 證據，確認是原先固定的 `inputs_sha256` 且兩個對比均通過。未具備上述資料時只走 CPU 路徑。隔離來源來自固定作者與 repo commits，生成與訓練只用本地資產。

每格產出在 `/work/ChihChungHsu/cactus8603/icml/runs/inoculation-lambda/gcd-reliability-<array>`。核對作者最終存檔標記、BF16 LoRA 設定、實際設定及檔案指紋後，沿用作者 HF 載入器和 tokenizer 的左 padding，直接送入六族的原始已渲染字串。每題一次 greedy，最多 64 tokens。`no_system` 不會再套聊天模板。JSONL 與 `score_gate3.py` 相容。每格收據附 Git、Gate 2 憑證、數據、設定、adapter 和回答 SHA-256，需等全組完成後另外獨立核對。
