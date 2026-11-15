# 配對 B 的原作重現與短預算資格

[codex] 2026-11-15，台北。這份補充依序測 Gate 0(a) 與 Gate 0(b)(c)。配對 A 繼續沿用原判讀。配對 B 的任何新原作重現、短預算或主判準結果皆尚未產生。送出前須經獨立審查並凍結本文件、配對 B 配置、啟動腳本與所呼叫程式。 [原預登記](PREREG.md) · [既有門檻](configs/criteria.json) · [配對 B 輸入版本](configs/pair_b.json)

## 原作重現

使用固定版 Qwen2.5-1.5B Base 和 Qwen-2.5-1.5B-SimpleRL-Zoo 公開權重。取作者 v1 `cf1c7858bfc145a27aa733e75d930d34f0d318a6` 的 MATH 500、Minerva Math 與 OlympiadBench 原始測試資料（依次 500、272、675 題）及其 prompt、答案抽取、答案等價判分程式。B.3 列 1.5B 使用簡單提示，B.5 說評測沿用訓練模板，公開評測指令沒有額外套用聊天模板。因此事前採作者的 raw `abel` 零樣本模板。作者未公開產出表列 1.5B 數字的完整指令。原作範例以 32B 示範 `qwen-boxed`，本次也沿用本機 vLLM 0.33.0 與 BF16。結果描述為依公開規格的重現，並報告重現差距。 [論文 B.3–B.5](https://arxiv.org/html/2503.18892v2#A2.SS3) · [作者評測說明](https://github.com/hkust-nlp/simpleRL-reason/tree/cf1c7858bfc145a27aa733e75d930d34f0d318a6#main-results)

- 每題一次，依作者資料順序與 benchmark 分別批次執行。引擎 seed 0，Base greedy，RL temperature 1、top-p 0.95，兩臂最多生成 16,000 token。語境上限 32,768，只用模型的原生 context。一旦不容納完整生成預算，即停該組。
- 沿用作者對 Qwen 與 abel 的字串及 token 停止條件。計分呼叫作者的答案抽取及符號等價判分，每筆逾時 3 秒記錯。計分也會處理達長度上限的輸出。另報截斷量與截斷輸出中仍被判對的量。原作重現只有文字推論與答案解析。
- 採現行 `criteria.json` 的 Gate 0(a)。RL 在 MATH 500 及 Minerva Math 都須距原報告值不超過 2 個百分點（各 59.0%、20.2%）。OlympiadBench 及 Base 的逐來源差只作描述。AIME24 與 AMC23 不參與此次閘門，AIME 兩類表格欄亦不混用。閘門未過時保留數字，先區分資料、模板、停止規則、引擎及計分差異。
- 原作程式與資料各檔 SHA-256 記於設定。CPU 計分的 SymPy、ANTLR 等依賴與 GPU vLLM/torch、短預算 Math-Verify 隔離。作業輸出凍結 commit、兩個 checkpoint revision、逐臂資料雜湊及取樣設定。

## 短預算配對資格

由相同權重分別跑 Base、RL。沿用本專案已固定的 `data/eval.jsonl` TEST（1,053 題），每題各取 128 樣本，temperature 0.6、top-p 0.95、BF16、語境上限 4,096，最多生成 3,000 token，長 prompt 會受同一語境上限約束。保留 `pairs.json` 的 B 模板 `simplerl_abel_chat` 與停止 token。兩臂都用四個固定 modulo 分片、既有 vLLM seed 1234 和相同每分片題序，使用真的 Base、RL checkpoint 執行現行正確性檢查。CPU 復用原 Math-Verify 0.9.0 判分、只依正常停止的有效 boxed 答案算分。

Gate 0(b) 要求兩臂截斷率差小於 2 個百分點。Gate 0(c) 要求 RL 的 pass@1 至少高 5 個百分點，pass@64 至少低 2 個百分點，且兩臂 pass@64 都在 30%–85%。短預算資格依原作重現、截斷與交叉依序判讀。原作重現沒過則短預算作業停止。通過此資格後才另行準備稀疏替換與 WiSE-FT。範圍限凍結的模型、測題與短預算設定。

## 離線資產與作業

本機共用模型目錄須具備設定中的兩個 snapshot。上游資料與程式已置於共用 `icml/tmp/rsd-pair-b-cf1c785/`，包含作者 `examples/simplelr_math_eval/` 的內容。重新準備時從上游固定 commit 的 tarball 擷取該目錄，將 `configs/pair_b.json` 指定套件安裝到下層 `repro-deps/`。現有 `pylib/` 提供短預算的 Math-Verify 0.9.0。準備作業先檢查完整權重分片、tokenizer、公開程式及資料指紋與計分依賴，取樣、判分全程離線。

從 repo 根目錄執行。先依協作規則核查佇列及共享配額。每個 GPU 作業用 1 張 H200。原作重現兩臂各設 90 分鐘，短預算兩臂各設 4 小時，兩者均以陣列最多同時跑一張。準備、正式正確性和兩種 CPU 計分各自保存憑證。原作計分沿用作者的單行程，短預算 CPU 評分設 16 核，實際運行時間以 Slurm 為準。每個檔案上限 1 GiB，兩個長取樣 gzip 和八個短取樣 gzip 加上驗證檔，建議留 12 GiB 空間。配額不足時依協作規則處理。

```bash
freeze=REVIEWED_FULL_COMMIT
run=pair-b-v1
launcher=killtests/rl-sparse-distill/slurm/pair_b.sbatch
mkdir -p killtests/rl-sparse-distill/slurm/logs
prep=$(sbatch --parsable --gres=none --cpus-per-task=2 --mem=8G --time=00:20:00 "$launcher" prepare "$run" "$freeze")
correct=$(sbatch --parsable --dependency="afterok:${prep%%;*}" --time=00:40:00 "$launcher" correct "$run" "$freeze")
long=$(sbatch --parsable --dependency="afterok:${correct%%;*}" --array=0-1%1 --time=01:30:00 "$launcher" long "$run" "$freeze")
long_score=$(sbatch --parsable --dependency="afterok:${long%%;*}" --gres=none --cpus-per-task=2 --mem=8G --time=03:00:00 "$launcher" score-long "$run" "$freeze")
```

原作重現通過且獨立核對後，使用同一 run ID 和 commit 送短預算：

```bash
short=$(sbatch --parsable --dependency="afterok:${long_score%%;*}" --array=0-1%1 --time=04:00:00 "$launcher" short "$run" "$freeze")
sbatch --dependency="afterok:${short%%;*}" --gres=none --cpus-per-task=16 --mem=64G --time=01:30:00 "$launcher" score-short "$run" "$freeze"
```

作業輸出根目錄為 `results/pair_b/<run-id>/`，原作重現彙總為 `reproduction.json`，最終短預算結果為 `pair_b_gate0.json`。保留原始檔與憑證後，才另行挑選小型彙總進版本控制。每次新跑次用新 ID。

- [codex] 2026-11-15 06:33。預檢 26492 已寫出完整來源與完成標記，Slurm 仍回報 exit 1，並自動取消其相依作業。預檢以 `exec /bin/true` 完成，並使用新 run ID 重送。模型、資料、門檻及計分固定。
