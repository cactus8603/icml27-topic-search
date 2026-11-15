# 文生圖候選 Stage 1 入口

沿用原 Stage 1 生成程式、六個模型、十四個提示和每格種子 `[0, 2000)`。固定 DDIM 25 步、512×512、FP16、batch 32 及每塊 500 種子。原三偵測器、判準、事件門檻及縮圖策略由封存版本讀取。每個 array 任務處理一個模型，沿用原一張 H200、16 CPU、96 GiB RAM 及四小時上限。 [原 Stage 1 入口](../../killtest/slurm/stage1.sbatch)

`--check-only` 核對凍結程式、離線資產及 26516 候選憑證，並印出面板和輸出路徑。也用原載入器的預設 Hugging Face 快取解析方式核對五個模型與偵測器來源的 revision 和 checkpoint 路徑。正式執行在生成前再次驗證。 [候選憑證核對](../t2i-gate-repair/README.md)

```bash
shared=/work/ChihChungHsu/cactus8603/icml
stage=codex/experiments/t2i-stage1-candidate
export HF_HOME="$shared/hf_home" HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 TMPDIR="$shared/tmp"
"$shared/envs/t2irare/bin/python" -IB "$stage/test_run.py"
"$shared/envs/t2irare/bin/python" -IB "$stage/run.py" --check-only
bash "$stage/run.sbatch" FULL_FREEZE_COMMIT --check-only
```

## 啟用條件

只有使用者明文採用 26516 作為本次 Stage 1 的代理修復後，才把該決策記入此目錄的 `activation.json` 並提交。記錄使用 `schema` 整數 `1`、`scope` 字串 `candidate_stage1_only`、預檢輸出的 `proxy_receipt` 物件原樣填入 `adopted_proxy`，以及可回查的使用者決策放在 `user_decision_ref`。實際入口接受包含決策的完整提交，並重新核對 Gate (a)、原 bench、候選工作與憑證。歷史 Gate B 產物留在原位置。

## 保存與重試

結果位於 `icml/runs/t2irare/stage1-candidate/<freeze>/`。原保存規格最多有 336 個分塊、33,600 張 256×256 JPEG 縮圖。這些縮圖的未壓縮 RGB 等量為 6.61 GB（約 6.15 GiB），實際磁碟使用依命中、JPEG 大小及檔案系統區塊而異。168,000 筆分數及種子的原始陣列約 3.36 MB，另有壓縮包與側錄開銷。

同一啟用提交下，每個模型只允許一個程序寫入。重試前驗證已有分塊和側錄，再沿用原生成程式跳過完整分塊。若分塊遺失側錄、混入不符種子或留有中斷暫存，入口會停止並保留原檔。查明來源後，將該塊相關檔案和已有的同 seed 範圍縮圖完整移到同一候選路徑下的診斷保留區，再用同一啟用提交重試該模型。成功後每個任務只追加一份完成收據。

此入口的範圍到 Stage 1 產物。Stage 2、正式彙整及盲標清單仍受原 Gate B 驗證，是否接上候選路徑需另行定案。B 的百分比陽性對照最低比率也留待獨立定案。
