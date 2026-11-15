# 文生圖偵測器操作預檢

原預登記第 4 節容許在代理閘門前量測全偵測器吞吐與批次重現。26288 已執行原端到端 bench。這個小型預檢補量離線模型載入、偵測器單獨及合併操作、NudeNet CPU provider、H200 記憶體。輸入是程式生成的固定 512×512 RGB 算術圖案。所有分數只在記憶體檢查批次形狀、資料型別與數值有效性後丟棄。輸出只有操作時間、資源用量和版本。 [原預登記](../../../killtests/t2i-rare-failure/PREREG.md#4-閘門順序)

H200 路徑用原三偵測器的 FP16／ONNX CPU 設定，分別預熱 batch 32 和 64，各量三次。個別耗時另行在同一個已載入的偵測器實例量測。峰值記憶體包含三個已載入的偵測器，省略生成模型。純 CPU 路徑用兩張圖案，驗證離線載入和實際呼叫。Slurm 預算為一張 H200、16 CPU、最多 15 分鐘，每次寫一份小型 JSON 至 `icml/runs/t2irare/detector-probe/<job>.json`。

```bash
shared=/work/ChihChungHsu/cactus8603/icml
stage=codex/experiments/t2i-detector-probe
export HF_HOME="$shared/hf_home" HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 TMPDIR="$shared/tmp"
export SLURM_CPUS_PER_TASK=4 OMP_NUM_THREADS=4 OPENBLAS_NUM_THREADS=1
"$shared/envs/t2irare/bin/python" -IB "$stage/probe.py" --cpu-smoke
"$shared/envs/t2irare/bin/python" -IB "$stage/stage1_check.py" --check-only
bash "$stage/run.sbatch" FULL_FREEZE_COMMIT --check-only
```

## Stage 1 操作預檢

`stage1_check.py` 核對原啟動腳本、取樣配置和全部六個模型與三個偵測器的離線資產，並列出原 Stage 1 面板。原 `verify_gate_b.py --require-pass` 守衛仍隨原入口執行。使用者若核准 26516 作為原代理閘門的修復，先凍結啟用決定及驗證入口，使 Stage 1、後續 Stage 2 與彙整採用同一個驗證語意。26516 只涉及代理 CLIP 的 logits 精度。正式生成沿用原取樣與三個偵測器配置。 [候選證據核對](../t2i-gate-repair/README.md) · [原 Stage 1 入口](../../../killtests/t2i-rare-failure/slurm/stage1.sbatch)
