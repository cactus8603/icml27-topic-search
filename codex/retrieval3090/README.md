# 3090 本機檢索開發

[codex] 2026-11-16（台北）。配合原 [dev-3090 規範](../../topics/compositional-retrieval/code/dev-3090/SPEC.md)，在本機做小規模方法開發；大顯存計算、訓練與正式驗證回叢集執行。

## 本機位置與環境

- 執行根目錄：`dl/retrieval3090/`，已由既有 `dl/` 規則排除於 Git。
- 專用 Python：`dl/retrieval3090/envs/retrieval3090/bin/python`。
- 模型快取：`hf_home/hub/`；暫存與套件下載快取：`hf_home/tmp/`、`hf_home/pip-cache/`，均排除於本機 Git。
- Python 3.12、PyTorch 2.6.0+cu124、torchvision 0.21.0+cu124、Transformers 5.20.2、open_clip_torch 2.32.0。此組合適用目前 NVIDIA 550.163.01 驅動；不需更動系統驅動。
- `requirements-lock.txt` 記錄完整依賴；`setup_env.sh` 優先使用它重裝。建立環境時可用 `PYTHON3_12` 指定現有 Python 3.12 解譯器，預設使用這台主機既有的 3.12 解譯器。只在新的 venv 裝套件。

以下命令從 repo 根目錄執行：

```bash
bash codex/retrieval3090/setup_env.sh
dl/retrieval3090/envs/retrieval3090/bin/python -B codex/retrieval3090/download_assets.py --include-reconstruction-inputs
```

下載器僅從官方公開來源取得三個固定 revision 的模型、COCO 標註與重建舊面板所需的 Half-Truths parquet，驗證來源檔案的大小及 SHA。收據在執行根目錄內。

## 重建原開發圖庫

這台主機沒有叢集掛載。`rebuild_panel.py` 從固定 Git 提交的原始資料建構函式，依序重建三代已曝光面板，只有完全符合各代原始 SHA-256 時才發布 manifest。它會多下載歷史面板的 JPEG 作去重與排除驗證；不讀取封存的 `retrieval-independent-panel`。

```bash
dl/retrieval3090/envs/retrieval3090/bin/python -B codex/retrieval3090/rebuild_panel.py \
  --root "$PWD/dl/retrieval3090" \
  --annotations "$PWD/dl/retrieval3090/data/coco2017/annotations_trainval2017.zip" \
  --half-truths "$PWD/hf_home/hub/datasets--kbora--Half-Truths/snapshots/ddcd448cc27c7c1b1d485c4754623566c245fdca" \
  --fetch-images
```

若被中斷，可重跑同一指令，已有圖片會重新檢查後沿用。進度記錄在 `dl/retrieval3090/rebuild-panel-status.json`。

## 開發入口

`run.sh` 每次從 Git 取出原凍結程式，在自行建立的暫存副本中替換叢集路徑，原始方法、模型設定、batch 大小與基線容差保持原樣。以 `RETRIEVAL_ROOT` 可另指定本機資料根目錄，`PYTHON` 可另指定環境。

```bash
bash codex/retrieval3090/run.sh test
bash codex/retrieval3090/run.sh preflight
CUDA_VISIBLE_DEVICES=3 bash codex/retrieval3090/run.sh smoke \
  --out "$PWD/dl/retrieval3090/smoke-new.json"
```

執行 GPU 步驟前先檢查顯卡使用情形，以上的 `3` 要換成當時有足夠空間的卡。`smoke` 僅對兩張圖片測試三個模型的載入與輸出形狀；沒有開發圖庫時使用純色測試圖片，報告會註明。它不計研究指標。

需要完整開發快取時才執行以下較長的 GPU 步驟，之後分析在 CPU 執行：

```bash
CUDA_VISIBLE_DEVICES=3 bash codex/retrieval3090/run.sh cache \
  --out "$PWD/dl/retrieval3090/runs/dev-new"
bash codex/retrieval3090/run.sh analyze \
  --cache "$PWD/dl/retrieval3090/runs/dev-new"
```

輸出目錄必須尚未存在。完整快取仍須通過原本基線重現容差，失敗時不進入方法分析。

## 與叢集的證據界線

原作業 `26601/summary.json` 是叢集實測收據，無法從公開程式還原，本工具不建立替代品。若它缺少，本機入口仍要求原 manifest、JPEG、模型設定與面板大小全部通過驗證，並在輸出記錄 `source_receipt_available=false`、原收據預期 SHA 與本機重建來源；不宣稱已驗過原作業收據。若日後取得原收據，則走原本的完整校驗。

本機基線重現與技術測試各自紀錄。此處挑選的策略仍只供開發，正式結論須先凍結方法，再由叢集的封存面板與獨立複核確認。原本暫停或淘汰的三題沒有改動。

## 完整開發、共用 GPU 與清理

新版流程與資料／判讀邊界在 [開發協定](PREREG.md)。`run.sh` 預設取已提交的 `HEAD`，即使其他代理正在編輯也不執行未提交檔案。長工作應用 `RETRIEVAL_CODE_REF` 固定完整 commit，續跑仍指定同一 commit。僅技術測試可用 `RETRIEVAL_CODE_REF=working`。

`resources.py` 從 repo 原路徑執行。它接受共用卡，但預設須至少 14 GiB 空閒、GPU utilization 至多 20%，連續兩次檢查後才啟動；沒有適合的卡時持續等待。這是本機合作程序的排程鎖，不會干預其他人的程序。

```bash
python3 -B codex/retrieval3090/resources.py queue \
  --queue "$PWD/dl/retrieval3090/queues/dev-new.json" -- \
  bash codex/retrieval3090/run.sh cache2 --out "$PWD/dl/retrieval3090/runs/dev-new"

bash codex/retrieval3090/run.sh supplement prepare \
  --out "$PWD/dl/retrieval3090/data/lcse-calibration-new"

# D3 texts 先以 diagnostics d3-prepare 固定，再排程執行 supplement encode。
bash codex/retrieval3090/run.sh analyze2 --cache "$PWD/dl/retrieval3090/runs/dev-new" \
  --lcse-calibration "$PWD/dl/retrieval3090/runs/supplement-new/lcse_calibration.json" \
  --lcse-vectors "$PWD/dl/retrieval3090/runs/supplement-new/lcse_vectors.npz" \
  --out "$PWD/dl/retrieval3090/runs/analysis-new"

# 先預覽；確定完整成功、無執行者且重複陣列與最終NPZ一致後才可 apply。
python3 -B codex/retrieval3090/resources.py cleanup-cache "$PWD/dl/retrieval3090/runs/dev-new"
python3 -B codex/retrieval3090/resources.py cleanup-cache "$PWD/dl/retrieval3090/runs/dev-new" --apply
```

清理只刪已驗證、可由最終輸出取代的逐批重複陣列；保留原分數、向量、資料、模型、timing、來源與清理收據。失敗或仍在執行的工作保留續跑檔，不以時間或大小決定刪除。
