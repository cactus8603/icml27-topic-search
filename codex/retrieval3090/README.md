# 3090 本機檢索開發

[codex] 2026-11-16（台北）。配合原 [dev-3090 規範](../../topics/compositional-retrieval/code/dev-3090/SPEC.md)，在本機做小規模方法開發；大顯存計算、訓練與正式驗證回叢集執行。

## 目前研究基線

[codex] 2026-11-16 13:39，依使用者指示，後續方法比較正式採用本機torch214平台的CORE50為研究baseline。主要環境為 `dl/retrieval3090/envs/retrieval3090-torch214/bin/python`，基準快取為 `dl/retrieval3090/runs/platform-torch214-20261116/`。本機環境、資料、模型及評估保持一致；舊叢集數字的對齊紀錄只作來源追溯，不再作本輪研究是否可繼續的條件。

現有分析入口沿用成對的 `--platform-plan`、`--platform-receipt`，驗證本機快取及來源；不用修改歷史產物狀態。舊相容環境留作診斷，不與主研究快取混用。新的獨立確認仍須事前固定方法與評估方式。

## 本機位置與環境

- 執行根目錄：`dl/retrieval3090/`，已由既有 `dl/` 規則排除於 Git。
- 專用 Python：`dl/retrieval3090/envs/retrieval3090/bin/python`。
- 模型快取：`hf_home/hub/`；暫存與套件下載快取：`hf_home/tmp/`、`hf_home/pip-cache/`，均排除於本機 Git。
- Python 3.12、PyTorch 2.6.0+cu124、torchvision 0.21.0+cu124、Transformers 5.20.2、open_clip_torch 2.32.0。此組合適用目前 NVIDIA 550.163.01 驅動；不需更動系統驅動。
- `requirements-lock.txt` 記錄完整依賴；`setup_env.sh` 優先使用它重裝。建立環境時可用 `PYTHON3_12` 指定現有 Python 3.12 解譯器，預設使用這台主機既有的 3.12 解譯器。只在新的 venv 裝套件。

另有同代 PyTorch 對照環境 `dl/retrieval3090/envs/retrieval3090-torch214/bin/python`：PyTorch 2.14.0+cu126、torchvision 0.29.0+cu126，其餘模型端主要套件版本保持相同。它使用官方 CUDA 12.6 wheel，在現有驅動上先做實際運算驗證；技術相容不等於原研究數字已重現。完整依賴見 `requirements-torch214-lock.txt`。兩套環境分開保存，快取禁止交叉混用。

以下命令從 repo 根目錄執行：

```bash
bash codex/retrieval3090/setup_env.sh
dl/retrieval3090/envs/retrieval3090/bin/python -B codex/retrieval3090/download_assets.py --include-reconstruction-inputs

# 安裝同代框架的獨立對照環境；不覆蓋原環境。
RETRIEVAL_ENV_PROFILE=torch214 bash codex/retrieval3090/setup_env.sh
# 其他入口以 PYTHON 指向所需環境。
PYTHON="$PWD/dl/retrieval3090/envs/retrieval3090-torch214/bin/python" \
  bash codex/retrieval3090/run.sh preflight
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

## 獨立的本機平台開發

原基線未重現時，預設分析入口繼續拒絕執行。只有 [條件式平台協定](platform_plan.json) 規定的同代環境也未通過原驗收，才可另建 `platform-cache`。它保留原 `replay=MISMATCH`，核對已提交協定、來源檔案、同環境的固定批次重複及候選順序，再續算其餘候選。產物附 `platform-receipt.json`，不當作叢集重現或正式證據。

消費該快取時，`analyze2`、`benchmark`、`diagnostics d3-analyze`／`blind-export` 必須成對指定 `--platform-plan` 與 `--platform-receipt`；補充向量也必須通過同環境與完整檔案核驗。報告只比較相對同一本機CORE的差值。禁止把歷史指標與本機方法混比。

MCMR可先建與方法無關的原生圖文商品索引，再在固定政策的新輸出中指定 `--index` 復用；兩步使用同一套環境。索引不讀入池政策或重排分數：

```bash
OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1 \
  dl/retrieval3090/envs/retrieval3090-torch214/bin/python -B \
  codex/retrieval3090/mcmr_pilot.py --stage index \
  --data dl/retrieval3090/data/mcmr \
  --images dl/retrieval3090/data/mcmr/pilot-images \
  --out dl/retrieval3090/runs/mcmr-shared-index-new
```

以上GPU步驟仍由前述共用卡排程器啟動；可換同型同平台3090，實際卡號另記，不必占住原卡。

## 結果與獨立審查位置

本輪本機開發的結果、證據限制與下一輪建議見 [完整實驗報告](../reports/2026-11-16-retrieval-experiments-codex.md)；可追溯的精簡數字與輸入SHA見 [結果摘要](results/2026-11-16-local-study.json)。原基線重現、同平台相對探索與正式確認的狀態分開記錄。

固定政策的外部快取若使用 `mcmr_reuse.py`，須同時指定已驗收的 `--baseline`、`--index` 與事前鎖定的 `--selection-lock`；cache與analyze都經此wrapper入口，不能改用只認原生COMPLETE的分析入口。原始候選圖文資訊、batch及政策保持，重用配對與固定抽樣新算的收據一併驗證。

## 擴大的公開文獻資料

[codex] 2026-11-16。資料來源與適用範圍見 [資料調查與本機驗收](../reports/2026-11-16-retrieval-scale-data-survey-codex.md)，事前固定設定在 `literature_plan.json`。這條獨立入口不改舊面板的重現與評估流程。公開NegBench採來源圖Recall／MRR；FACTOR採單圖caption配對診斷，兩種真值分開。

共用原圖位於 `dl/retrieval3090/data/coco2017/val2017/`；原始CSV／FACTOR JSONL、來源收據與完整／抽樣清冊位於 `dl/retrieval3090/data/literature-development/`。圖片、查詢清冊、向量與分數不進Git。首次取得CSV按來源報告的作者Drive連結；`acquire_literature.py`可續下載其餘固定公開資產。重新建清冊會改變SHA時，應使用新設計與輸出目錄，不覆蓋原快取。

```bash
python=dl/retrieval3090/envs/retrieval3090-torch214/bin/python
"$python" -B codex/retrieval3090/literature_data.py verify

# 設定與程式須已commit；無合適卡會等待。
"$python" -B codex/retrieval3090/resources.py queue \
  --queue dl/retrieval3090/queues/literature-development.json -- \
  bash codex/retrieval3090/literature_run.sh \
  "$PWD/dl/retrieval3090/runs/literature-development"
```

整個流程依序做技術smoke、完整向量索引、CPU檢索分析、固定否定開發子集的CORE50、CPU重排分析。中斷後以相同清冊／設定／來源重跑即可續算；每個已完成batch保存state。四份向量NPY與重排分數是canonical產物，不再另存重複NPZ。完成與來源資訊由 `index-meta.json`／`cross-meta.json`記錄；結果只在報告位置保存，不能由技術smoke或執行中的進度推定研究效果。

## 後續新方法的固定比較

方法家族、尺寸、階段、固定資料與公平比較規則見 [公開開發基準v1](BENCHMARKS.md)。機械身份清冊為 `baseline_suite.json`，官方權重及程式釘版為 `benchmark_sources.json`。先以CORE及Qwen3-VL兩個家族，各自2B／8B、純向量及同尺寸50配對管線作目標矩陣；缺少實作、推論或獨立驗收的格子另列待完成，不因registry驗證通過就稱已完成baseline。

```bash
python3 -B codex/retrieval3090/benchmark_suite.py verify \
  --suite codex/retrieval3090/baseline_suite.json
```

本命令只唯讀驗固定檔案SHA與矩陣引用，不啟動GPU或重算效果；既有執行入口的模型／處理／環境驗證仍需保留。新增方法不改v1面板或指標，先固定本輪新方法與計算預算，再用相同query配對比較。
