# 檢索資料檢查、D3 機制診斷與 MCMR 外部資料協議

[codex] 2026-11-16（台北）。本報告只記錄資料協議、公開輸入核對與合成測試；沒有讀取本輪新快取的研究分數、舊 `26393` 原始結果或封存獨立面板，沒有執行 GPU。程式為 `codex/retrieval3090/diagnostics.py`，測試為同目錄 `test_diagnostics.py`。

## 1. 人工盲審匯出

完整開發快取通過基線驗收後，由主代理執行：

```bash
dl/retrieval3090/envs/retrieval3090/bin/python -B codex/retrieval3090/diagnostics.py blind-export \
  --cache <完整開發快取> \
  --image-root dl/retrieval3090/data/coco2017-anchor-rerank/images \
  --out dl/retrieval3090/runs/audit-20261116 --samples 150
```

抽樣規則先固定，分三層：基線精排前十名的標註負例、CORE 前五十名以外的標註正例、固定跨名次控制。每層最多五十筆，去重並以固定種子重新打亂。此抽樣是診斷用途；不能把原始陽性比例直接當全庫 PPV 或漏標率。

- `review/`：可交給標註者的匿名 PNG、HTML 與空白 `labels.csv`。圖片重新建立乾淨 RGB 畫布以移除 EXIF 和原始檔名，不提供方法、名次、抽樣層或正負標籤。
- `private/key.json`：匿名編號對應原查詢、圖 ID、抽樣層與原標籤；獨立目錄權限為 `0700`，不得交給標註者。
- `status.json`：初始狀態固定為 `AWAITING_HUMAN_LABELS`，完成人工標註數為零。

標註者只判斷全部指定物件是否可見，填 `yes/no/uncertain`，另註記微小、遮蔽或難辨識物件。人工標註不自動覆蓋原協議。對照時須分開「COCO 未標註」與「物件可見但未達原面積 2% 門檻」，後者不是資料遺漏。建議兩人獨立標註並對分歧另作仲裁；本次沒有執行或假造人工審核。

## 2. D3：固定圖庫、固定正負圖的 nested query family

目的不是單看條件越多、召回是否越差，而是檢查：同一對圖片的差別固定時，加入兩圖都符合的條件，整句嵌入能否維持對缺失條件的辨識。

設計已用公開 COCO 標註完成，不使用任何模型分數：

1. 圖庫固定為已曝光的 8,191 張圖；非 crowd 且單一實例面積至少 2% 才算類別存在。
2. 正圖 I 含四個物件 `c0,c1,c2,c3`；負圖 J 缺 `c0`，但含其餘三物件。J 以其餘物件共現背景及面積作標註式匹配。
3. 同一 family 的四句依序為 `c0`、`c0+c1`、`c0+c1+c2`、`c0+c1+c2+c3`。因此正負圖、背景、缺失物件與固定圖庫都不變，負圖在每個 k 都恰好缺一項。
4. 每個正圖最多選一個 family，同一四類集合不重複；固定種子 `2026111664`，上限 128 families。
5. 不假稱正例盛行率已固定：每句另記全庫正例數、類別頻率、正圖物件面積、正負圖共享物件面積差、恰缺一項的難負例數。全庫 Recall@50 仍會受正例盛行率影響；平衡正負圖的 margin 是主要機制讀數。

已備妥：

- `dl/retrieval3090/data/d3-nested-20261116/design.json`：128 families、512 句查詢。
- `texts.jsonl`：419 個唯一文字，包含真正完整句與單條件文字。
- `encoding_contract.json`：新文字向量的格式與正規化要求。
- 共有 107 個圖像連通元件。任何 family 共用正圖或負圖便歸同一元件，以此估不確定性，避免把重用圖片當獨立樣本。
- 圖庫 ID 順序摘要為 `95c0298d1eac3288b011128f9ee70e128e330dc1b059ff0f84fc53929363ba75`。
- 文字順序摘要為 `e4a4bc0c85ac5ef6541c35acee6c32d7f6b73f62a64d6965412359f49bab1ea2`。

準備介面：

```bash
dl/retrieval3090/envs/retrieval3090/bin/python -B codex/retrieval3090/diagnostics.py d3-prepare \
  --panel dl/retrieval3090/data/coco2017-anchor-rerank/manifest.json \
  --annotations dl/retrieval3090/data/coco2017/annotations_trainval2017.zip \
  --out <新的設計目錄> --max-families 128
```

GPU 階段只需補編碼這 419 條文字，使用與原快取相同的兩個模型、提示、精度及單位向量正規化。NPZ 欄位為：

- `texts`：Unicode 字串陣列，順序必須與 `texts.jsonl` 完全一致。
- `core_embed_text`：`419 × 2048`。
- `cs_clip_text`：`419 × 512`。
- `metadata_json`：Unicode scalar JSON，`schema="d3-holistic-encodings-v1"`；必須綁定 `design_sha256`、`texts_sha256`、`gallery_ids_sha256`、兩模型的 `model_revisions`、`model_config_sha256`、`normalization="unit_l2"`。`source/freeze/commit` 都必須是完整 40 位小寫十六進位 Git revision，且 `source/freeze` 須與影像快取 `meta.bundle` 相同；補編碼的 `commit` 可較新。模型 revision 同既有快取，模型設定 SHA 為 `862591c60c5609f16727d902993df0cba9cab96ded72611d69d263a02fe4ad1d`。

分析與盲審匯出都要求完整開發快取的 `meta.json` 為 baseline replay PASS，且模型 revision 與設定 SHA 符合原值。分析另核對補充向量的完整來源；family/k 使用明確鍵值配對，設計即使重排也不會把 k=4 減到別的 family。

**必須編碼完整句。單條件向量相加不能取代整句向量。** 分析會另把單條件向量相加再正規化作比較，以區分整句模型本身的現象與簡單可加表徵的行為。

CPU 分析介面：

```bash
dl/retrieval3090/envs/retrieval3090/bin/python -B codex/retrieval3090/diagnostics.py d3-analyze \
  --design dl/retrieval3090/data/d3-nested-20261116/design.json \
  --cache <完整開發快取> --embeddings <新文字向量.npz> --out <新的分析JSON>
```

### 固定的分析讀法

主要診斷為 **CORE 整句、固定正負圖的標準化 margin，在 k=4 相對 k=1 的配對差**；margin 為正圖分數減負圖分數，標準化分母為同一查詢在固定圖庫上的分數標準差，不能把它稱為估得的噪聲或理論 SNR。以圖像連通元件重抽 2,000 次估區間。CS-CLIP、原始 cosine margin、成對答對率、目標圖名次和全庫 Recall@50 為次要讀數。

原始 margin 與標準化 margin **各自**另作 family 去均值回歸，控制 log 正例數、類別頻率、面積、log 難負例數及共享物件面積差。只有設計矩陣滿秩才報 k 係數，否則明列不可識別；係數區間以連通元件重抽一千次，至少九百五十次滿秩才報。只有一個連通元件時不給群集區間，少於二十個元件時註明區間可靠性有限。

負的 margin 差可支持這份控制設定下的經驗性稀釋，不能直接證明所有資料／模型皆遵守 `1/√k`，也不能單憑 Recall 下降宣稱因果。此圖庫已曝光，所有結果屬開發診斷。若需驗證原 `26393`，必須另走讀取登記，本支線未開啟該原始輸出。

## 3. MCMR：實際核對與已完成的準備

官方來源：[GitHub](https://github.com/EIT-NLP/MCMR)、[HF 資料](https://huggingface.co/datasets/Lux1997/MCMR)。程式版本固定為 `8bf618a89cb41c1e12f1bad640eb12bcf0ef31a0`；資料版本固定為 `12d53f4063804ee6a976024c7375ab2fbe4368c8`。

已實際下載兩份公開 metadata，並驗固定 Git blob SHA-1 或 LFS SHA-256：

| 檔案 | 位元組 | 完整性 |
|---|---:|---|
| `query.jsonl` | 1,369,474 | SHA-256 `28925c2d015a75fda763d7bd3092bfd01bf96c8d3e8cd2397702e1fda049e68c` |
| `candidates.jsonl` | 148,279,983 | LFS SHA-256 `05242fb414333a99c562a986edb8b2d1997db1726f731b6131ab97bf557fcbc0` |

共 **104,981 個候選、3,997 條查詢**，沒有查詢的正例 ID 缺於候選 metadata。

實際 schema：

- query：`qid`、`query`、`pos_ids`。
- candidate：`candidate_id`、`images`、`title`、`description`、`features`、`price`、`Date First Available`、`domain`。
- `images` 中第一張圖的 `url`／`large`／`image` 對應官方腳本使用的圖；檔名取 URL basename，**不是 candidate ID**。

已核對官方 [LLaVE](https://github.com/EIT-NLP/MCMR/blob/8bf618a89cb41c1e12f1bad640eb12bcf0ef31a0/eval/retrieval/llave.py)、[GME](https://github.com/EIT-NLP/MCMR/blob/8bf618a89cb41c1e12f1bad640eb12bcf0ef31a0/eval/retrieval/gme.py)、[VLM2Vec](https://github.com/EIT-NLP/MCMR/blob/8bf618a89cb41c1e12f1bad640eb12bcf0ef31a0/eval/retrieval/vlm2vec.py)、[CORAL](https://github.com/EIT-NLP/MCMR/blob/8bf618a89cb41c1e12f1bad640eb12bcf0ef31a0/eval/retrieval/coral.py) 與 Qwen pointwise 腳本：原生協議是 **文字查詢 → 商品圖與商品文字共同輸入**。檢查過的腳本沒有直接可開啟的 image-only 切換。查詢包含材質、保養等未必能由圖片得知的條件；把商品文字丟掉會改變任務與可用資訊，必須標為純圖消融，不能和原生結果直接比較。若延伸現有方法，應另外實作相同商品圖文資訊的固定索引與重排適配器。

資料夾 `dl/retrieval3090/data/mcmr/` 已有原 metadata、正規化 JSONL、`preparation.json` 與 `pilot_manifest.json`。pilot 以固定 hash 選 128 條查詢，保留它們的全部所需正例，加 8,192 個固定 hash 候選，共 **8,320 候選**。這是開發子集，不是完整原生 MCMR 評估。

重現 metadata 準備：

```bash
REQUESTS_CA_BUNDLE=/etc/ssl/certs/ca-certificates.crt \
dl/retrieval3090/envs/retrieval3090/bin/python -B codex/retrieval3090/diagnostics.py mcmr-prepare \
  --download --out <新的metadata目錄> --pilot-queries 128 --pilot-negatives 8192
```

### 圖片取得與安全抽取

圖片是 `images.tar.gz`，不是 ZIP。固定下載 URL：

`https://huggingface.co/datasets/Lux1997/MCMR/resolve/12d53f4063804ee6a976024c7375ab2fbe4368c8/images.tar.gz`

- 壓縮檔大小 **2,263,663,574 bytes**。
- LFS SHA-256：`7cbda9a5fbdba1240590885c576b4b31dccd047f14d2cc5b8a8e87c94bda2b86`。
- 實際只探測前 **1,048,576 bytes**，收到 HTTP **206**、`Content-Range: bytes 0-1048575/2263663574`、`Accept-Ranges: bytes`。
- 首段 tar header 確認 entry 為 `images/<Amazon URL basename>`，含 JPEG 與 PNG，沒有以 candidate ID 命名。
- Range 可以續傳壓縮檔；由於是單一 gzip 串流，且未找到候選至壓縮位移的索引，不能據此對任意候選隨機下載單張。直接下載 metadata 中的 Amazon URL 是另一個來源，但沒有原 tar 的逐圖 SHA 可立即證明位元相同。

建議完整下載原 tar，先驗整檔 SHA，再只抽 pilot 所需檔案。已實作以下介面，使用串流讀取與明確檔名清單，不使用無限制 `extractall`，核對完成後留下逐圖 SHA 與候選對照收據：

```bash
dl/retrieval3090/envs/retrieval3090/bin/python -B codex/retrieval3090/diagnostics.py mcmr-extract \
  --archive <已下載images.tar.gz> \
  --pilot dl/retrieval3090/data/mcmr/pilot_manifest.json \
  --out <新的pilot-images目錄>
```

本支線沒有下載完整圖片壓縮檔。GitHub 程式是 Apache-2.0；HF dataset card metadata 未宣告資料授權，不能沿用舊摘要的 CC BY-NC-SA 宣稱。

## 4. 驗證與待辦

十四項合成測試已通過，涵蓋 nested 邏輯、相同正負圖及缺失條件、固定種子、面積／crowd 規則、連通分群、群集加權、不可識別回歸、完整句向量與相加基線區別、盲審公開／私有隔離、MCMR 正例隔離、快取及補充向量來源綁定、family 重排和缺列拒絕與封存路徑拒絕。

尚待主代理：完整開發基線通過後匯出真實盲審包；新增 419 條文字 GPU 編碼並依凍結協議分析；安排真實人工審查；取得 MCMR 圖片後進行明確相同輸入資訊的外部比較。以上待辦都不是已取得的研究結果。
