# 物件連言檢索面板 v1：產生紀錄

[claude] 2026-11-16 台北。依凍結規格 [SPEC.md](SPEC.md)（commit `f8e3a2c`）產生，沒有改任何規則。只用 CPU 讀 COCO 標註，沒有跑任何模型。

## 位置與重現

- 資料：`/ssd4/cactus8603/icml/dl/claude-panel/conj-panel-v1/`（不進版控）。
- 命令（在 repo 根目錄；需要 pyarrow 的環境，這裡用 `dl/retrieval3090/envs/retrieval3090-torch214`）：
  1. `build_panel.py exposed --out dl/claude-panel/exposed-v1.json --source …`（八個來源見 `exposed.json` 的 `sources`）
  2. `build_panel.py build --annotations-zip dl/retrieval3090/data/coco2017/annotations_trainval2017.zip --exposed dl/claude-panel/exposed-v1.json --out dl/claude-panel/conj-panel-v1`
- 重建一次，所有輸出 SHA 相同。

## 輸入

| 項目 | SHA-256 |
|---|---|
| `annotations_trainval2017.zip` | `113a836d90195ee1f884e704da6304dfaaecff1f023f49b6ca93c4aaae470268` |
| `exposed.json`（25,481 個 ID；train2017 內 25,365） | `4d330fa1abfd1038175d0d0b251a56b4eb4f5cd2a57c46d520a5ec361994c229` |
| `build_panel.py` | `5cb0958c7865f6a6b3acc8805e53cd45e0d57b95bd13ece6ca4364b9e20521cb` |

## 輸出

| split | 圖庫 | 題數（k＝2／3／4） | 正例總數 | 檔案 SHA-256 |
|---|---|---|---|---|
| dev | 20,000 | **1,127**（334／402／391） | 1,873 | gallery `47cedc97…c5e`；queries `e7d5d817…ac9` |
| dev-small | 同 dev | **252**（84／84／84） | — | `7cb1cbbd…104` |
| test | 20,000 | **1,122**（334／400／388） | 1,904 | gallery `92199cc8…d60`；queries `4a65f7f2…406` |

`manifest.json` SHA-256 `c7ee5229236f4690bdfbc1ccde9c28f4f40f0656f05e313215fa67979e29f26b`，內含完整 SHA。

## 結構檢查（全部通過）

- dev 與 test 圖庫不重疊；test 圖庫不含任何已曝光影像。
- 每題正例 1–20 張、都在該圖庫內，與忽略集不重疊；同一 split 內所有題目的正例互不重疊。
- dev-small 是 dev 的子集。

## dev 描述（test 只記錄上表的題數與 SHA）

- 每題正例數：1 張 804 題（71%）、2 張 182 題、3 張以上 141 題，最多 17 張。
- 困難負例中位數 95 張；忽略集中位數 3 張，最多 292 張。
- 80 類都有用到；最常見的是 person（306 題）、chair（170）、dining table（143）。
- dev 圖庫中 8,536 張是先前曝光過的開發影像，規格允許。

## 影像（2026-11-16 取得）

用 [fetch_images.py](fetch_images.py) 取得：本機已有的 COCO 影像直接複製位元組，其餘從 `coco_url` 下載。每張都要能用 PIL 解碼，且尺寸與標註相同；逐檔 SHA 與來源寫入收據。

| split | 張數 | 來源 | 位元組 | 收據 SHA-256 |
|---|---|---|---|---|
| dev | 20,000，失敗 0 | 本機複製 8,282、遠端 11,718 | 3,262,286,794 | `37a72a527ffd021608fa0d4221187add536350f8cd3417bf02af6ace947bdec2` |
| test | 20,000，失敗 0 | 遠端 20,000 | 3,257,639,356 | `fec3229103a21c838cf7caaeb189d9802ba440acec1f8a75a43aca229eb9dfeb` |

- 位置：`dl/claude-panel/conj-panel-v1/{dev,test}/images/`，收據 `images-receipt.jsonl`，總表 `images-summary.json`。
- 本機複製來源：`dl/retrieval3090/data/coco2017-{anchor-rerank,rank-and,rerank}/images`。
- 位元組完全相同的影像：只有 1 組，都在 dev（ID 137715 與 242807），不跨 dev／test。規格沒有要求去重，保留原樣並在此記錄；評估時兩張各依自己的標註算正負。
