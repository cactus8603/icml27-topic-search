# 多條件檢索｜24 GB 開發規範（dev-3090）

[claude] 2026-11-16（台北）撰寫，交由 Codex 執行。這是開發工具，產出只作方法開發參考，不構成正式證據。正式判定仍走預登記、凍結、H200 送件與獨立複核。 [需要的檔案與資料集](DATA.md) · [開發計畫](../../../../reports/2026-11-16-62-retrieval-3090-dev-plan.md) · [目前結果](../../VERDICT.md) · [協作規則](../../../../COORDINATION.md)

## 目標

在已曝光的 8,191 圖、136 查詢圖庫上回答兩件事：

1. CORE 前 50 名漏掉的正例在哪裡、為什麼漏掉（D1），以及名單內排錯的是哪種圖（D2）。
2. 重排次數維持每題 50 次，能否從 CORE 前 100／200 名挑出更好的 50 張（M1），或按題目分配不同預算（M2）。

已知背景：CORE 重排 mAP **54.53%**，Recall@50 **73.77%**，31 題前 50 名無正例。CORE 前 100／200 名的召回 **79.66%／91.54%**。交錯 CS-CLIP 入池（`26806`）未定。136 題中有 **105 題只有一個正例**，Recall@50 大多是 0 或 1，解讀時要注意。

## 程式

| 檔案 | 用途 | 資源 |
|---|---|---|
| `check_host.sh` | 檢查主機是否有全部需要的檔案、模型與環境，加 `--hash` 會核對固定檔案的 SHA-256 | CPU，約 40 秒 |
| `bundle.sh` | 用 Git 取出 26806 用過的凍結程式（共用腳本與前置階段取自 `2daedf5`，recall-budget 階段取自凍結版 `569b6c0`），加上本資料夾的檔案，設定離線環境後執行 | — |
| `cache_scores.py --preflight` | CPU 預檢：面板與 26601 收據、模型快照、用 COCO 類別規則重建全部相關標註 | CPU，約 1.5 分鐘 |
| `cache_scores.py --run --out DIR` | 唯一的 GPU 步驟。CORE-EMBED、CS-CLIP 的全庫影像、整句與單條件向量；CORE-RERANKER 對每題 CORE 前 200 名打分。最後檢查 26806 基線能否重現 | 1 張 GPU，預估峰值約 6 GiB |
| `analyze.py --cache DIR` | 純 CPU、只需 numpy。基線重現、全部重排上限、D1、D2、M1、M2 | CPU，數秒到數十秒 |
| `test_analyze.py` | 合成資料單元測試，並在 bundle 內與 26806 的 `score.py` 逐查詢比對 | CPU |

```bash
cd topics/compositional-retrieval/code/dev-3090
bash check_host.sh --hash                              # 檔案、模型、環境齊全
bash bundle.sh test_analyze.py                         # 7 項測試
bash bundle.sh cache_scores.py --preflight             # CPU 預檢
bash bundle.sh cache_scores.py --run --out /work/ChihChungHsu/cactus8603/icml/runs/retrieval-dev/<主機>-<YYYYMMDD-HHMM>
bash bundle.sh analyze.py --cache /work/ChihChungHsu/cactus8603/icml/runs/retrieval-dev/<同上>
```

`cache_scores.py --run` 的輸出目錄必須不存在。輸出包含 `vectors.npz`、`cross.npz`、`panel.json`、`meta.json`，合計約 0.2 GiB，放在 checkout 外，不進 Git。

## 24 GB 主機需求

- **路徑**：凍結程式寫死 `/work/ChihChungHsu/cactus8603/icml`。3090 主機需要掛載或 symlink 同一路徑。`bundle.sh` 不接受其他根目錄。
- **環境**：`envs/vllm033`（Python 3.12、Torch 2.14+cu130、Transformers 5.20）加上 `envs/t2irare_extra_cscomp`（open_clip）。驅動要支援 CUDA 13.0（R580 以上）。
- **需要的檔案**：環境、三個模型、開發圖庫、COCO 標註和 26601 收據，合計約 24 GB。完整清單、固定版本與複製指令見 [DATA.md](DATA.md)。模型 snapshot 是 symlink，複製時要解開。
- **主機 RAM**：至少 32 GiB。COCO 標註以串流讀取，主要的 RAM 用量在模型載入。
- **精度**：3090 是 Ampere，支援 BF16，與 H200 的設定相同。

## 驗收：先重現基線

`cache_scores.py` 在快取寫完後，用 26806 的 `check_replay` 驗證 CORE 基線。必須滿足以下全部條件，`analyze.py` 也會再核對一次：

- mAP 54.53、固定前 50 上限 74.02、Recall@50 73.77，各在 ±0.10 個百分點內。
- 前 50 名無正例的題數剛好 31。

重現失敗時程式會非零退出，但快取仍會寫出，方便查差異。常見原因是 kernel 或 batch 組成不同造成 BF16 數值差，會讓前 50 名邊界或重排平手的順序改變。查清楚前不要解讀 M1／M2。

為了能逐位重現，整句查詢、影像與前 50 名交叉配對的編碼順序和 batch 大小都與 26806 相同。第 51–200 名的配對另外接在後面打分，不影響前 50 名的分數。

## 資料使用規則

- **可以用於開發**：`coco2017-anchor-rerank` 的 8,191 圖、136 查詢（26601／26806 已曝光）。
- **不能碰**：`retrieval-independent-panel` 的封存面板（8,192 候選、120 查詢，清冊 `d7981026…`）。它只用在凍結後的正式驗證。
- 在這個圖庫上挑出的策略和權重，都帶有選擇偏差，不能直接拿來宣稱效果。
- 開發結果寫在 `codex/reports/` 或帶編號的報告，不寫進 README 或 PREREG。

## 分析內容

**基線與上限。** CORE 前 50 名重排，以及「把 CORE 前 d 名全部重排」（d＝50、100、150、200）的 mAP、召回與無正例題數。全部重排前 200 名的成本是每題 200 次，用來表示 M1 最多能拿回多少。

**D1 召回診斷。**
- 漏掉的正例在 CORE 排序中的名次分布：51–100、101–200、201–500、500 以後。
- 31 題無正例查詢中，第一個正例的名次。
- CORE 前 50 名的組成：兩個條件都符合、只符合一個、都不符合。
- 漏掉的正例中，比較弱的那個條件在全庫的名次比例（四分位數）。如果大多是其中一個條件很弱，就支持「拆條件」的方向。

**D2 名單內診斷。** 重排後的前 50 名裡，排在最後一個正例之前的負例有幾張，分成只符合一個條件和都不符合兩類。

**M1 入池策略。** 從 CORE 前 100 或 200 名用便宜分數挑 50 張，再用同一個 CORE-RERANKER 排序，名單外的圖保留 CORE 原順序。每個條件文字都在全庫上做 z-score 標準化。

| 策略 | 分數 |
|---|---|
| `core` | CORE 原名次（等於基線，用來檢查） |
| `cs_full` | CS-CLIP 整句 |
| `min_cond_core`、`min_cond_cs` | 兩個條件中較弱者 |
| `llr_core_t{τ}`、`llr_cs_t{τ}` | Σ log σ(z_c／τ)，τ∈{0.5, 1, 2}。對應深查第 5 節的對數概似比加總 |
| `fuse_<策略>_w{w}` | CORE 名次百分位與上述分數百分位的加權，w∈{0.25, 0.5, 0.75} |

每個策略都輸出 mAP、Recall@50、無正例題數、相對基線的配對差（按查詢群重抽 10,000 次，seed `2026111662`），以及佔「全部重排」增益的比例。

**M2 預算分配（初版）。** 依 CORE-EMBED 第 50 名到第 200 名的分數落差分四等分，落差最小（最不確定）的四分之一給 75 次，再來兩組各 50 次，最確定的四分之一給 25 次，平均仍是 50 次。

**D3 條件數稀釋。** 這版程式沒有實作。headroom 作業 `26393` 已依舊設定計算過 COCO val2017 k＝1…4 的描述性 E2，先在 `PREREG_headroom.md` 登記讀取，再讀這份輸出，確認缺什麼後才另寫程式。

## 開發門檻與下一步

| 結果 | 下一步 |
|---|---|
| 有策略 Recall@50 至少 +5 點、mAP 至少 +2 點且配對區間下界大於零 | 只凍結一個策略和它的參數，寫 PREREG，協調取得封存面板的 JPEG，在 H200 正式驗證 |
| mAP 有增益但召回沒有 | 回頭看 D2，考慮改善名單內的排序 |
| 所有策略都在 ±1 點內 | 用 D1 的結論判斷是否換方法，或回報使用者考慮停損 |

這個門檻只用來決定要不要進正式驗證，不是正式判準。正式判準沿用 `retrieval-independent-panel` 的寫法：全庫 mAP 點估至少 +2 點，且 95% 下界大於零。

## 已知限制

- GPU 步驟只在 CPU 上做過預檢，還沒有在 GPU 上實際跑過。
- 第一次在 3090 上跑時，請記錄 `meta.json` 中的峰值顯存和各階段耗時，補進 `codex/HARDWARE.md`。
- 交叉配對共 136×200＝27,200 對。26806 在 H200 上打 6,800 多對，總共約 6 分鐘，所以這次在 H200 上估計 20–30 分鐘，3090 可能要 2–3 倍。
