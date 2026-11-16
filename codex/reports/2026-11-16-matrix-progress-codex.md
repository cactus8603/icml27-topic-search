# [codex] 報告 64 執行進度

[codex] 2026-11-16 20:24（台北）。這是執行狀態與資料準備紀錄，沒有本輪新模型效果。

使用者要求完整執行報告 64，先 P0，開發限 dev／dev-small，主重排 K=100、補充 K=50/200。後續階段未被縮減成只完成目前首批。

## 已完成準備

- `c26b7f9f2a942f34480a0715762814d6697c1205` 凍結 `codex/retrieval_matrix/p0-qwen3-batch1.json`、程式、預登記及官方來源；26 項 CPU tests 和獨立官方協定審查通過。
- CORE／Qwen3-VL 各 2B／8B × COCO／NegBench／Flickr 共 12 格。完整 COCO、NegBench 各 5,000 圖與25,014句，保留否定資料的1個空 caption；Flickr為CORE指定HF版本的1,000圖/5,000句，保持其原caption字串，不改為另一套Karpathy字串。
- 官方 source／模板／池化／BF16相似度／torch.topk／pytrec 計分已核對；原環境不相容的 Transformers API 使用獨立固定4.57.3依賴，不改舊環境。2,734 個隔離依賴檔與所有模型 payload 有固定 SHA。
- 新模型取得67檔、31.578GiB，Qwen3-VL Emb2/8、GME2、VLM2Vec V2 DoRA與Qwen2-VL基底全齊。
- 新連言面板40,000圖齊；SugarCrepe++4,757題與全部所需圖片齊；Flickr完整公開版本齊；COLA10,623 union圖片全量offline驗收完成。
- MCMR官方archive解壓123,561個有效圖片；104,981商品所需103,176個互異圖片全部覆蓋，零缺。原archive另含1個0byte圖片，未被任何商品引用，獨立列為來源缺陷。

## 已提交的作業

四個本機共享 GPU 佇列，均先等至少20GiB空閒與低使用率、連續兩次確認，再做synthetic smoke與完整推論；不取消或干預其他工作。凍結版本皆為上述完整commit。

| 模型 | 佇列 PID | 目標 |
|---|---:|---|
| CORE2 | 2226876 | COCO、NegBench、Flickr |
| CORE8 | 2226877 | COCO、NegBench、Flickr |
| Qwen3-VL2 | 2226878 | COCO、NegBench、Flickr |
| Qwen3-VL8 | 2226879 | COCO、NegBench、Flickr |

狀態目錄 `dl/retrieval-matrix/runs/p0-qwen3-batch1-20261116/`，各模型有 `*-queue.json`、stdout log；程序存活與terminal狀態必須另核對，不能只看此表或舊state檔。`killtests/SUBMITTED.log`逐筆登記。送件不代表GPU smoke、研究推論、P0數字或獨立結果驗收已通過。

## 使用者補充決定

- [codex] 2026-11-16 20:23：使用者回答「沿用 COCO dev，COLA 僅測試（建議）」。因此不再要求缺少的COLA官方train，也不擅切COLA評估資料作開發。

## 仍需完成

1. 首批12格實際GPU smoke、完整推論、CPU官方計分與獨立重算，逐欄與論文比±1pp。
2. GME與VLM2Vec的隔離舊API、DoRA完整載入、GPU smoke及四個資料集P0。獨立模組 `codex/retrieval_matrix_qwen2/` 正在準備。
3. MCMR商品text renderer的CORE公開轉換腳本未釋出，不能把合理的自建renderer冒稱exact reproduction；MRR目標須為@10。需查原作者benchmark轉換並明示差異。
4. CORE ms-swift generative_reranker與其他重排器的MCQ驗收，VOC資料與缺權重取得；獨立模組 `codex/retrieval_matrix_rerank/` 正在準備。
5. train2017否定開發資料的固定建法、產生與品質驗收。
6. 報告64方法A的完整P2矩陣、方法B×兩個第一階段的K100主P3及50/200深度實驗、實測成本，最後P4按資料集彙整；未執行格不補0。

P0未通過的模型仍需保留完整失敗／差異紀錄，不按test效果調參追論文數字。只有主判準機械輸出、未實作對應模組者獨立核對後，才交使用者作研究判斷。

## 2026-11-16 20:37 更新

- `1c328a2c35c486f1768e9e9fea1c51f98dbb226d` 已凍結GME2／VLM2Vec-V2各三個完整資料集，共6格；CPU preflight與獨立重跑的21項測試、2,544檔隔離依賴均通過。V2補上官方DoRA而非裸基底；相容修正與上游版本差異見獨立審查報告。
- 新增等待空卡的佇列：GME2 `local3090-2264834`、VLM2Vec-V2 `local3090-2264835`。目前六個embedding模型合計18格已準備／送件，均尚未宣稱GPU smoke或效果完成。MCMR完整P0另外準備，不在18格內。
- 否定開發面板先於產生凍結於 `e6b0172`，產生在 `dl/retrieval-matrix/data/coco-negation-dev-v1/`：同完整20,000張dev圖，244題（正條件k2/3/4=119/89/36，每題另加1個NOT），small204題。獨立set實作從標註重建全部候選及hash round-robin、逐題全庫真值／ignore／hard-negative與模板均PASS，證據 `independent-audit.json`。
- 244題只作否定診斷，不冒稱達到1,000群或2pp檢定力；正式開發主面板仍是既有1,127題連言dev-full。若後續需對否定效果作相同強度的正式開發判斷，另立較大版本，不能改這份v1規則以湊題數。
- COLA與MCMR全量資料的獨立稽核已提交；舊MCMR pilot8,310圖逐SHA仍同原收據且沒有覆寫。Flickr提供CORE-HF原caption與Karpathy原caption兩份明確版本。
- 重排器CPU環境、VOC資料與固定MCQ pair清冊正在獨立模組準備；其餘方法A的模型來源／權重取得亦並行處理。P2、P3、P4仍未取得完整研究結果，目標保持未完成。
