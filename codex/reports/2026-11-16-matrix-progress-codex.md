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

## 2026-11-16 21:19 更新

- 使用者補充資源原則：記憶體放得下即可使用非完全空卡，但避免三、四個人同時爭搶。由root統一改中央排程，同張卡最多一個本輪工作、全局先限兩個；2B依歷史CORE peak4.72GiB留餘裕，排程門檻8GiB；8B暫留20GiB。另檢既有計算程序數，避免加入已塞多個程序的卡。模型／輸入／評分凍結版本不隨排程改變。
- `25f7bd515875a0c698202d0047d7b7952fd6846b` 已凍結CORE2／8 MCQ：15,912原題、58,891 pairs、9,425圖片。root獨立從原始來源逐欄重建一致，10項CPU測試與實際payload SHA核對通過。舊佇列 `2344674/2344675` 與六個embedding等待佇列一併由中央接手；實際取消／新提交以SUBMITTED.log為準，不能把此處計畫當已執行。
- COCO development v2事前凍結於`613dc3f0028c536d992ab0726580d32adcb01579`：完整20,000圖片已齊、連言1,089題（small252）、否定237題（small197，僅診斷）。排除PACO全部COCO IDs、Karpathy test5k IDs、Half-Truths COCO IDs及原自建test20k，對這四組的ID交集皆0。原v1 dev原先分別重疊PACO1,044／Karpathy767／Half-Truths254張，原資料保留歷史曝光；後續方法選擇改用v2。
- v2的資料、選題、三值真值、20,000圖逐SHA／bytes／解碼尺寸均經非實作者獨立重建PASS。另比對Flickr原photo ID與COCO flickr_url，3張跨源交集均不在v2 dev；GQA VG→COCO官方映射尚缺，列unknown，不能宣稱跨源近重複已全部排除。
- `17e990b` 已凍結P2接線：15項CPU測試、兩環境builder/factory與四份完整資料adapter獨立核對通過。資格收據必須保留COCO/Flickr/NegBench三欄、至少兩個不同dataset的native官方metric在1pp內、綁完整模型spec及非產生者的重算證據。**沒有產生假PASS或可執行P2正式plan**；等待真實P0結果。
- 新增四個重排模型74檔41.232GiB全齊；額外embedding共12個獨立身份的實際payload／processor／權重鍵形狀CPU驗收完成，模型來源限制仍揭露。UniME原版未知checkpoint不以V2冒充；CLIC COCO轉檔來源缺口不隱藏。
- 主計畫資料已齊；額外原生CLIP資格需要的Karpathy COCO test5k版本尚在補圖：目前可重用2,125張，補缺2,875張。原始25,010句與UniME作者HF／前5句協定分開；不把CORE val2017的5k與此版本混用。
- MCMR完整MMEB-V3 archive已驗官方LFS SHA，其query／candidates與原來源逐byte一致。六模型原生融合adapter、完整context稽核與104,981商品／103,176圖片loader已齊，官方renderer逐筆獨立核對一致。CORE轉檔原腳本未公開，仍標非strict CORE exact。v2在首次GPU前修正一維逐列topk呼叫形狀與磁碟chunk64（模型逐商品batch1不變），9項CPU測試通過，兩plan待最終review／freeze。
- 目前仍無本輪正式GPU模型分數；P0效果及獨立重算、完整P2／P3／P4均未完成。未push。

## 2026-11-16 21:34 更新

- 中央排程已實際啟動：PID `2451282`，凍結commit `726115dc4b49c399473971a84421c029df6a7093`，狀態／事件位於 `dl/retrieval-matrix-dispatch/central-v1/`。14個作業包含原6個embedding各3集、CORE2／8完整MCQ、六模型完整MCMR。各命令保持其原模型凍結commit；所有工作共用global cap2與每卡1個本輪工作限制。
- 原八個queue均已確認`waiting/child=None`後退休，全部terminal cancelled，逐筆登記；中央與14個task也已登記，LOCKS已釋放。單一scheduler本身存活不等於模型已啟動；21:34快照14個仍pending。現在允許非空卡、utilization不限，容量足夠但已有≥2個compute PID的卡仍跳過，以符合避免多方爭搶的要求。
- `65f4f4850a82adf283b34dfddc319b72657562e6` 已凍結MCMR final v2與兩環境plan，source gate通過；一維topk、storage chunk64／實際模型batch1、完整context與metadata身分均已獨立驗收。Qwen2影像token上界已依真正runtime修正，VLM4463／GME保守19175、Qwen3 4072皆fit；舊audit與原tokenizer統計保留。
- 原Karpathy COCO補圖也已完成：5,000張、25,010原caption、820,306,454 bytes，2,125張復用＋2,875張官方新下載，零失敗。全圖另核官方ZIP中央目錄CRC／bytes、尺寸／JPEG／SHA及原caption/列序，4項來源完整性CPU測試root獨立重跑通過；不把此文字版本冒充UniME作者指定HF版。
- 額外12模型來源與接口正在非實作者審查；原生SigLIP2等兩benchmark資格計畫、通用CLIP資格schema與P3接線仍在各自新目錄CPU準備。UniME方向／數值路線及CLIC缺轉檔來源等差異未被當作PASS；P3仍須一個合格MLLM加一個合格CLIP，不能暗用兩個MLLM代替。
- 尚未產生正式P0分數或宣稱完整矩陣完成。未push。
