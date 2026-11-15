# Controlled Images B 空間關係檢查

## 問題與範圍

給定同一對物件、相反的空間關係及兩張照片，CS-CLIP 加入正確物件詞的無關係 anchor 後，是否比加入打亂的 anchor 更常把查詢所描述的照片排前？這次檢查兩張照片之間的排序及 anchor 的作用。原本 COCO 複合查詢的 mAP 結論另見既有預註冊及報告。

## 凍結資料

使用作者公開的 What's Up With Vision-Language Models? Controlled Images B。作者說明每列 caption options 的第一句是正解。論文描述 412 張；官方連結實際提供 408 張 JPEG 及 408 列標註，程式要求兩者逐一對應，並驗證每列四句只改關係、照片檔名符合正解。採用釋出的全部 102 組，每組使用同一對物件及四個標籤。每組包含左／右及前／後各一組相反關係，每組相反關係測兩個方向。共 204 組照片配對、408 筆有方向的測試。 [作者資料說明](https://github.com/amitakamath/whatsup_vlms#datasets)、[論文 §2.1](https://aclanthology.org/2023.emnlp-main.568/)

使用官方 B 影像 tar.gz（65,817,200 bytes）及標註 JSON（86,532 bytes），存於共用區 `icml/data/whatsup-controlled-b/`。Git 只記錄釋出檔案及個別影像的 SHA-256、物件組別、固定 donor 對應和重疊檢查摘要。原圖在程式中從 tar 串流讀取。作者儲存庫有 MIT 軟體授權；照片留在本機研究資料區。 [作者儲存庫授權](https://github.com/amitakamath/whatsup_vlms/blob/main/LICENSE)

外部重疊比對涵蓋全部 5,000 張 COCO val2017 影像與固定 revision 的 1,437 張 Half-Truths 輸入。用 SHA-256 比對原始位元組；近似比對用 Lanczos 縮圖的灰階 DCT pHash（32×32、前 8×8 去 DC、以中位數二值化）與水平 dHash（9×8），兩者的 Hamming 距離都 ≤6 時須人工確認及重新凍結。兩種比對均找不到跨資料來源的候選重疊。B 內部的 408 張均有不同 SHA-256。同一室內、背景及重複物件是作者資料設計的一部分，因此 B 內部以已驗證的物件組別和正反標籤分組。比對範圍是現有實驗輸入；模型訓練資料是否涵蓋這個公開 benchmark 未知。

## 固定打分

- CS-CLIP 原始分數是凍結 encoder 的 caption／image 向量 cosine。正確 anchor 為同一物件組固定的 `a {subject} and a {object}`，保留物件名稱及順序。打亂 anchor 用同句型及另一組互斥物件，以固定種子 2026111511 做一對一最大匹配。每組兩個 anchor 分別在四個方向共用。
- 正確及打亂 anchor 都沿用既有 `score + λ × (score − anchor_score)`，固定 λ = 0.125。這次只用合成資料驗證程式。模型版本、前處理及原始分數重用既有固定配置。CORE-EMBED 用完整句子對相同兩張照片排序，列為描述性參考。
- 每個方向算正解照片的分數減相反關係照片分數。正、平手、負分別記 1、0.5、0。每組四個方向先平均，主效果是正確 anchor 和打亂 anchor 的配對勝率差，以百分點呈現。用固定種子 2026111512、有放回地重抽完整物件組 10,000 次，報告雙尾 95% percentile CI。固定資料、donor 配對及場景共享物件限制這個區間的推廣範圍。
- 描述項包含各方法整體勝率、平手、各關係勝率、正反都嚴格正確的比例，以及正確 anchor 相對原始 CS-CLIP 的配對差和區間。正確與打亂 anchor 的有符號邊際差在反向查詢中相消，所以主效果使用排序勝率。主要差值會與原始分數一起解讀，區分原始排序改善與打亂 anchor 的損失。

## 執行與留存

使用 CPU `--preflight` 重建資料清單、摘要、donor 配對並比對凍結 manifest，確認兩個離線模型的固定檔案都存在。正式執行在一張 H200 上依序載入兩個 encoder，上限 30 分鐘。從指定 Git commit 封存執行程式，輸出到新的 `results/controlled_spatial/<job_id>/`，留存 frozen commit 與彙總 JSON。

- [codex] 已讀 `results/controlled_spatial/26514/summary_controlled_spatial.json`，2026-11-15 07:17 整合核對與 Astra 獨立複核。凍結版 `e4ce774`，本次排序比較沿用原設計。
