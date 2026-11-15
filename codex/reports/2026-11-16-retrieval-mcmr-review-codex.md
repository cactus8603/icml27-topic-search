# MCMR 外部 pilot 獨立審查

[codex] literature_check，2026-11-16（台北）。未實作 MCMR runner／reuse wrapper；只讀產物並做 CPU 核驗，未改政策或子集、未跑 GPU、未讀 blind/private 或封存資料。首次直接讀 `mcmr-selected-torch214-20261116/analysis.json` 與方法 raw scores：**06:05:54.075**；此前於06:04–06:05收到 root 摘要。05:39:20.766 曾依授權載入 baseline raw cross 作完整性驗收，當時未計 AP／Recall 或外部成效。

**產物及計算完整性通過；固定政策外部轉移嚴重退步。** 未見足以解釋落差的索引、配對、公式、模型檔案或統計錯誤。這是指定 COCO→MCMR adapter／政策的失敗，不能推成所有 AND／LCSE 方法或整個研究方向皆失敗。

## 已獨立重算的結果

| 指標 | 原生圖文 CORE50 | 固定政策 | 配對差 [95% CI] |
|---|---:|---:|---:|
| mAP | 89.5303% | 51.1820% | **−38.3483 pp [−47.3911, −29.5097]** |
| Recall@50＝入池 Recall | 99.2188% | 52.3438% | **−46.8750 pp [−55.4688, −38.2813]** |
| 前50無正例題數 | 1 | 61 | 增加60題 |

**60題丟掉原有命中、0題新增命中、67題雙方命中、1題雙方未命中**。AP為9題改善、61題惡化、58題相同。全部128題恰有一個正例，所以逐題 AP=`1/positive_rank`，本子集 mAP 等於 MRR，Recall@50 等於 query hit rate；評估完整8,320商品排名，非截斷AP。

唯一 CORE50 漏例原名次為 **232**，前200支援集合只有原本已命中的127個正例，因此本 pilot 沒有 top200 入池 Recall 的正向 headroom。60個被移除正例中，52個原本在CORE前五、53個在前十，並非僅邊界替換。仍有入池內排序的mAP空間，不能說一切改善都不可能。

## 完整性與統計核驗

- 政策 **`m1/top200/fuse_logsigmoid_cs_t1_w0.75`** 於 **05:45:39.499** 依COCO結果鎖定，未在MCMR再選法。已驗policy檔、selection lock、COCO report SHA（只計雜湊，未解析內容）及platform plan SHA。
- 在shared locks下，以已提交 `baseline_bundle`／`validate_receipt` 驗source bytes、config／revisions、完整runtime identity、adapter、index receipt與最終產物。baseline 6,400對確為各題實際CORE前50；CORE candidate／full向量逐值等於已驗共用索引。原生分析入口拒絕 `REUSE_COMPLETE`，須經wrapper驗receipt。
- 從raw vectors自行重算全池cosine／z-score、clause log-sigmoid和、平均平手percentile、0.25 CORE／0.75 CS融合、選50，與frozen admission逐元素相同；再自行按cross及CORE tie-break生成完整排名，核逐題AP／Recall與全部summary。
- **8 common pairs** 以batch1新算，fresh／expected hash完全相同；通過才重用。最終 **9,883對＝6,400重用＋3,483新算**，另有8次驗證。共同分數與baseline逐值一致，receipt／checkpoint／producer binding均過。這是小樣本重複與分數重用，並非跨卡完整重跑全部baseline。
- 清理前獨立驗CS三個checkpoint：image `8320×512`、full `128×512`、condition `374×512`，皆float32／finite／unit、逐值等於final；state的完成數、shape、dtype、SHA及items-order fingerprint均正確。固定CS checkpoint每個下載檔案大小及SHA再驗通過。root其後依此驗收清掉三個重複NPY，保留final、state、cross與清理收據。
- 獨立重算10,000次paired bootstrap，seed **2026111664**；所有差值、區間及SE重現。8,320商品有8,310個唯一JPEG SHA（10組各兩商品同圖），但128個positive product IDs及positive JPEG SHA均唯一，合併後仍128群。一個正例另有同圖商品別名，不連接其他query正例；未重標或刪樣本。區間仍不涵蓋所有語義相依、共同負例或候選圖庫抽樣變異。
- 原baseline controller為 **143／interrupted**、child exit未知，完成產物另經獨立驗證；不能寫成baseline OS exit0。selected queue實際記錄completed／returncode0。產物成功與方法效果失敗分開。

## 原生資訊與研究邊界

CORE candidate index及reranker皆使用圖片，以及保留的title、description、features、price、Date First Available文字（domain不進此renderer）。不作字元裁切；CORE joint／cross token上限4096／8192，超限即報錯。query沿用固定COCO system prompt，因此保留原生圖文資訊不等於逐項重現官方MCMR全部adapter／完整基準。

CS分量只看商品圖片，沒有商品metadata；query依句號／分號等規則切clause，沒有原子條件或coreference保證。112題三clauses、11題兩、4題四、1題一，共374唯一clauses。收據中的5個CS整句超77 tokens不進本政策score；實際374個clause皆未截斷，不能把這5題整句截斷當作落差原因。政策不是LLR，也沒有使用LCSE。

結果顯示，這個視覺clause融合強烈覆蓋原生圖文CORE已有的正例，未成功轉移。它同時改變encoder／checkpoint、query表徵與可用資訊，不能據此斷言metadata缺失是唯一原因。入池Recall在rerank前已下降，cross分數重用無法造成或修復被排除正例的損失。

此為128query／8,320商品的固定外部開發pilot，不是完整MCMR官方SOTA比較或正式確認。每arm各50次cross不代表相同端到端成本；額外CS索引、編碼與載入尚有成本。應保留這份負轉移證據，不在同一pilot再挑政策後改稱獨立確認，也不將單一固定adapter失敗外推為AND研究普遍失敗。

## 主要追溯指紋

| 產物 | SHA-256 |
|---|---|
| selected analysis.json | `18a87fb19f1f5f82233a8499df8bf043820d209a2263a5a5a3a7fbfc6aa8aece` |
| selected cache.npz | `8af067f402eecd178b7a69207251528b5f9b2238d7e57def67b893d333686d05` |
| reuse-receipt.json | `dbafd504bf35985af1b3088153182ec1ff101ccc0252087a39363e37e417baf7` |
| selected-policy.json | `e48c8da2ba14c85b3698ed0ade4c1c8c60b3c4221e94de1d3a24b0fff1d0dcbe` |

producer／wrapper execution commit：`1f9caa2b10b2864689592393409decd4092c9cd7`。結果與收據均維持 `formal_evidence=false`。
