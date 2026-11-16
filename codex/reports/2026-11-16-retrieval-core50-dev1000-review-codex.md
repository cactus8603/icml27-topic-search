# [codex] NegBench dev1000 CORE50：獨立效果與完整性驗收

[codex] 2026-11-16 16:30（台北時間）。審查者未實作本輪producer。只讀已完成NegBench公開開發產物，以純CPU獨立核算；只新增本報告，未改來源／plan／選法，未GPU、未清理canonical raw，未讀封存、MCMR新效果或8B效果。

**驗收PASS：保存產物與固定協定一致；CORE50在本次來源圖標註下使早期排名下降。** 可以作為後續固定suite的一個版本化baseline，不能把這次公開開發結果改稱新獨立確認。

## 首次讀取時間

以下均為2026-11-16台北時間；主路徑為 `dl/retrieval3090/runs/literature-dev-20261116/main/`：

| 產物 | 首次實際讀取 |
|---|---|
| `cross-report.json` | 16:26:56.086501 |
| `cross-per_query.json` | 16:26:56.087042 |
| `cross-meta.json` | 16:26:56.089267 |
| `cross50.state.json` | 16:26:56.089689 |
| `cross-provenance.json` | 16:26:56.105772 |
| `cross-pairs.json` | 16:28:56.735425 |
| `cross50.npy`數值 | 16:28:56.769800（此前僅SHA bytes核驗，沒有解讀數值） |

既有cheap vectors於前次審查已讀；本輪確認index仍COMPLETE，canonical arrays及state的SHA與既有index receipt一致。未取得writer lock，也未改動任何原始檔。

## 固定輸入與來源

- 凍結 `literature_plan.json` SHA256：`f0685e544409d0082f642d3ad36ce5b85323465f7159c1880549f396093fb2e2`；prepared receipt：`86814f40da7a74c7f606cc99ccc4d6bae0a7229ab1eb1008b884cb885638525a`。
- 全COCO val2017圖庫5,000；固定 `negbench-negated-dev1000.json` 的1,000條query／1,000個來源JPEG群，每query只認其官方source image為positive。Query／JPEG群及caption索引未更換，未按admission或gold篩query。
- Admission為CORE整句embedding的原top50；reranker為 `Alibaba-NLP/core-reranker-2b`，revision `64e1a36cd6ecace459a4c7cd8f7d0ef5d6fee30b`。Scorer使用凍結的yes−no log-odds；沒有改成新prompt、sigmoid閾值或另一模型。
- Cross與index的完整environment identity相等：RTX3090、Torch2.14.0+cu126、CUDA12.6、SDPA、CORE bfloat16，原cross batch4。模型輸入只有verbatim query text和candidate RGB影像，不讀objects／parse／gold作模型輸入或admission。
- 四份source-file SHA與producer `cbd545ac96e4978454c41a78f78b75419d413b77`及cross記錄commit `5e7b6f01551c0a60b379825e9f6f38722d450574`均一致；後續其他檔案提交沒有改變本次執行的四份凍結來源。Cross provenance、meta、index、plan、panel、model revisions及checkpoint fingerprint的綁定通過。

## 獨立重建與完整性

未呼叫producer的shortlist、rerank、metrics或bootstrap helper。從canonical float32 query／image vectors重算全部1,000×5,000 cosine，以lexicographic score／incoming-rank規則排序；重建每條query的top50，與保存的50,000個 `(query_index,candidate_index)` pairs逐值完全一致。

以保存的50,000個float32 log-odds，獨立重排各50項prefix，再接回原rank51–5000的tail。驗得：

- 1,000個完整結果各含5,000個candidate且恰為原圖庫permutation；tail逐項不變。
- 1,000個embedding／CORE50來源排名與 `cross-per_query.json`逐題完全相同。
- 原top50未含來源圖的**31題全部保留**，各自tail rank不變；它們仍消耗每題50對，合計1,550 pairs，沒有使用gold作oracle跳過。
- `cross50.npy`為50,000個finite float32；state完成50,000、shape／dtype／SHA一致。**12,500個batch records全部連續、各4筆**；沒有缺口或重複，active seconds與各batch秒數加總一致。
- Checkpoint fingerprint由完整cross identity、50,000 items與batch4重新計算，與保存值相同。未另跑fresh model forward；本驗收證明保存分數與來源／pairs／state的完整一致性，不冒稱新增GPU重複推論。

同分規則需保留：cheap同分按原numeric-image-ID圖庫順序；cross同分按**進入該prefix的原cheap rank**，不是重新按image ID。998題在50項中至少有一組exact分數tie，46題的最高分有tie，沒有整個50項全同分的query；上述規則已用合成tie及實際完整重建核實。沒有因看到結果更換tie-break。

## 指標與paired bootstrap

全部指標以百分比列出，差值為CORE50−embedding的百分點。獨立按1,000個JPEG群、seed2026111602、有放回抽樣10,000次，重算percentile95%CI及sample SE（ddof=1）；全部與報表吻合，數值容差1e-10百分點。

| 指標 | CORE embedding | CORE50 | 差值［95% CI］ | Bootstrap SE |
|---|---:|---:|---:|---:|
| MRR | 65.16088 | 55.99737 | −9.16351［−11.49604, −6.78240］ | 1.20311 |
| R@1 | 54.40 | 42.90 | −11.50［−14.70250, −8.20000］ | 1.66554 |
| R@5 | 79.70 | 71.70 | −8.00［−10.50, −5.40］ | 1.30393 |
| R@10 | 85.30 | 82.40 | −2.90［−5.20, −0.70］ | 1.15234 |
| R@50 | 96.90 | 96.90 | 0［0, 0］ | 0 |

來源排名改善207題、下降377題、不變416題；top1新增89題、失去204題，淨減115題。R@50始終969/1,000，是固定同一prefix只做內部重排的結構性不變量；其零差值／零CI不是模型召回一般不受影響的統計發現。

這裡是**單來源圖MRR／Recall**。其他圖可能也符合文字要求，原資料沒有窮盡Boolean語義正例；不能將這些數字稱為完整Boolean mAP／full-match AP，也不能據此宣稱reranker普遍無效。群組區間條件於此公開開發集合，未完整建模共享gallery與語義近重複。新固定suite應保留embedding和CORE50的既有基準數字，後續方法比較不能隱去本次重排下降。

## 成本範圍

| 保存量測 | 秒／數量 |
|---|---:|
| 實際pairs | 50,000（每query50） |
| 完成checkpoint active time | 3,538.12412秒（約58.97分） |
| Active time／pair | 0.0707625秒 |
| Active time／query，按完整作業攤分 | 3.53812秒 |
| 本次cross invocation wall | 3,968.86607秒（約66.15分） |
| 本次checkpoint呼叫wall | 3,943.91814秒 |

Active interval涵蓋每批compute（含JPEG解碼、processor、模型推論、回CPU）與NPY寫入／flush；不涵蓋state JSON寫入、其他invocation的載入或失敗／in-flight batches。Invocation wall另含本次驗證、cheap排名和模型載入；這不是fresh query encoding＋搜尋＋重排的線上端到端延遲，不能把攤分值冒稱單次query實測延遲。原始timing及每批紀錄全部保留。

## 關鍵產物SHA256

| 產物 | SHA256 |
|---|---|
| `index-meta.json` | `52325472152758089383733d599b39a2feaff7d9dd547653ca40c708f17408c6` |
| `cross-report.json` | `57e39d7063ab494274300f2f0fba94bac3dcb53f3902ce0cc378cf5141cebf9f` |
| `cross-per_query.json` | `06f256d90067619ed71dc779a208100b414f883e5b949900644fee4ddd37faaf` |
| `cross-meta.json` | `4d6dfa49f3724e1f5bd89b716df63bd0ecd6712e44f761a8b3c688abd403a283` |
| `cross-provenance.json` | `fec2789550ec530d1f3490a5c77e711ed59621f3116071920ee889d9c180d3fd` |
| `cross-pairs.json` | `2fe731a0b616ef38dc66ef8b89d3dccbb06f3ddce2edcd3c06fd08cadb413ba5` |
| `cross50.npy` | `6ffb54662e6ec34893a82a587170cad97400c7ae75d1a82abf405ab5d59f7dd7` |
| `cross50.state.json` | `723a76cac88a004e893b74901bdebfed5a346456014bcd68b6bfb9a361ad6dad` |

所有canonical raw／vectors／state／receipt原地保留；本次沒有清理、重算GPU或更換baseline設定。
