# [codex] 擴大 retrieval 開發資料：來源、用途與最小組合

[codex] 2026-11-16 14:36來源調查，15:15整合更新（台北時間）。來源與規模依官方論文、作者資料卡／發布檔／程式核對；後段另記本機實際資料驗收、已完成的向量檢索基準及執行中的精排。沒有讀取private／sealed或retrieval-independent-panel；新公開開發協定與程式在GPU執行前另行凍結，既有確認協定未改。

## 推薦決定

採用 **COCO val2017 共用 5,000 圖庫＋NegBench 原始／否定 caption 全量 cheap evaluation** 作主開發層；昂貴 reranking 另固定 **1,000 個互異JPEG來源群**，明定每圖選哪個 caption。加入 FACTOR-Bench 的公開配對診斷；Half-Truths v2先列備用，未納入本輪執行；MCMR 保留原生「商品圖像＋文字 metadata」跨域層，暫不擴其昂貴評估。Flickr30K、COCO train 整包、CC12M 與 CS-CLIP training archive 不必先安裝。

這是依根代理本輪決定，將**公開測試資料明示轉為開發用途**。它們能增加樣本量、語句變化與診斷覆蓋；經開發選擇後不能再稱未見的正式確認。既有 sealed 確認協定維持原狀；本報告沒有讀取或自行核對其內容／排除名單。

**數量更正：NegBench 的 5,000 是來源影像 rows，不是全部 query 數。每個 CSV 保留原始位置展開後均有 25,014 queries；否定檔其中一條為空字串，不送模型，所以實際有效否定查詢為25,013條。** 全量 cheap evaluation 因而是同一圖庫上的 25,014 原始 captions 與 25,014 否定 captions，兩套指標分開列。每 query rerank K、query 筆數和 gallery 圖數是不同的擴張軸。

## 六個來源的核對

| 來源／版本 | 實際內容與規模單位 | 官方 split 與本研究用途 | 取得／授權與優先級 |
|---|---|---|---|
| **COCO 2017** | 自然場景影像，附 instances／categories、captions；不是商品 metadata，也沒有現成 Boolean query relevance matrix。官方下載標 train 約118K、val 5K。 | `train2017`／`val2017` 是資料版本命名，不能等同本研究 fitting／development／confirmation。此輪只需 val2017 作公開開發圖庫；train 保留後續另案。 | 官方 val zip 約1GB、train 約18GB、train/val annotations 241MB。annotation CC BY4.0；影像保留 Flickr 原條件，不能一概稱 CC BY4.0。[C1–C2] **先取／重用 val 與所需 annotation。** |
| **NegBench，CVPR2025；repo `2ac009b…`** | 本次選 COCO retrieval：每檔5,000 unique image rows，各25,014 captions；另有 MCQ、VOC、合成圖／影片等，這些不是本次都要下載。 | 作者將這些列為 evaluation；沒有本次可直接沿用的獨立 train/dev split。本研究明示轉 public development。CC12M-NegCap／NegMCQ 是另外的 fine-tuning corpus，不混為 eval。 | 官方 Drive 提供 CSV；影像另用 COCO val2017。repo MIT 是程式／文件授權，不能據此重授 COCO 影像。[N1–N4] **主開發來源。** |
| **FACTOR-Bench／LCSE；repo `8af1972…`** | 1,100 operator＋450 equivalence＋145 compound＝1,695 單圖／雙caption配對；不是 retrieval gallery。發布 JSONL 共1,320 unique images；operator 子集979張。 | 全部引用 val2017；HF 公開 split 僅 `test`。LCSE 的 train2017 校準是另一流程，未公開完整校準子集，不能拿1,100冒稱dev。 | JSONL約1.32MB，重用同一COCO val影像。作者 annotations CC BY4.0；不發布影像，原圖條件另計。[F1–F3] **低成本邏輯診斷。** |
| **Half-Truths v2，arXiv2602.23906v2，2026-10-23** | 正確短 anchor、正確延伸 full_truth、錯誤延伸 half_truth；entity／relation 診斷。三設定509、454、474比較，合計1,437；比較可共享影像，不等於1,437張圖。 | 作者三設定均是 `test`，不是train/dev/test三分。COCO部分可見 `COCO_val2014_…` source IDs；CC3M是另一影像來源。作額外公開診斷，明示是否曾參與開發。 | 官方HF約202MB，Parquet含實際評估圖像、`image_sha256`。作者新增annotations CC BY4.0，沒有重授原始圖／caption。[H1–H2] **可加入，但不把它改造成完整retrieval真值。** |
| **MCMR，arXiv2603.01082v1；HF `12d53f4…`** | 商品圖像＋title／description／features等長metadata；query需視覺與文字互補證據。paper表2為3,997 queries／104,981 candidates，與正文「10,400 products」不一致，應以鎖版發布檔驗數。 | 官方HF把不同schema JSONL誤嘗試合併為 `train` 並報錯；這不是作者定義的training split。作者是zero-shot benchmark，發布candidate/query清單，未見獨立dev定義。 | 官方發布 `candidates.jsonl` 148MB、`query.jsonl` 1.37MB、`images.tar.gz` 2.26GB。repo Apache2.0不能自動解釋為Amazon來源圖／metadata已換授權；HF未見獨立dataset license宣告。[M1–M3] **保留跨域層，不先擴新昂貴計算。** |
| **Flickr30K／Entities v1.0** | 原版31,783自然圖像、158,915 captions。Entities另有phrase對應／boxes，並非現成AND/NOT全圖庫真值。 | 作者Entities發布檔可核：train29,783、val1,000、test1,000 IDs；不要混稱另一套Karpathy split。若未來用於retention，固定來源split並明示用途。 | 原作者頁需申請影像；下載頁另供image links／captions。研究／教育非商業使用且遵循Flickr條件。[L1–L3] **暫不安裝：此次已有COCO自然caption控制，不增加同類下載與adapter。** |

## NegBench 的實際檔案與語意

作者官方資料準備文件 [N1] 指向 Drive folder：<https://drive.google.com/drive/folders/1kSEq0mkV1t1T8GuOAM65iz_iAA7e5gxB?usp=sharing>。本次核對兩份完整 CSV 文字，UTF-8 bytes 數與 Drive 宣告大小相符：

| 檔案 | 官方 file URL | bytes | SHA256 |
|---|---|---:|---|
| `COCO_val_negated_retrieval_llama3.1_rephrased_affneg_true.csv` | [negated CSV](https://drive.google.com/file/d/1EAqblIkLWtaHme7fT2ESlYH0FTeE4WEw/view?usp=drivesdk) | 2,912,742 | `08a8db9b96130ab27f91f5ef14f413981d70a96bc9b3557436a26256db115b7d` |
| `COCO_val_retrieval.csv` | [original CSV](https://drive.google.com/file/d/1pSIEIJwIsf_jdb0Zj9VEgzdt5rFz2BC3/view?usp=drivesdk) | 1,994,138 | `dfe9217359b42f3f68bd93245f92a04701b8d3cde0aaf3fd1c26694d6de8d547` |

兩份 header 均為：

```text
positive_objects,negative_objects,filepath,image_id,captions
```

每檔有5,000個互異 `image_id`，兩檔順序相同，正／負 object lists 亦相同；全部filepath指向val2017。caption list長度分布：4,987 rows各5條、12 rows各6條、1 row為7條，合計25,014。官方runner逐圖展開全部captions [N2]，所以原方法的評估單位不能簡化成5,000 queries。

**有圖像層正／負object lists，但沒有逐caption的條件對應欄。** 例如 `image_id=397133` 的負list為chair、fork、car；五個否定caption前兩個否chair、後三個否fork，不能把三項都當每條query要求。`positive_objects` 也不是每條caption必須逐一提及的全部條件。若要 condition scoring，需先凍結caption解析／配對流程，記錄成功／失敗與覆蓋率；不得用source image的標註代替真實query parse，或事後只保留容易解析者。

官方retrieval ground truth是 **caption的來源圖像**：runner把每query對其source row的配對標成positive [N2]。它沒有標盡圖庫中所有可能滿足該語句的影像。因此可報source-image Recall／MRR／admission，以及原始與否定caption的差異；不能把所有其他圖像當已驗證的Boolean false positive，再聲稱得到完整full-match AP。此限制也不能靠增大K消失。

reader以list字串儲存captions；本地匯入應使用 `ast.literal_eval`／嚴格型別驗證，不照抄作者reader的 `eval` [N3]。固定檔案SHA、source `image_id`、caption原始索引與完整文字，不能只保存重新排序後的row number。

## 可實作的最小多層組合

1. **共用影像與全量cheap層。** 固定val2017的5,000 IDs，原圖bytes／解碼規則、labels／captions各自保留SHA。每個既有frozen backbone只建一次影像index；NegBench原始與否定25,014 queries各自全量評估。這增加自然語句覆蓋並保留完整圖庫，毋須新增baseline。若為2,048維float32，單一5k影像表約40.96MB；25,014×5,000 score矩陣約500.28MB，屬CPU儲存估算，可用query chunk避免一次保留多份。
2. **昂貴開發層。** 在不看模型效果下，按固定seed/hash挑1,000個source image IDs；再明定每圖一個negated caption索引。這才是1,000昂貴queries；若保留所有captions則約5,000，必須另算成本。本輪plain只作cheap對照，不增加plain cross。若進一步做原／否定逐query配對，需查兩份lists的逐caption對齊，不能僅因同一image row就假定文本索引已語義對齊。所有開發query均搜尋原5,000圖庫，不因1k子集而縮圖庫。
3. **邏輯與half-truth公開診斷。** FACTOR全1,695比較先建立完整句子cosine的配對基準；使用oracle parses執行LCSE需另凍結方法，不能把本輪基準稱LCSE重現。Half-Truths v2全1,437比較分三設定、entity／relation列出，但本輪尚未啟用。它們是不同測量，不能把比較數加進retrieval query數當同一個總樣本量。若作開發選法，標作development；若只作選法鎖定後的描述，仍說明COCO影像重疊，不能宣稱資料獨立。
4. **原生跨域與正式確認。** MCMR需要保留圖像和metadata輸入，不能用純圖分支替代後稱原生MCMR。此輪不新增全量cross；現有public pilot只按其已登記用途解讀。真正confirmation由既有封存協定處理，本報告沒有變更，也不把公開資料的增量當新增formal evidence。

資料準備與GPU預算分開：1,000 queries×K50為50,000 query–candidate pairs，K200則200,000；全25,014 queries×K50已達1,250,700 pairs／政策。這些是工作量算術，不是RTX3090實測耗時或完成保證。先保證全量cheap層和固定昂貴子集可重用，再由實測吞吐決定K與政策數；不把cache建置時間當線上延遲。

## split／重疊與版本要點

- **FACTOR可完全共用COCO val2017圖像。** JSONL checksum `37f179a66913d2fbf55bf4f9a852273e894d53683dda0c7a8a7fff6c7b7d387d` 已與作者README相符；1,320 unique images是讀發布JSONL後按 `image`／`image_id` 去重所得，不是paper另報的資料規模。
- **Half-Truths不能假定全部可共用val2017。** 它的COCO來源為val2014命名；COCO2017重組了2014 splits [C1]。以數字image ID核交集，必要時再用其 `image_sha256` 核bytes；不在val2017的圖可直接用官方評估Parquet，不必先抓完整val2014。CC3M圖另算。不要僅憑「val2014」與「train2017」名稱宣稱互斥。
- **不要把公開HF顯示的 `train` 一律當學習資料。** MCMR頁面目前因candidate/query欄位不同而自動builder失敗，其預覽split標籤沒有定義一個可靠的train/dev/test方案。[M2]
- **Half-Truths training是另一包。** `kbora/CS-CLIP-Training`有410,340個train2014 caption records、約155MB annotations，圖像另取；不是410,340 unique images，亦不是人類驗證的gold開發集。作者資料卡明示其training foils有噪聲。這輪不fine-tune，毋須下載。[H3]
- **LCSE的約50張僅是校準充分性說法。** 論文附錄A用train2017五算子對μ／β grid search；headline校準實際張數、pairs／IDs沒有公開。FACTOR val上的ablation與方法選擇不能冒稱獨立dev資料。[F3]
- 預備新下載時，固定remote revision、fileSHA與有效records，記錄gallery unique IDs／bytes hash、query數、source-image群、語義family、parser版本／失敗列；同來源影像的多captions須成組進fold。若共享positive或近重複圖使連通群過大，應報告並改用事前明定的資料分割，不可拆群製造虛假的獨立樣本。

固定HF metadata亦已核：Half-Truths revision `ddcd448cc27c7c1b1d485c4754623566c245fdca`；CS-CLIP-Training revision `7c4782c687cd751efdecd02ca9f2d54388a0950f`；MCMR revision `12d53f4063804ee6a976024c7375ab2fbe4368c8`。前兩card license為cc-by-4.0，MCMR API未宣告license。Half-Truths的「v2」指論文版本，資料凍結另記HF revision，不把兩者混為同一版本號。

## COCO val2017.zip 的下載驗收依據

來源調查階段，官方下載URL及其S3 provider HTTPS URL均以HEAD驗得200；後續實際下載與驗收另見下節：

- 官方：<http://images.cocodataset.org/zips/val2017.zip>。
- 同一provider物件的有效TLS URL：<https://s3.amazonaws.com/images.cocodataset.org/zips/val2017.zip>。
- `Content-Length: 815585330` bytes；`Last-Modified: Wed, 11 Jul 2018 05:08:47 GMT`。
- `ETag: "d366be60d3dc737327160d62453e3973-98"`。這是multipart ETag，**不是檔案MD5**。
- `https://images.cocodataset.org/...` 在本次標準TLS驗證下報hostname mismatch；不應因此關閉TLS驗證。上列S3 HTTPS路徑已通過標準驗證。

COCO下載頁未列本次所需SHA；但Google官方TensorFlow Datasets checksum表 [C3] 已公開同一官方來源檔的大小與SHA256，可作獨立驗收依據。它是**TFDS發佈的來源檔checksum，不是COCO網站自行列出的checksum**：

| 官方來源檔 | bytes | TFDS發布SHA256 |
|---|---:|---|
| `val2017.zip` | 815,585,330 | `4f7e2ccb2866ec5041993c9cf2a952bbed69647b115d0f74da7ce8f4bef82f05` |
| `annotations_trainval2017.zip` | 252,907,541 | `113a836d90195ee1f884e704da6304dfaaecff1f023f49b6ca93c4aaae470268` |

下載後先比size／SHA256，再檢ZIP所有entry CRC；驗相對路徑安全、恰有5,000預期image IDs、無重複／多餘JPEG，與官方 `instances_val2017.json`／`captions_val2017.json` image inventory一致。CRC只能檢傳輸／解壓完整性，不能代替來源SHA。若圖像bytes或解碼尺寸有異，應停止建index查明，不可靜默跳圖讓gallery變小。

上述是下載前的驗收依據；實際bytes／JPEG inventory已於後續資料準備階段核對，見下節。本次不先取COCO train2017整包；它是需要更多gallery或另立fitting資料時的後續選項，不能在未排除跨資料集image IDs前直接混入本輪。

## 官方來源

- **C1** [COCO官方下載頁原始碼／2017重組說明](https://github.com/cocodataset/cocodataset.github.io/blob/master/dataset/download.htm)；[val2017官方zip](http://images.cocodataset.org/zips/val2017.zip)；[annotations官方zip](http://images.cocodataset.org/annotations/annotations_trainval2017.zip)。
- **C2** [COCO官方Terms of Use](https://github.com/cocodataset/cocodataset.github.io/blob/master/dataset/termsofuse.htm)。
- **C3** [Google TensorFlow Datasets發布的COCO checksum清單](https://github.com/tensorflow/datasets/blob/252301af4a62f264bddf16491fd0bfe57db8140f/tensorflow_datasets/url_checksums/coco.txt)，本次讀取git blob `0bd2c4e99a5a66f5f73b46b51547b93e61e9bd96`。
- **N1** [NegBench官方datasets.md，固定commit](https://github.com/m1k2zoo/negbench/blob/2ac009b5adfb18d1d7c185c466b2bced10f43f6a/datasets.md)。
- **N2** [NegBench官方retrieval.py：展開caption、來源圖GT與全gallery scoring](https://github.com/m1k2zoo/negbench/blob/2ac009b5adfb18d1d7c185c466b2bced10f43f6a/benchmarks/src/evaluation/retrieval.py#L43)。
- **N3** [NegBench官方CsvImageCaptionDataset reader](https://github.com/m1k2zoo/negbench/blob/2ac009b5adfb18d1d7c185c466b2bced10f43f6a/benchmarks/src/training/data.py#L148)。
- **N4** [NegBench官方LICENSE](https://github.com/m1k2zoo/negbench/blob/2ac009b5adfb18d1d7c185c466b2bced10f43f6a/LICENSE)。
- **F1** [FACTOR-Bench作者DATACARD](https://github.com/SultanMo/factored-vlm/blob/8af1972ddfd03e8437d5f5c96d5a784e94b82d99/docs/DATACARD.md)；[固定版JSONL](https://github.com/SultanMo/factored-vlm/blob/8af1972ddfd03e8437d5f5c96d5a784e94b82d99/data/factor_bench.jsonl)。
- **F2** [FACTOR-Bench LICENSE：annotation與圖像分列](https://github.com/SultanMo/factored-vlm/blob/8af1972ddfd03e8437d5f5c96d5a784e94b82d99/LICENSE)。
- **F3** [LCSE原論文§5.1／附錄A、B、K](https://arxiv.org/html/2607.23052)。
- **H1** [Half-Truths原論文v2](https://arxiv.org/abs/2602.23906v2)。
- **H2** [Half-Truths作者evaluation資料卡](https://huggingface.co/datasets/kbora/Half-Truths)。
- **H3** [CS-CLIP作者training資料卡](https://huggingface.co/datasets/kbora/CS-CLIP-Training)。
- **M1** [MCMR原論文§3／表2／§4.1](https://arxiv.org/html/2603.01082)。
- **M2** [MCMR作者HF資料頁](https://huggingface.co/datasets/Lux1997/MCMR)；[官方發布檔列表](https://huggingface.co/datasets/Lux1997/MCMR/tree/12d53f4063804ee6a976024c7375ab2fbe4368c8)。
- **M3** [MCMR作者README](https://github.com/EIT-NLP/MCMR)；[repo LICENSE](https://github.com/EIT-NLP/MCMR/blob/main/LICENSE)。
- **L1** [Flickr30K原作者頁：31,783圖／158,915 captions及影像申請](https://shannon.cs.illinois.edu/DenotationGraph/)。
- **L2** [Flickr30K原作者下載頁／使用條件](https://shannon.cs.illinois.edu/DenotationGraph/data/index.html)。
- **L3** [Flickr30K Entities作者repo／v1.0／split檔](https://github.com/BryanPlummer/flickr30k_entities)。本次直接核該repo `train.txt`／`val.txt`／`test.txt`互異ID數為29,783／1,000／1,000；對應git blob為 `20d370a205af79b99c928e6d8b6efda0227e4ae1`／`54b04f99694014942fa7afa3b8b383fb692cacea`／`a3cfe568a00deda3e3bc89eb2fdbeeb6c2aa8301`。

## 本機資料準備與獨立驗收

[codex] 2026-11-16 14:45（台北），root整合。已從作者官方Drive透過Google Drive連接器取得三份原始CSV，固定SHA；從官方S3取得val2017.zip，大小與SHA符合上列TFDS清單。全部5,000 JPEG完成原始固定fingerprint、尺寸／格式與完整解碼校驗；literature_check另逐檔重算SHA與大小通過。原圖總814,705,164 bytes，共用 `dl/retrieval3090/data/coco2017/val2017/`；壓縮包驗收完成後刪除，回收815,585,330 bytes，保留來源與逐圖收據。

`literature_data.py prepare`已產生兩份full panel、兩份paired-dev1000 panel及FACTOR matching panel。準備收據SHA為 `86814f40da7a74c7f606cc99ccc4d6bae0a7229ab1eb1008b884cb885638525a`。5,000圖的JPEG SHA與COCO Flickr來源ID各自均互異；固定抽樣1,000來源群，普通／否定同圖、同原始caption位置。這排除已知exact-source重複，並不證明所有語義或近重複依賴均已消失。

發現並事先固定清潔規則：否定CSV在image301061、caption索引0有一條空文字，完整保留原始位置、標為不可模型輸入；full neg有效分母25,013，plain為25,014。48個來源圖的positive_objects包含空字串，這些image-level lists僅存audit，不作模型輸入。paired dev選雙方皆非空的共同位置，仍維持1,000條；沒有根據任何模型效果篩題。FACTOR的450個`both`是equivalence，與1,100 operator及145 compound二選一分開評估。

Half-Truths指定snapshot的三Parquet已在本機；只讀footer可核共1,437例，無須再下載。但允許查閱的公開來源設計，尚未將該snapshot與原封存排除收據完整綁定。本輪先不啟用其圖片或query內容，也不為補證而打開封存清冊。MCMR保留既有原生圖文跨域pilot；Flickr30K、全COCO train與大規模訓練資料未新增。

以上為資料驗收；新增模型基準及精排狀態分開記錄如下。

## 本機開發已開始：完成範圍與第一輪觀察

[codex] 2026-11-16 15:15（台北）。凍結來源commit `cbd545ac96e4978454c41a78f78b75419d413b77`，tag `codex/retrieval3090-literature-20261116`。39項資料／runner測試、shell語法及實際GPU smoke通過。完整5,000圖及52,775個去重文字的CORE／CS向量已建妥；canonical NPY總591,616,000 bytes，保留作後續重用，不另存NPZ複本。setup_review已獨立核來源、state／payload與processor／environment一致性。這裡的52,775是包含FACTOR captions的去重編碼數，不是獨立retrieval樣本量。

全量普通25,014／否定25,013有效query的向量檢索計分已完成。literature_check未實作該producer，另以不同排名算法逐條重算dev四組共4,000個來源圖排名，並重算四組paired bootstrap，全部一致；full部分核資格／分母並由已存ranks重算指標，沒有再做全部full dot products。FACTOR全1,695項cosine與分組亦重新核算通過。詳見[獨立效果審查](2026-11-16-retrieval-literature-results-review-codex.md)。

固定1,000來源群、否定query上的第一輪基準（同一5,000圖庫；這裡尚未精排）：

| 方法 | 來源圖MRR（%） | 來源圖R@1（%） | 來源圖R@50（%） |
|---|---:|---:|---:|
| CORE-EMBED | 65.16088 | 54.40 | 96.90 |
| CS-CLIP | 53.36044 | 40.80 | 95.30 |

CS−CORE的MRR差為−11.80044百分點，來源群paired bootstrap95%CI［−14.24351, −9.37316］。它顯示本輪checkpoint比較有清楚差距，但不是同backbone fine-tuning因果ablation；也沒有證明任一新方法有效。較大的資料能降低部分抽樣不確定性，不能保證小幅改善的CI不跨0。固定dev的兩種文字都沒有CS超77-token query，不能用這個截斷事件解釋dev差距。

目前CORE前50已包含969/1,000個來源圖，表示在這份source-image任務中，只增加候選數的入池Recall理論剩餘空間為3.1百分點；排序改善仍是另一個問題。這也說明大樣本開發可幫我們分辨入池與排序瓶頸。FACTOR的AND配對上CS較強，整體來源圖檢索卻較弱，兩種任務不可互相替代；FACTOR基準不是LCSE結果。

**CORE50精排仍在執行，尚未讀取效果。** 固定1,000條否定query×50候選，共50,000配對，不依來源圖是否入池篩題。背景pipeline會在精排完成後自動產生 `cross-report.json`／`cross-per_query.json`，未完成前不稱其通過效果驗收。佇列狀態：`dl/retrieval3090/queues/literature-dev-20261116.json`；技術進度：`dl/retrieval3090/runs/literature-dev-20261116/main/progress.json`。本次實際選physical GPU3，啟動時20,792MiB空閒；依序載入模型、batch影像8／文字32／cross4，截至本節更新未見OOM。15:15快照為精排完成5,400/50,000配對，整卡used8,419MiB、free15,835MiB；這是含其他使用者的當下整卡快照，不是本模型peak量測。GPU准入仍為至少14GiB空閒且util≤20%、兩次確認，無合適卡即排隊。

本輪已清理驗收後的官方下載ZIP815,585,330 bytes，以及逐byte相同的smoke文字表複本3,918,306 bytes；保留原圖、原始資料、canonical向量／state、來源與清理收據。未刪可重用模型或其他研究產物。後續M1／LCSE或條件解析改編需在新一輪事前固定，再使用這份公開開發資料比較；目前沒有執行新增訓練。

最後的測量邊界：NegBench只標來源圖；本輪不能檢驗所有其他圖是否同時滿足AND／NOT條件。若要主張多條件full-match檢索改善，仍需原本相應的全相關性標註／獨立確認資料，不能把本輪MRR替換成該mAP。所有公開caption已用於全量cheap評估；未抽中昂貴dev的其他caption也不能再稱完全未見。
