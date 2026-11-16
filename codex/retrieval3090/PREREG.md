# 3090 多條件檢索開發協定

[codex] 2026-11-16。使用者授權執行前一輪審查的補強與結果分析。本協定在新的全池模型快取與方法分析之前固定。這是已曝光資料上的開發；不把選出的最佳設定或其名目區間當作正式確認證據。

## 研究問題與資料邊界

主問題：不增加每題 50 次 CORE 精細重排，從 CORE 前 100／200 張選出更好的候選，能否提高全池 mAP？既有結果與資料來源見 [原規範](../../topics/compositional-retrieval/code/dev-3090/SPEC.md)、[本機資料驗收](../reports/2026-11-16-retrieval3090-setup-codex.md)。本機的公開資料重建不冒充缺失的叢集實測收據。

沿用已曝光 8,191 張圖、136 查詢、118 群及原 JPEG／manifest／模型 revision。禁止開啟 `retrieval-independent-panel` 的 JPEG、query manifest 或研究輸出。原 120 查詢面板的設計文件可讀，但原比較是交錯入池；若新方法進入正式驗證，另寫凍結協定，不偷偷改原方法。三個已暫停／淘汰題目保持原狀。

## 完整性、快取與開發判讀

- 固定程式從提交取出。全圖庫與整句／條件向量依原順序與 batch 計算；重排先依原順序計前 50，再分開計 51–200，共 27,200 對。所有 200 對分數只供離線開發評估；選入池政策只可讀便宜 embedding 分數，不可讀標籤或 cross 分數。
- 可續跑快取每個 batch 留進度、來源與順序指紋。來源／depth／設定不同不得接續。完成後同時計算原基線驗收；不符合時只作差異診斷，不產生方法結論。
- 驗收條件沿用原值：mAP、固定前50名 oracle mAP、Recall@50 各在原值 ±0.10 pp；無正例題數精確一致。數值不因本機結果改門檻。
- D1：正例落點、零命中查詢、單條件負例；D2：入池內排序錯誤。以標註定義的 oracle 與實際模型分開。前200全重排只稱高成本參考，不能稱嚴格上限。
- M1：原25政策×depth100/200；`llr` 正名為 `logsigmoid`；新增同backbone mean/product、完整 LCSE AND 公式的明示改編、隨機選50（10個固定seed）與跨query打亂條件。每個模型／解析／候選／尾序一致，分開說明額外索引的差別。
- 開發內選法：`SHA256([2026111663,group]) mod 5` 固定群分割。每fold只在其餘fold選政策，依 mean AP、固定Recall@50、政策id排序；CORE保底，排除隨機／打亂對照。報全部探索結果及 OOF 結果；OOF區間仍只是條件式開發估計，不能取代封存確認。
- M2：原分數落差及1–50 margin、softmax entropy（T=1/0.1/0.01）按整批query四分位分配75/50/50/25；固定64個seed的隨機分配作對照。分開報入池Recall@budget、最終固定Recall@50、mAP。這是整批分配，不宣稱已有逐筆服務的預算控制器。
- 統計：配對query-group bootstrap10,000次，原seed2026111662；按1正例/多正例分層。原50組加LCSE控制後，共98個M1政策設定中的最佳值有選擇偏差。用群間配對變異估區間精度與80% power的近似MDE，明示假設；不以120查詢直接保證能辨識+2pp。
- 開發門檻沿用Recall@50至少+5pp、mAP至少+2pp、mAP配對區間下界>0作送正式的參考。未達只描述第一版方法，不自動淘汰整題。正式需新協定、未看過資料及獨立審查。

## 補充實驗

1. 成本：固定重排25/50/75/100/150/200，量GPU warm計算、CPU admission、文字編碼、索引儲存、模型載入的攤提邊界。快取建置時間不能冒充線上延遲；以實際時間比較直接多重排與M1。
2. LCSE校準：從先前已曝光conjunction圖中，排除dev/rerank池與val，依固定seed2026111664雜湊挑512張train圖；與評估池互斥，不觸及封存池。四個官方模板逐一算cosine後取平均，不重新normalize平均向量。以全概念Bernoulli NLL選全域mu/beta，mu為[-1,1]81點、beta=(1,2,5,10,20,30,50,60,100)，不rebalance。這是官方AND公式配合train-only presence校準的改編，不冒充原作多operator調參協定的重現。另列paper固定參數transfer以便追溯。
3. D3：同固定圖庫，以同圖nested條件family k1…4；保存正例比例、物件頻率／面積、缺一負例數等協變量。新增holistic文字必須實際編碼。報原始與控制後描述性關聯；不能把k增多的自然難度當作一般定理，也不能由兩條單調公理直接推出跨圖全符合排序保證。
4. 標註：匯出匿名盲查樣本和另存映射；空白人工審查表不算完成的人類標註。若由模型協助看圖，明確標為AI初審，保留不確定，不冒充人工gold。
5. 外部泛化：先核對MCMR schema與原始輸入資訊，取得公開metadata、釘版本和schema；原本的圖文candidate／query若與本題不同，列獨立協定，不能任意刪metadata後仍稱官方主設定。訓練與正式大實驗依使用者決定回叢集；本機無叢集連線時保留可執行交接與未執行狀態，不偽造完成。

[codex] 2026-11-16 05:06，任何M1／M2／D3效果仍未計算或讀取，固定外部pilot／成本的單一政策選法：在完整COCO開發的所有 `m1/` 政策中，依mean AP降序、固定Recall@50降序、policy id升序選一個，保存分析檔SHA與政策收據；即使未過送正式門檻也可做探索pilot，但不得因此宣稱正式成功。保留CORE及其等價控制作保底；若選中基線等價政策，照實報無新增政策收益。該政策及深度直接轉移MCMR，train-only LCSE參數如需則原樣轉移，不在MCMR看標籤後重新選法。成本也只加測此同一政策。外部pilot使用事前固定的128 query／8,320商品原生圖文輸入，並列其不同於官方重現的prompt／batch／重排改編；不稱完整MCMR benchmark。

[codex] 2026-11-16 05:10，為平行利用空閒時段，MCMR方法無關的CORE50共6,400對可在COCO政策鎖定前先計算raw scores，不分析外部效果。政策鎖定後，僅在同來源／資料／索引／環境／原cross batch1的共同實際pair才可復用；先固定抽8個共同pair逐值重算驗證，另記wrapper程式與cache指紋、復用與新算數量。不能改query/商品metadata、挑subset、改batch或藉復用讀取cross來選入池。若無法完成此一致性驗證，走直接計算流程。

[codex] 2026-11-16 05:26，盲審包及任何AI審查尚未生成／讀取。補充一輪明示的AI盲初審：依公開CSV固定順序分三個不重疊區段，由未見私有映射、模型排名或標籤的子代理，只看query與原尺寸匿名圖片，逐例記yes/no/uncertain與tiny/occluded等不確定性。先凍結完整AI標籤與檔案SHA，再合併私有COCO規則標籤作分層描述；圖片可見性、2%面積規則及缺標註分開討論。保留空白人工表，AI不稱人工gold、模型評分oracle或整體錯標率估計。

## 分工與產物

root整合、固定協定、快取／追加編碼／成本執行、結果報告。setup_review只寫analysis_v2與其測試；literature_check只寫logic_baselines、測試及theory審計；potential_audit只寫diagnostics、測試及data協定。各自不修改他人檔案。報告置 `codex/reports/`，大量陣列／圖片置ignored `dl/retrieval3090/`。主README只記操作方法。

## 讀取紀錄

- [codex] 2026-11-16，固定前僅讀既有公開VERDICT／報告、來源與設計，未讀本輪新研究結果或封存資料。新增研究結果的讀取在此逐項追加時間與路徑；判準內容不隨結果改寫。
- [codex] 2026-11-16 04:21–04:23，root已讀 `dl/retrieval3090/runs/supplement-20261116-v2/lcse_calibration.json` 作機械來源與同參數重現校驗；已讀 `dl/retrieval3090/runs/dev-20261116-v2/baseline-early-audit.json` 的top50早期重現結果。未通過原驗收，僅作版本差異診斷，未跑或解讀M1/M2/D3模型效果。授權setup_review讀同早期重現檔協助查因。原容差及未通過停止方法分析的規則不改。
- [codex] 2026-11-16 04:25–04:27，setup_review/root已讀相同開發快取的 `checkpoints/core_embed_image.npy`、`core_embed_full.npy`、`cross_top50.npy` 及 `parity-cpu-threads-20261116.json`，僅作原基線重現差異診斷與輸入對齊核驗，未讀政策或封存結果。
- [codex] 2026-11-16 04:38–04:43，root/setup_review已讀 `dl/retrieval3090/runs/parity-torch214-20261116/probe.json` 及 `parity-torch26-20261116/probe.json`，只作固定batch的向量、輸入雜湊與重排分數技術對照。literature_check於04:40–04:43讀前者與 `dev-20261116-v2/baseline-early-audit.json` 獨立複核基線閘門；沒有讀M1/M2/D3或封存資料。新環境全池基線執行提交固定為 `4c5c20188e3db6f9aeef5e1f9ef77c4dd7f9eb52`。

## 前置完整性修正

- [codex] 2026-11-16 04:06，僅檢查公開標註群分割、未看本輪任何M1／M2／D3結果時，發現原sentinel群仍可能共享其他正例。因此OOF選法改用「同原群或共享任一正例」的連通群，同一seed/hash規則；原基線比較仍保留原群bootstrap以便對照，OOF另按連通群估計。另加四模板原向量、文字順序、模型revision及來源指紋驗證。這些是結果讀取前的完整性補強。
- [codex] 2026-11-16，依使用者新增要求：後續GPU工作只選有足夠餘裕的卡（預設至少14 GiB空閒且整體util不高、連續兩次檢查）；無卡持續排隊等待。完整輸出驗證成功且無writer時，清除可由最終NPZ逐位驗證的重複batch中間陣列，保留來源、timing、raw scores、分析與清理收據。
- [codex] 2026-11-16 04:33，原基線重現未通過後，只作版本查因，不變更接受容差。新增同代Torch 2.14.0+cu126／torchvision0.29.0對照環境，其他模型／資料／查詢／batch／Pillow／Transformer設定保持，先以原batch的前16圖、前32查詢及8個既存重排pair做技術差異測試。僅版本與driver相容測試，不依M1結果挑環境。原環境與原結果保留。後續全池重現改成前50完成立即驗收，未過不再計51–200。
- [codex] 2026-11-16 04:48，仍未讀取任何M1／M2／D3方法效果，torch214全池基線尚在執行時，另外固定條件式 [本機平台開發協定](platform_plan.json)。原重現流程與原容差不變；僅當這一次同代框架重現也失敗時，另開新輸出與明示入口，保留原 `MISMATCH`，通過來源、逐值重複、完整快取及同環境驗證後才做相對本機CORE的探索比較。不是把失敗改為PASS，也不是原H200結果的確認。算法、seed、控制組、OOF與正式驗證要求沿用原已凍結設定；不再為湊基線搜尋版本或backend。
- [codex] 2026-11-16 04:51–04:55，root已讀 `dl/retrieval3090/runs/supplement-torch214-20261116/lcse_calibration.json` 及相應原始校準向量，僅核對train-only來源、cosine指紋及重新擬合結果，確認後清理可重建的重複中間陣列。未讀取政策或D3效果。
- [codex] 2026-11-16 04:55–04:58，root已讀 `dl/retrieval3090/runs/dev-torch214-20261116/candidate-replay-only.json`，只檢查原候選召回與oracle重現。條件式平台協定於04:48起草；本次提交前已知原候選閘門未過，但仍未計算或讀取任何M1／M2／D3效果。正式採用新平台流程須本次提交後，另一次明示啟動、完整來源／逐值重複驗證及獨立審查。
- [codex] 2026-11-16 05:06:54–05:09，potential_audit/root已讀 `dl/retrieval3090/runs/dev-torch214-20261116/baseline-top50-replay.json`；potential_audit亦讀獨立本機續算的 `reranker-repeat.json` 作8對技術重複核驗。原重現檔保留不變，未讀M1／M2／D3或外部效果。
- [codex] 2026-11-16 05:21–05:23，literature_check於05:21:16首次讀取MCMR共享索引的 `index.npz` 及來源／環境／圖片manifest作獨立技術驗收；root讀取相同索引及逐值清理收據以刪除已由最終NPZ保留的兩份重複checkpoint。未讀MCMR重排效果或任何COCO方法效果。
- [codex] 2026-11-16 05:39，literature_check於05:39:20.766首次讀取外部CORE50原始 `cache.npz`／checkpoint並作來源、pairs及逐值完整性驗收；未計算或讀取外部AP／Recall。控制程序中斷、子程序退出碼未知，COMPLETE產物獨立驗收，不虛報exit 0。
- [codex] 2026-11-16 05:43–05:44，root首次讀取 `analysis-platform-torch214-20261116/report.md`、`report.json` 與 `d3-platform-torch214-20261116.json`；potential_audit於05:44:16讀report.md、05:44:27讀report.json與per_query.json，進行未實作該分析器的獨立審查。皆僅為已凍結本機平台探索，未讀外部效果、私有盲審映射或封存資料。
- [codex] 2026-11-16 05:45，依05:06固定規則，root機械選擇單一COCO政策並於05:45:39保存 `analysis-platform-torch214-20261116/selected-policy.json`，綁定完整分析、平台計畫、政策檔SHA；在任何外部效果分析前鎖定。外部與成本工作使用此同一政策。使用者同意以本機結果繼續開發及分析，原重現MISMATCH仍保留。
- [codex] 2026-11-16 05:48，literature_check於05:48:35.212首次讀取D3結果及其平台來源，繼續以design、同214文字／影像向量獨立重算。其稍早被動收到potential_audit的COCO彙總審查訊息，但未開COCO原始分析、未讀外部效果或私有盲審映射。
- [codex] 2026-11-16 05:46–05:51，三個新子代理（fork_turns=none）只看公開query／原尺寸匿名圖片，按固定0–47／48–95／96–143分段完成144例AI初審；未讀private、方法排名／效果或其他審查者標籤。root於05:51:16核對覆蓋與原欄位後保存 `ai-review/AI-freeze.json`、合併標籤及SHA，之後於05:51–05:52首次讀 `private/key.json` 作分層描述。人工標籤仍空白；AI不稱人工gold或整體錯標率。
- [codex] 2026-11-16 05:53，root在AI標籤凍結後，對兩個已見分歧案例（image218589、125017）看原圖並查官方train標註bbox，另讀image567609標註；屬事後質性診斷，未改任何凍結標籤、評分或政策，不作整體錯標率或人工裁決。
- [codex] 2026-11-16 06:00，potential_audit於06:00:16讀D3獨立審查文字、06:00:25讀已凍結AI audit.json與AI-freeze.json，對root整合稿作文字核對；沒有重判圖片、改標籤或讀取外部效果。
- [codex] 2026-11-16 05:52–06:03，setup_review首次於05:52:10讀成本checkpoint、06:03:37讀完整report.json，作未實作benchmark.py的獨立成本審查；root於06:02–06:03讀完整成本report。setup_review另於06:01:13以原vectors與header完成固定8題候選忠實性核對，未讀AP或cross分數、未使用標籤，沒有改政策。
- [codex] 2026-11-16 06:04–06:05，root首次讀取已鎖定單一政策的 `mcmr-selected-torch214-20261116/analysis.json` 外部效果；政策與超參數不改，不在MCMR另選政策。授權literature_check作未實作MCMR runner／reuse wrapper的獨立效果與完整性審查，實際首次讀取時間另記。
- [codex] 2026-11-16 06:05，literature_check於06:05:54.075首次直接讀MCMR分析及方法raw scores；此前被動收到root的已讀摘要。其獨立重算全部128題選入、排序、指標與bootstrap，並逐值核對三個CS checkpoint後同意清理重複檔；沒有改政策或子集。
- [codex] 2026-11-16 05:50–06:12，root已讀COCO、D3、成本及MCMR的獨立審查文字，將計算／統計／證據邊界整合到reports與results摘要；主判準、政策及子集不改。保留所有讀後敏感性分析的探索標記，並維持人工gold與正式叢集確認尚未執行的狀態。

## 使用者指定的後續研究基準

[codex] 2026-11-16 13:39，使用者明確指定以本機重跑作baseline。後續研究採用已驗收torch214平台的CORE50；舊叢集重現對齊只作來源追溯，不以其是否成功判定本機研究可否進行。既有凍結產物與歷史狀態不回寫，保持同機／同資料／同評估的相對比較，方法與確認資料另先固定。
- [codex] 2026-11-16 13:32–13:39，root重讀既有實驗報告、results摘要、主分析與LCSE程式、已曝光校準參數，用於使用者方法／CI說明；potential_audit於13:33:15重讀既有COCO report及per_query並另作連通群變異的讀後樣本量粗估；setup_review於13:35–13:37重讀指定方法摘要／CI及選法收據。未執行新GPU實驗、未讀封存或盲審私有檔，沒有改政策或原判準。規劃估算只放reports，不作新確認結果。
