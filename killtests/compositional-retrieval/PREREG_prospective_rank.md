# 組合檢索前瞻排名驗證

[codex] 2026-11-15 台北。先讀了 Half-Truths 的已凍結結果及 anchor 診斷，才設計本試驗。這批新查詢的模型分數尚未產生。此試驗衡量一個固定索引組合規則在未見查詢上的實際排名增益，候選池資格沿用報告 60 與 26393。[入池判準](PREREG_headroom.md) · [既有 anchor 診斷](../../codex/retrieval-anchor-audit/REPORT.md)

## 資料與凍結

- 沿用已快取的官方 COCO val2017 JPEG 與 `instances_val2017.json`。每列 COCO 影像 ID 排除 Half-Truths 三個子集的 COCO source IDs，並排除任何與三子集 JPEG SHA-256 一致的檔案。DEV 與 TEST 的影像 ID、JPEG SHA-256 和查詢類別組合互斥。只檢查完全相同位元組及已知 COCO ID，對重壓縮近鄰與模型預訓練資料無法保證排除。
- COCO val2017 影像曾在 Probe 2 作無標籤正規化池，也在既有 E2 當候選。新單位為查詢條件組合及該查詢在固定池上的排名。將既有正式 E2 的 1,573 組類別 key 正規化排序並全部排除。舊檔與排序 key 名單的 SHA-256 保存在清單來源欄。工作執行時只讀已凍結的新清單。
- 以 `prospective_rank.json` 的固定種子，對影像 ID 與類別 key 分別穩定雜湊後取模 3，落 0 者為 DEV，其他為 TEST。全部模型及評分臂在同一 split 共用全部候選、逐查詢相同標籤和 tie 順序。條件為 COCO 非 crowd 實例面積至少影像的 2%。依兩項類別組成主查詢，三項類別為描述。影像含全部指定條件，才計為標註相關。固定抽一個正例及共享 anchor、缺指定條件的硬負例。合成文字僅用 COCO 官方類別名。
- 鎖定的候選數量為 DEV 1,578、TEST 3,306。主評估 DEV 54 個 key、49 個指定正例影像群，TEST 236 個 key、178 群。三條件 DEV／TEST 為 51／176 個 key。TEST 主查詢 236 個中，161 個只有一個標註相關圖，故 mAP 罕見類別及單正例較多。最小構造量 DEV／TEST 50／140 個 key 與 35／85 個指定正例群。這些是 score-blind 的可行性門檻。樣本規模只能檢驗約數個 pp 以上的穩健增益，沒有實測差值變異可承諾 2pp 的檢定力。
- 鎖定 `configs/prospective_rank.json`、`configs/prospective_rank_manifest.json`、`configs/params_headroom.json`、`scripts/{prospective_manifest,prospective_rank,test_prospective_rank,common,e2_build_queries,headroom,headroom_models,e2_analyze}.py` 和 `slurm/prospective_rank.sbatch`。工作從指定 Git commit 封存的檔案執行。清單 SHA-256 `ade6a92afd554fe7ee9412b72d9c125b67024b95ae2124e58497360141092959`，執行前校驗其設定摘要、兩組固定模型配置摘要及全部候選檔案 SHA-256。建構清單時只讀公開標註、先前 E2 查詢 key、Half-Truths 排除欄與候選 JPEG 位元組，未讀模型分數。所有模型離線。正式只執行一次。

## 排名規則與控制

- `s(q,I)` 是凍結 CS-CLIP 文字與影像向量的 cosine。固定雜湊由查詢 key 決定拿掉哪項條件，剩下的 `a(q)` 是 anchor。推論規則 `s(q,I) + λ[s(q,I) − s(a(q),I)]` 在同一影像索引上計算。等價的查詢向量是 `(1+λ)f(q) − λf(a(q))`，可以用一次固定影像向量內積做全池排序。
- DEV 只用主查詢 mAP，從 `λ∈{0, 0.125, 0.25, 0.5, 1, 2}` 選最高者，同分取較小值。先寫入 DEV 選擇與全部格點 mAP，之後才開始 TEST 推論。TEST 的同一 λ 用於真 anchor 與打亂控制。λ=0 時兩做法精確重現原分數，仍照判。
- TEST 對照一為 CS-CLIP 原整句分數。對照二為將各 k 的 anchor 做固定一對一錯配，供體 anchor 必須與接收端完整查詢的任何類別都不相交，沿用同 λ。所有主查詢與既有三條件樣本均有合法錯配。另以 `λ=0` 逐分數相等及減去 query-only 常數後逐排名相等作兩個程式閘門。全部向量有限、各 split 完整候選、任一控制不過即不發正式 PASS／FAIL。
- 用 CORE-EMBED-2B 在同一組候選做全文 cosine 以提供固定索引參考，照樣報告 mAP、Recall@50 和 Success@50，判定不使用其名次。硬負例成對影像排名，以及同一正例影像中 `s(full,I)>s(foil,I)` 分別報告。後者只用同時有可由官方標註構造的有效文字 foil 的查詢，foil 以在硬負圖存在且在指定正例未達標註條件的另一類替換被拿掉的類別。成對訊號與全池排名各有不同母體，並列解讀。

## 正式判讀

- 唯一主量為 TEST `k=2`、對全候選的查詢宏平均 AP。每查詢對其所有標註相關圖計 AP。排序精確平手按升序 COCO ID，成對平手算失敗。比較真 anchor 對原整句及對打亂 anchor 的配對 AP 差。
- 每組比較用同一批 10,000 次配對群集 bootstrap，指定正例影像 ID 為群組，同一群內的查詢一併重抽，再按列數算宏平均。兩組比較使用中央 97.5% 百分位區間取得 Bonferroni 同時覆蓋。所有查詢共用固定候選池，區間反映在此固定池下以查詢為單位的變異，未另估新候選庫和重新選 λ 的變異。
- PASS 須同時對兩個對照差估計 ≥+2pp，且各區間下界 >0。任一對照區間上界 <+2pp 時，此組合規則 FAIL。其他為灰區。建構、來源或程式閘門失敗記未能測試。Recall、Success、硬負例、同圖文字 foil、三條件結果以及 CORE 比較只作描述。
- 本結果的適用範圍為使用 COCO 標註的二物件 conjunction、固定候選池及所試的 anchor 差分序則。標註缺漏及 2% 面積門檻可能把實際具備某條件的圖算成負例。結果無法直接外推到關係、屬性綁定、自由文字、其他資料集或 reranker。

## 執行

保留同一個離線模型環境 `icml/envs/vllm033`。H200 單卡，最多 1 GPU 小時。共 4,884 張 CS-CLIP 影像和 3,306 張 CORE-EMBED 影像，另有少量去重文字；兩模型從快取逐次載入。前一個同硬體作業 26393 的完整 Half-Truths 與五模型 E2 約 11 分鐘，現試驗預估 5–25 分鐘，實際耗時由工作記錄。

從題目目錄指定凍結 commit 執行 `sbatch slurm/prospective_rank.sbatch <40-char-commit>`。輸出 `results/prospective_rank/<Slurm jobid>/`，含凍結 commit、manifest／設定 SHA-256、套件版本、DEV 校準及 TEST 判讀，另存每個主查詢和各評分臂的 AP 供重核。`scripts/prospective_rank.py --preflight` 只查離線權重及候選雜湊，無模型推論。`scripts/test_prospective_rank.py` 使用合成分數、構造測試及已凍結清單檢查。

## 讀取紀錄

- [codex] 2026-11-15 06:09 主代理與獨立子代理已讀 `results/prospective_rank/26450/verdict_prospective_rank.json`。子代理第一次預覽一併顯示同檔前 11 個逐查詢項，判讀僅用事前指定的彙總。凍結版沿用 `5062061`。
