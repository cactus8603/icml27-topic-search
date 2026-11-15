# 組合檢索聚合器：理論與 LCSE 對照審計

[codex] 2026-11-16 03:56（台北）。本報告只檢查設計、程式與公開作者來源；未讀本輪研究結果或封存面板，未跑 GPU。既有提案來源為 `reports/2026-11-14-46-deepcheck-compositional-retrieval.md` 第 5 節；本次實作限 `codex/retrieval3090/logic_baselines.py` 與合成測試。

## 需要更正的主張

### 1. log-sigmoid 加總不是目前已成立的 LLR

原 `dev-3090/analyze.py` 的分數是 `Σ log σ(z_c / τ)`，其中 `z_c` 是同一條件在圖庫上的均值／標準差標準化，沒有用正負標註估計條件機率。這個轉換可以作平滑聚合消融，但不能直接稱為對數概似比。

若有可信的 `p(c|I)`，且明確加入條件獨立等假設，`Σ log p(c|I)` 才能解釋成分解的條件對數機率。再減去 query-only 的 `Σ log p(c)`，可形成相對於背景的分解比值，但對固定 query 的所有候選減掉同一常數，候選排序完全不變。prior 修正不能被當成同 query 召回提升的機制。

每個 `log σ(·)` 都小於零，因此新增一個條件總會降低其總分，連正確條件也一樣。它不滿足原提案「新增正確條件分數嚴格上升」的 A1。將這種分數用於固定兩條件的候選選擇仍有實驗意義；不能把該實驗當成 COR 公理的證明。

本次 API 使用名稱 `logsigmoid_cond`，並保留 `product_cond`。固定條件數時，乘積與其對數有相同的數學排序；兩者不是兩個獨立的新方法。乘積數值下溢時會產生平手，對數版是穩定的計算方式。

### 2. A1、A2 不推出跨影像的 A3

定義 `S(I,Q)=Σ e_c(I)`，且每個真條件的證據為正、假條件為負。這已滿足新增真條件加分、新增假條件減分。然而同一 query 的全符合影像可有證據 `(0.1,0.1)`，部分符合影像則為 `(100,-0.1)`；其總分分別為 `0.2` 與 `99.9`。A1、A2 是同圖跨 query 的性質；A3 是同 query 跨圖的性質，需要另外的假設。

即使每個機率對真／假的方向正確，乘積仍不保證 A3：全符合 `(0.6,0.6)` 的乘積 `0.36` 低於部分符合 `(0.99,0.49)` 的 `0.4851`。單看「正確方向」不足以控制其他已滿足條件的補償。

若全符合影像在每個條件上的證據都不低於另一影像，且至少一項嚴格較高，嚴格遞增的聚合器可保證排序；這是較強的 coordinate-wise dominance 條件。若每個真條件分數均高於共同閾值、每個假條件均低於該閾值，取最小值也能給出硬式 conjunction 分離，但真實模型未必滿足此條件。LCSE 另保留 holistic residual，不能自動沿用這個純 min 的保證。

### 3. 條件數 k 的通用分離界尚未證明

單純「影像嵌入可加法分解」不足以證明所有線性單向量評分都有 `1/√k` 的訊雜比衰減，更不足以證明 log-sigmoid 後訊雜比與 k 無關。需要先給出正規化、信號大小、跨條件與跨影像的雜訊協方差，以及候選抽樣模型。

一個可檢驗的反例模型是：每個已滿足條件經轉換後的分數為獨立隨機變數，均值 `m+`、方差 `v+>0`；缺失條件分數的均值與方差為 `m-`、`v-`。比較兩張獨立影像的 k 項加總，一張全符合、一張缺一條件，期望差為 `m+−m-`，差值方差為 `(2k−1)v+ + v-`。因此無論轉換是線性或有限飽和的 log-sigmoid，只要 `v+` 固定且非零，訊雜比仍可隨 k 衰減。飽和可能減少方差常數，但不消除一般的 k 依賴。反過來，若兩張影像的其他證據完全配對而相消，線性分數也未必有此衰減。

[How can embedding models bind concepts?](https://arxiv.org/abs/2605.31503) 的加法分解觀察不是上述 k 界；[The Limits of Binding in Dual Encoders](https://arxiv.org/abs/2608.15971) 處理附加可檢查假設下的 swap／巢狀深度等障礙，不能直接移用成新增條件數的定理。本次未做完整最新文獻排除，不能宣稱已找到或排除所有相同定理。

## 完整 AND LCSE 實作及來源

對照 [LCSE 論文第 4 節與附錄 A/G/H](https://arxiv.org/html/2607.23052)，並讀取作者 [eval.py 固定版本](https://github.com/SultanMo/factored-vlm/blob/8af1972ddfd03e8437d5f5c96d5a784e94b82d99/eval.py)。固定 commit 為 `8af1972ddfd03e8437d5f5c96d5a784e94b82d99`，Git blob 為 `b5aa7db8201538cc0997ed7ea4e01812006e5965`。

現階段只做 COCO 物件正向 conjunction，不宣稱覆蓋否定、OR、關係、屬性綁定或一般語句解析。API 接收 raw cosine，不能把原先 z-score 直接代入。

1. `p_i = σ(β(s_i−μ))`。
2. AND 邏輯分數使用 power mean；預設 `γ=−1`，即 harmonic mean。
3. `p_soft` 是每個條件機率的 arithmetic mean。
4. 最終分數是 `s_hol + [logit(p_logic)−logit(p_soft)]/β`。這個 reference correction 必須保留，單用 min 不能冒稱 LCSE。

作者程式的 concept score 由四模板產生：`a {}`、`a photo of a {}`、`an image of a {}`、`a {} in the image`。應對原始物件名稱套模板，逐模板正規化後取 cosine 的算術平均；不可把平均後的向量再正規化成另一個 cosine。本模組只接受已計算的分數，不做文字編碼。

另外遵循作者程式將 holistic 機率先截到 `[1e−6,1−1e−6]` 再逆映射，power mean 的輸入與 logit 亦作穩定截斷。NumPy 實作使用 float64，作者 reference 的 power mean 使用 Torch float32，可能有浮點尾差；沒有宣稱逐位重現作者所有 benchmark。

作者 CLIP 預設 `μ=.22, β=30` 可追溯到上述來源；它不是 CORE 或 CS-CLIP 的校準保證。本次提供同 backbone 的 holistic、原 cosine mean、原 cosine min、sigmoid product、log-sigmoid sum、完整 LCSE 六組分數。

## 獨立校準的界線

`fit_presence_calibration` 只接受來源標為 COCO `train2017` 的影像×概念 raw cosine 與二元 presence labels。呼叫者須提供完整 evaluation image 排除集合；實作檢查非交集，記錄影像清單、標註與分數雜湊、manifest 雜湊、每概念正例數、固定搜尋格及來源。驗收 artifact 亦可再次帶入實際 evaluation IDs 檢查。

搜尋格固定為 `μ∈linspace(−1,1,81)`、`β∈{1,2,5,10,20,30,50,60,100}`，取未重平衡的 image×concept Bernoulli NLL 最小值，平手按 μ、β 升序。本次 objective 是可追溯的 **train-only presence NLL 校準變體**；作者原來依多種 operator pairwise accuracy 選參數，兩者不能混稱。512 張圖可以作固定小校準集，不能單憑數量宣稱罕見類別校準充分；零正例概念須揭露。

結構化 metadata 只能檢查一致性，不能證明呼叫者確實提供正確來源；runner 仍須保留真實 COCO IDs、檔案雜湊與訓練分割清冊。不得用開發面板、留出測試或封存標註重新選 μ、β。不同 backbone 應各自校準，使用相同 calibration images、概念與 objective。

## 後續可驗證的窄主張

- 在相同 backbone、concept parsing、candidate pool、reranker 與 50 次 rerank 預算下，完整 LCSE／其他聚合是否提高 admission recall 及最終 mAP；額外圖文編碼、模板、索引及聚合成本均計入。
- 同時跑真條件與固定跨 query 打亂的條件，測試收益是否取決於正確條件資訊。訓練版權重與新聚合的收益分開歸因。
- [CS-CLIP](https://arxiv.org/html/2602.23906v2) 是訓練得到的 checkpoint；同一 checkpoint 上比較聚合器，才能避免把權重收益歸給聚合方式。
- COR（同圖、三種描述）與 retrieval（同 query、候選圖庫）分開驗證。固定兩物件的 admission 結果不能證明新增正確條件升分的 A1。
- k=1…6 的現象須控制物件頻率、圖像複雜度、正例比例與候選難度；在明確生成／噪聲假設前，僅報經驗趨勢，不報通用架構不可能定理。
- 開發面板只用於選一個固定策略。封存前固定策略、校準來源與成本核算，再作獨立評估。這些是可驗證假設，尚無本輪效果結論。

## 純合成檢查

2026-11-16 的 11 項 CPU 測試通過：獨立 scalar AND 公式、相等／弱 conjunct、單條件與交換性、官方 holistic 截斷與極值穩定性、同 backbone 消融、理論反例、非法輸入、train-only 校準與資料排除、artifact 指紋異動、非二元／無正負樣本拒絕。測試只用合成陣列，不讀研究結果。

執行：`PYTHONDONTWRITEBYTECODE=1 dl/retrieval3090/envs/retrieval3090/bin/python -m unittest codex.retrieval3090.test_logic_baselines -v`。
