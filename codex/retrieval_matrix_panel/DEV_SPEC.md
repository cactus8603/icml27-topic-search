# [codex] COCO 開發面板 v2：排除跨資料集測試圖片

[codex] 2026-11-16 20:51（台北）。此修訂只建新的開發資料，保留 Claude 的 conjunction v1 dev／test 與 Codex negation v1。先提交規格、來源清冊與程式，再產生資料；不讀任何模型分數，也不開自建 test queries。

使用者已決定沿用 COCO dev、COLA 僅測試。來源盤點發現 COLA PACO 的 train2017 圖片和 v1 dev 重疊 1,044 張，原生 CLIP 資格用的 Karpathy COCO test 另重疊 767 張。這些圖片不能繼續用於新方法的開發。Half-Truths 的 COCO source IDs 亦納入排除。原有版本只保留歷史基線／診斷用途；後續新方法選擇使用 v2。既有曝光不會因新版消失，論文仍須揭露。

## 固定圖庫

- 來源是官方 COCO train2017 instances，zip SHA256 `113a836d90195ee1f884e704da6304dfaaecff1f023f49b6ca93c4aaae470268`。
- 排除聯集：既有 conjunction v1 test 的完整20,000個 image IDs；COLA PACO 原 JSON 所有 image IDs（含 val，雖不屬 train）；Karpathy 原始 COCO `split=test` 的5,000 IDs；Half-Truths 固定 snapshot 全部三份 parquet 的 COCO source IDs。
- 只讀 v1 test 的 gallery IDs 與原 manifest，不讀 test query、模型或標籤結果。
- 其餘 train2017 圖片仍按 v1 的 `sha256("conj-panel-2026111601:pool:<id>")` 排序取前20,000張；不因題數或效果更換 seed。輸出按 numeric ID 排序，保持固定全庫。
- v1 dev 中合格圖片可重用，缺圖由官方 COCO URL 取得，驗解碼、尺寸、逐圖 SHA。原圖庫與收據不覆寫。

## 查詢

連言重用已提交 v1 builder 的三值狀態、k=2/3/4、正例1–20、hard negative≥5、hash round-robin、正例互斥、單模板。split 名固定 `dev-v2`，其他 hash seed 不變。小版每k最多84題、保持同一全庫。

否定重用已提交 negation v1 builder 的三值真值、k個正條件加1個NOT、未加NOT前6–200正例、加入NOT後1–20正例且≥5 hard negatives、原固定seed與模板。只替換為v2圖庫，不為湊足題數調規則。否定小版每k最多84題。若否定題數不足1,000，僅當診斷。

產生收據報告全部候選數、k分布、正例、ignore、各排除來源、原dev/test與各官方test的交集、所有輸入輸出SHA。正例互斥不保證統計獨立，也不宣稱保證2pp檢定力。若連言不足1,000題，照實記錄，不更改此版規則。

## 評估與版本界線

主K100，補充50/200；全庫取K後才於計分去ignore、不補K。完整多正例mAP與Recall保持全體正例分母。dev-small只作快速診斷，方法選擇用dev-full。COLA、Karpathy與Half-Truths只作固定基線／最終測試，不利用其分數決定本面板。

輸出為 ignored `dl/retrieval-matrix/data/coco-development-v2/{dev,negation}/`；不建立新的 test。此版消除的是明確列出之公開資料的影像ID重疊，不主張跨資料來源的視覺近重複已完全消除。
