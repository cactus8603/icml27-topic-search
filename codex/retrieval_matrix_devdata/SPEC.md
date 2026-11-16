# [codex] COCO train2017 否定開發面板 v1

[codex] 2026-11-16 20:27（台北）。承接報告64 P1的train2017否定開發資料需求，以及使用者「沿用COCO dev，COLA僅測試」決定。先提交規格／builder，再產生；不使用任何模型分數選題。只讀既有連言面板的dev圖庫與COCO原始人工標註，不讀test查詢或影像。

## 固定來源

- COCO官方 `annotations_trainval2017.zip`，SHA256 `113a836d90195ee1f884e704da6304dfaaecff1f023f49b6ca93c4aaae470268`。
- 連言v1的 `dev/gallery.json`，20,000張train2017圖，SHA256以v1 manifest及建置收據驗收。使用同一圖庫，不另外抽小庫。
- 影像狀態沿用連言v1：非crowd單實例segmentation area≥圖面積1%為present；有標註但僅小物件／crowd為ambiguous；完全無該類標註為absent。這是基於標註的操作定義，不能保證真實世界完全無漏標。

## 固定查詢與真值

1. 每題是k=2、3、4個正物件類別的連言，再加1個不同的否定類別；總條件數k+1。不得將本面板的k混作純連言面板總條件數。
2. 未加否定前的連言在圖庫內有6–200張present正例；加入否定後，所有正條件present且否定類別absent的正例為1–20張。
3. 至少5個困難負例：所有正條件present且否定類別present。否定條件因此實際改變相關集合，避免只是裝飾性否定。
4. 明確負例為任一正條件absent，或否定類別present；其餘非正例為ignore。小物件／crowd帶來的歧義不當確定負例。
5. 類別原名與冠詞形成固定模板：`A photo that contains a dog and a cat, but no person.`。正類別次序依固定hash，否定類別明示在but no之後；已知複數類別scissors/skis不加a。
6. 所有候選依 `sha256(seed:k:positive_category_ids:negative_category_id)` 排序，k=2/3/4 round-robin，若與已選題共享任何正例則略過，直到候選耗盡。正例互斥不宣稱完全統計獨立。
7. dev-small依另一固定hash，k各取最多84題；為dev-full子集，保持完整20,000圖庫。實際題數／群數以產生收據為準，不為達到1,000題改規則。

seed=`neg-dev-20261116-v1`；small seed=`neg-dev-small-20261116-v1`。此為新合成文字的開發資料，不是官方NegBench重現或未見確認集。

## 輸出與評估

輸出留在ignored `dl/retrieval-matrix/data/coco-negation-dev-v1/`：gallery、queries、dev_small、manifest。記來源／程式／各產物SHA與k分布、正例／hard-negative數、ignore數，所有題目計入分母。計分為完整多正例mAP與Recall；不能把全部其他語義符合圖當來源圖負例，也不混用NegBench的單來源圖MRR。

主K100，補充50/200；完整庫先取K，計分才去ignore，不為ignore補K。品質驗收是標註真值、邏輯與模板逐題一致性；沒有新增人工標註，不聲稱人工逐題確認。
