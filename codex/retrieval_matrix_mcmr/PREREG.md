# [codex] MCMR 原生全庫基線批次

本題只測已發布的固定模型，不開發自家方法。唯一 metadata renderer 是本目錄 SPEC 的 MMEB-V3 官方實作；原始資料為 3,997 queries、104,981 商品、103,176 互異圖。同一圖片不同商品／metadata 必須各有一列與一個融合向量，不能按 path 去重或用 path→text 覆蓋。

此批採 CORE 官方模型／processor／i2t_emb／fused_emb 與 native BF16 normalization；融合時同時傳圖片與完整文字，固定 `max_length=4096`、`truncation=False`。CPU tokenlength 稽核先完成且須符合模型 context。checkpoint 身份包含每商品完整 `{id,path,text,sha256,bytes}`，不讀 qrel 決定候選 admission。VLM2Vec 使用已驗收完整 DoRA；因 CORE fused dispatch 漏列其名稱，以相同 generic Qwen2 分支的 `gme` 路由，明示修正，不用裸 base。

固定 image/fused batch=1、query/text batch=32；相似度依 CORE 原 MCMR default query block=1,000，以 native BF16 GPU matrix product 後直接 `torch.topk(k=100)`，保留來源 corpus 列序，不用 float32 scores 改排，不以模型結果挑 renderer、batch、欄位或查詢。保存每題 native top100 indices／scores，僅作 CPU 分析。score 矩陣按 block 暫存，不持續保存整個 3,997×104,981 的冗餘矩陣。

v2 在首次GPU前由非實作者審查修正兩項執行細節：topk必須對每個一維score row呼叫，與官方CUDA平手呼叫形狀相同；磁碟checkpoint每64個商品提交一次，compute內仍逐商品batch1，完整每商品identity與列序不變。避免104,981步反覆重寫完整歷史state的二次方IO；若中斷只重算尚未提交的最多63個商品，失敗與resume wall time照實計入，不改模型精度、輸入或batch。

CPU 報告固定列 Recall@1／5／10／50／100（pytrec 的 relevant fraction）、nDCG@5／10、MAP@10、MRR@10。MRR@10 使用 native top10 admission，交 pytrec 的 reciprocal-rank scorer；另外固定列 native torch ordinal Hit@1、fractional Recall@1、MRR@10，揭露 score ties 在 library 再排序時的差異，不選較好版本。全部 3,997 題計入分母，不丟零命中。CORE Table 5 targets僅作差異對照：因 CORE 缺原轉檔腳本，**不得稱 strict P0 exact match 或用此欄獨自通過 strict P0**。

技術 smoke 僅用合成圖片與合成 metadata／query，驗向量形狀、有限值、native normalization 及 batch1 對固定 batch 的數值一致性；另以合成 batch1 超過 4,096 tokens 的融合輸入確認 truncation=False 不暗中裁切，以及完整 context 推論可行。不讀原資料標籤或效果。來源／環境／模型／資料SHA、plan及本文件必須 commit，工作樹乾淨後才可 GPU 生產；root 審查並送出，本代理不自行送GPU。

成本保留每次 invocation（含失敗／resume）的 wall time、模型載入、各 checkpoint 累積 compute time、similarity／top100成本、GPU peak allocated／reserved。整輪成本不當作單題線上延遲。讀新模型分數與 CPU判讀前按協作規則另記已讀；獨立審查者重算後交使用者判讀。
