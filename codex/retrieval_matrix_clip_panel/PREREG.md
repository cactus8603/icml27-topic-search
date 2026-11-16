# [codex] Native-qualified CLIP panel

2026-11-16 21:46（台北）。本目錄接線 SigLIP2 的 P2 圖庫推論，不產生資格，也不以 CPU 合成測試冒充 P0。無模型效果讀取，無 GPU 執行，無正式 P2 plan。

模型必須先以固定 native P0 plan 完成 COCO 與 Flickr 兩個不同 benchmark，固定的原生方向與分母均保留，經非產生者從完整向量獨立重算，再由 `retrieval_matrix_native_p0.qualification.admit` 重建 PASS。P2 完整 model spec 必須逐位元組語意相同；不能改批次、精度、processor、variant 或 checkpoint。初版只支援已審的 SigLIP2 family，其他 CLIP 系須先完成各自 native P0。

共同資料是已獨立查核的 COCO development v2：完整 20,000 圖庫、正式連言 dev，否定 dev 僅診斷。COLA GQA/PACO 僅為固定基線跨資料集測試；使用者已同意不要求不存在的 COLA train。建 plan 與每次執行都重驗 data audit 並把實際 dataset entries 綁回 v2。不以舊 v1 或官方 test 挑新方法。

沿用固定的 CLIP encoder 與 processor。共同矩陣排名用 FP32 CPU dot、gallery 原序穩定同分；這是預先指定的 panel 協定，與 native P0 的 Google numpy quicksort 計分分開揭露。候選先取全庫 K=100；50/200 僅補充。ignore 只在評分時移除，不補足 K。COLA 主欄用 hard_list 上按 score threshold groups 計 AP，再按類別平均；GQA 2589 圖與原文1952圖差異、固定 caption adaptation、缺 seen/unseen 分割皆保留，不宣稱 exact COLA 重現。

建 plan 僅接受真資格收據與獨立 data audit；之後必須 commit plan 與全部 executing sources 才能送中央排程。失敗保留，不自動改設定、減批次或重試。GPU 工作由 root 的單一排程處理，不能由本目錄自行搶卡。

分析只產生 reports；讀取主結果前先凍結獨立 analysis PREREG，在該分析題目的 PREREG 登記實際讀取時間，保留來源 producer 的完整檔名／plan SHA／freeze commit。凍結且仍在執行的 P0 producer 不直接改其 PREREG；不能以追加日誌為由鬆綁原判準 SHA。此版本未登錄任何主結果讀取。
