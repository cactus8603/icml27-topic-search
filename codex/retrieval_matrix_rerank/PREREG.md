# [codex] CORE 重排器官方 MCQ 首批

[codex] 2026-11-16。首批只執行 CORE-Reranker-2B／8B，完整 COLA multiobjects、SugarCrepe++五子集、NegBench COCO與VOC MCQ；原題、重複題與有效分母全部保留。此文件與機械 plan 在正式推論前由 root 獨立審查、commit，工作樹乾淨後才排程。

使用 CORE 釘版官方程式所指定 ms-swift generative_reranker engine/request/score，fp16、FlashAttention2、IMAGE_MAX_TOKEN_NUM1200、context1500；作者 checkpoint args 記錄 swift4.0.0.dev0，本批固定可取得且CPU核對通過的穩定4.0.0，完整差異見 protocols.json。image與query的實際模板由官方ms-swift與checkpoint args決定，不使用自製yes/no替代。

原 engine max_batch32 保留；本機每呼叫固定4pairs，減少3090記憶體需求。研究資料之前以純合成圖文驗 batch1與4的同一pair絕對分數差≤0.01；此門檻不看真實資料設定。失敗即停止，不換精度、量化、CPU offload或調到論文數字。每模型獨立快取完整58,891pair activated原分數，保留斷點與逐檔雜湊。技術成功與論文重現分別記錄。

主驗收依CORE Table8三個benchmark，容差各1pp：2B COLA0.724、SCPP0.839、NegBench宏平均0.646；8B為0.843、0.875、0.698。每個子集都須列出，不隱藏偏差；任何主驗收不合時先核協定／資料／環境並照實記錄，不按test效果調設定。COLA按CORE每張圖選正caption雙嚴格比較；SCPP兩正都勝負；Neg caption0嚴格勝其餘三。平手失敗。Neg先各算5914列COCO及5031列VOC，再做兩者等權宏平均；SCPP五子集等權。不同benchmark不平均。

本批只用公開官方固定基線；不選自家方法、不開自建 test。GPU只產生score；CPU分析另行執行。讀模型效果前先依協作規則登記，由未實作producer者獨立重算與驗收後再交使用者判讀。

讀取紀錄：[codex] 僅讀官方來源、原資料metadata、CPU技術收據及既有公開表格；沒有產生／讀取本首批研究效果。來源、模型、依賴、圖片、題目清冊均由機械plan與收據釘版。
