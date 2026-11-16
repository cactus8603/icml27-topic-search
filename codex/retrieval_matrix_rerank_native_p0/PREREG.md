# [codex] Native reranker MCQ P0 預先登記候選

[codex] 2026-11-16（台北）。本模組尚未執行模型推論或讀取模型效果。本次只有官方來源、CPU processor/template、合成驗收與 REVIEW_REQUIRED 計畫，未創造任何實測 PASS。

每個 variant 固定完整 checkpoint revision、payload SHA、backend source／processor／原 prompt、原 score scalar 與 dtype。Qwen CORE-Swift 和 Qwen official-native 分開，通用 MCMR-Qwen True/False 與 yes/no 分開；不能用同權重名稱轉借另一協定的表數。來源 target 描述與真實缺口在 source registry 固定，沒有兩個相符官方 MCQ targets 者維持 TARGETS_INCOMPLETE，連此 P0 GPU smoke 都不放行。

三個原始 MCQ benchmark 全保留。COLA 使用各 released branch 的 strict方向，Jina是image-choice，CORE/Qwen是text-choice；同時輸出另一軸僅作diagnostic，不能挑較好一軸驗收。SCPP兩個正caption都須嚴格勝負caption，五個subset等權宏平均。Neg caption0須嚴格勝其餘三個，COCO／VOC等權宏平均。平手失敗、沒有跨benchmark混合平均、沒有因重複圖或題刪列。

原始資料 full rows／pairs／images、文字、檔案路徑、圖與metadata SHA、順序保持；分數生產前逐圖SHA／bytes與decode驗證。資料缺失或模版解析錯誤停止，不跳題、不補0、不改分母。MCQ沒有商品metadata，通用Qwen使用empty Candidate的image-only適配已事先揭露，不能宣稱MCMR exact重現。

固定實際呼叫：CORE-Swift engine32；Jina COLA每題2+2、SCPP每題3、Neg每題4；Qwen-native逐pair process；通用Qwen逐pair、原True和False分支兩次forward。storage32題對SCPP／COLA／Neg形成96／128 pairs，整除Swift32，不改batch邊界。每個原呼叫先保存immutable raw transaction，array/state後提交；中斷後驗原raw並直接恢復，已保存分數不重跑。原 scalar用float64容器無損保存，原content list／scalar／兩branch logprob另存，不rescale、不裁切、不量化／CPU offload。

GPU只可由root中央排程指派，owner/task與恰一assigned GPU、固定依賴與原模組path/version/SHA、source／plan完整commit及乾淨工作樹全部通過後才可執行。synthetic smoke比batch1與原batch，原activated分數maxabs≤.01；失敗保留，不自動換batch、精度或backend。calls／invocations先記STARTED，再記成功／可處理失敗、pair呼叫次數、成功呼叫可確定的forward數、未完成呼叫的forward預算、時間與peak；硬殺未知成本明示，不能當0。

模型數值驗收固定至少兩個不同benchmark在官方targets±1pp，三個全保留。數值符合只叫NUMERIC_MATCH；非產生者需獨立核原arrays/raw／全部分母、完整model及variant／環境／sourcefreeze，並重算 strict comparisons，才可形成正式PASS。另一cutoff、方向或同benchmark兩個subset不算兩個benchmark。正式效果讀取另以凍結analysis PREREG記錄原producer plan SHA／commit，保留本producer bytes。
