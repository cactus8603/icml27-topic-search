# [codex] CLIP P2 接線獨立預檢

[codex] 2026-11-16 21:54（台北）。審查者：`official_p0/p3_contract_review`；接線實作者：root。結論：**目前程式接線無剩餘阻斷，可凍結程式；不代表 SigLIP2 已通過 P0，亦不授權尚未建立的正式 P2 plan 推論。**

本次只讀新 `codex/retrieval_matrix_clip_panel/`、native P0 資格介面與既有共同 panel 的程式／凍結規格，執行 CPU 合成測試。沒有讀取真實 P0/P2/P3 效果、沒有 GPU 推論、沒有新增正式 plan、沒有修改接線程式或 commit。

## 資格、模型與父計畫

- native P0 是 schema 2，原始 plan 狀態保留 `NOT_RUN`；P2 builder 以新 phase `P2-qualified-fixed-clip-baselines` 和 schema 1 接到共同 producer。原始 `NOT_RUN` 字串不能當資格：builder 在打開 development metadata 前必須呼叫實際 `Q.admit`；執行各 stage 時再次呼叫。
- `Q.admit` 重建兩個不同 canonical benchmark 的資格，綁原始 report／完整向量、不同 producer/reviewer、獨立重算與完整 model spec digest。CLIP `contracts.admit_models` 再核模型 key、family、父 P0 plan 的完整 path/SHA；MLLM 的三資料集資格表沒有套用在 CLIP。
- P2 直接引用 `retrieval_matrix_native_p0.models.Siglip2`，沒有換成 extra 的獨立準備 wrapper。純 import 不改共享 runner；只有 executable `install()` 接入該相同類別及 CLIP validator。
- parent 與 P2 的 `dependencies` entry 現在均須等於固定的 `retrieval_matrix_extra/dependencies.json`。seed、停用 TF32、batch invariance 門檻均須與 parent 一致；完整 spec 仍包含 FP32 loader、checkpoint、processor 協定與原批次。

## 執行、來源與資料

- runtime 固定 seed `2026111665`、FP32 matmul `highest`，關閉 CUDA/cuDNN TF32；重用 E 的依賴驗證。E 會遍歷依賴表的**全部 9 項 modules**，比對實際 import path 與 version。
- 最新 runtime 在 import torch 前要求 root 中央排程的 owner/task/可見 GPU 設定，再要求恰好一張被分配的 GPU。CPU 分析不進入此模型 runtime。
- source gate 要求完整 commit、plan／所有宣告 source 的 Git bytes、SHA、乾淨工作樹及固定依賴 payload；native／extra／CLIP 目錄的來源表不能省略。parent plan 本身也包含在新 P2 source table 的 native JSON 範圍。
- builder 和每次執行都重驗原 panel `data_audit.verify`，再以 `bind_datasets` 把實際 conjunction/negation entries 綁回 development v2；20,000 全庫、query 數、gallery/query SHA、圖片收據不可改成 v1。
- 沿用共同 panel 的全庫 FP32 CPU dot／穩定原序同分排名，與 native P0 的 Google quicksort 計分差異有明確揭露。ignore 只於 CPU 計分移除且不補 K；COLA 保留 hard_list 上 threshold-group AP 與按類別平均。GQA 圖庫差異、caption adaptation、缺 seen/unseen 等限制保留，不宣稱 exact COLA paper 重現。

## CPU 驗收與審查更正

- 最後修版 CLIP panel：**9 tests PASS**，涵蓋缺真資格、錯 family/model/parent、修改 forward/smoke 設定、漏依賴 source 與繞中央排程的早拒絕。
- native P0：**10 tests PASS**，包含完整原始 caption 分母、雙資料集、方向／tie／FP32、原 arrays 損毀及非產生者資格等合成驗收。
- 額外獨立合成 probe 逐一替換依賴表 9 個 module 的 import path，**9/9 皆拒絕**；probe 結束時 `torch.cuda.is_initialized() == False`。CLIP shell 語法與所有 Python AST 檢查通過。
- 審查中曾誤把 E 補錄 receipt 的三項 module 誤認為驗證範圍，提出 runtime 缺少其餘依賴檢查。重讀原版第 72–76 行及上述 9 項 probe 後，**撤回此誤報；原版已有完整依賴驗證，並非後續修正。**
- 結果讀取登記與凍結 producer PREREG 的交界已釐清：另凍結 analysis PREREG，記實際讀取時間及來源 plan SHA／freeze commit，保留原 producer bytes。當前仍為零主結果 exposure，不藉追加日誌放寬原判準 SHA。

尚待真實 native P0 完成、獨立資格 PASS 後，才能建立並檢查正式 P2 plan，commit 全部執行來源及 plan，再交 root 中央排程。以上 CPU 驗收不構成官方數字重現或模型效果證據。
