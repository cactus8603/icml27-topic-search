# [codex] CLIC paired native P0

本模組只準備官方 CLIC-LAION/CogVLM 的 ViT-B/32、ViT-L/14 與 CLIPS ViT-L/14 三個固定公開 checkpoint。權重、revision、tokenizer與程式來源沿用已封存 extra 取得收據，不將相近命名或其他資料版本當作同一模型。本輪只做 CPU／規格／來源／資料轉接，不執行 GPU、不自行派工、不讀主結果、不製造資格 PASS。未來必須由 root 獨立審查、凍結新 plan/source，再由唯一中央排程；任何失敗保留且不自動重試。

兩個資格 benchmark 分別為完整 SugarCrepe（7,511 題，七子集）及 SugarCrepe++（4,757 題，五子集），共用原 COCO val2017 影像。子集、Replace/Swap/Add 與 ITT/TOT 方向都不能各算一個 benchmark。所有官方 raw caption 原樣送作者 tokenizer，不能套 retrieval 的 pre_caption30、預先 lower-case、重新產生或清除相同正負例。SugarCrepe++ 本身改過部分原始正／負句，不是在此處把 SugarCrepe 與 paraphrase 合併。沒有缺圖排除、抽樣或追加候選。

作者 paired 評估逐題呼叫每張圖與每句文字，故固定 image/text batch 均為 1；extra 原通用取得 wrapper 的 4/32 batch 不沿用。保留每一原題的 image embedding，即使圖檔重複也不去重；完整 counts 同時明列 paired image rows 與 unique physical images。模型 inference 介面只接受路徑或原句，不接正／負角色或標籤。CPU scoring 才使用原題 caption roles。

採固定作者 OpenCLIP2.31.0 factory、固定 HF 配置／權重本機路徑、FP32，原始 load_encode_img/load_encode_text 函式正規化亦在 FP32。B32/L14 使用各 HF 配置的 OpenCLIP tokenizer；CLIPS 使用作者 clips.CLIPS_Tokenizer／CustomTokenizer 與原 vocab.txt（WordPiece，BOS1、EOS2、末尾 class101、長度80）。重要更正：舊 extra 的小寫 alias 不會觸發 factory 的大寫 CLIPS branch，會改走普通 HFTokenizer。該舊技術預檢不是 CLIPS 原生資格，本模組保留此差異並使用正確作者分支；不改舊 frozen bytes。模型本體依作者 HF modelcards 使用 OpenCLIP，CLIPS fork只供作者 tokenizer。

SugarCrepe 原碼以 normalized vectors 逐元素相乘後 sum，strict positive > negative。SugarCrepe++ 再對 normalized vectors 用 torch.nn.CosineSimilarity(dim=1, eps=1e-6)：ITT 要 image-p1 與 image-p2 均嚴格大於 image-negative；TOT 要 p1-p2 同時嚴格大於 p1-negative 與 p2-negative。所有 equality 都判錯，不能任意排序或放寬 tie。每子集以完整筆數計 accuracy；Add/Replace/Swap 是子集 accuracy 的未加權平均，不能改成合併題數的 pooled accuracy。所有主要 grouped metrics均需距同 variant 的已發表值 <=1 個百分點，兩個 canonical benchmark 都通過才有資格。

GPU producer保存 normalized FP32原始 image/text arrays、原 batch1 checkpoint/state SHA、以及逐題作者 FP32 scalar score arrays（六欄定義在 profile）。向量按照 modality 分組快取後，在 GPU 逐題重放相同 scalar 公式；沒有合併模型 batch。此計算順序差異事前揭露。作者主程式 cudnn.benchmark=True 保留；本環境使用已釘版 Torch2.6cu124，而作者README為Torch2.2.2cu121，TF32在本機明確關閉，不能聲稱 GPU kernel 逐bit相同。作者沒有提供每個 released checkpoint 對 paper 表格 training seed 的完整映射，需獨立審查固定 release 與 LAION/CogVLM variant 的對應。

audit/analyze/review/qualify 都是 CPU 階段。獨立 reviewer 必須不同於 producer，重核完整原題／圖檔 SHA／向量與逐批 receipts，另從原始向量以 CPU Torch FP32 實作重算，逐題 strict events 必須與 GPU score arrays完全一致，任何差異 fail closed，不能看效果後改精度或 ties。review receipt 固定 NumPy及Torch實際 module path／version／distribution、計分 source SHA及本 facade SHA；admit 重新核全部證據。單純 CPU／meta technical PASS 不能當 P0 PASS。

來源與原文表格列於 SOURCES.md，精確 targets／分母／profile／payload SHA皆在每模型 plan。新計畫與本目錄尚未經 root 凍結前，central_task.sh 僅是可審草稿，絕不送出。
