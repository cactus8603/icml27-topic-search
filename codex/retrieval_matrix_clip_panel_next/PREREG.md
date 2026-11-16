# [codex] 新 qualification facade 的固定 CLIP P2 接線

[codex] 2026-11-16（台北）。本目錄僅 CPU 準備，沒有 GPU／排程、真實效果讀取、假資格或正式 P2計畫。建立計畫前先呼叫 constant allowlist 中真正的 family-specific admit；無資格即拒，不能先開 panel資料。paired CLIC 三模型只走5675ef59新paired runtime-fix qualification，SigLIP2修復環境走f3f787ee runtime-fix qualification，不用原SigLIP gate驗paired、不混用舊runtime。尚未獲獨立sourcefreeze的adapter一律拒絕載入。真正有資格後的P2候選仍為REVIEW_REQUIRED／reviewed=false，須另獨立review、FROZEN與commit才可執行。

完整parent model_spec、權重／revision／processor／raw tokenizer／原class／precision／batch／settings／依賴來源與actualruntime保留。新paired模型仍batch1，直接重用新paired runtime的factory與僅移除volatile repr地址的same signature policy；原PairedModel class／encode不改。SigLIP2仍原FP32 batch4／32。實際class/processor/tokenizer signature須與每個已資格P0 benchmark相同；P2不得量化、offload、改seed、TF32、cudnn設定或縮批。adapter名稱、模型集合、完整source entry／SHA／來源commit固定於allowlist；新修復協定需另明確加入，不能臨時載入外部module。

以private module重用4311a9d原clip_panel及matrix runner，不修改既有frozen R／E／native／paired modules或檔案。每次執行重新真admit、source完整Git bytes／SHA與所有executing folders clean、完整dependencypayload、devv2獨立audit與實際dataset entries皆必須相符。GPU需中央owner/task、單一assigned GPU；失敗保留，不自動重試或換設定。

COCO development v2保留20,000全庫與正式conjunction dev；negation僅診斷。COLA GQA／PACO僅固定基線test，不訓練或選方法。復用原COLA caption adaptation／hard_list threshold AP及差異披露，不聲稱exact COLA table重現。主K100，50／200補充；先全庫排名，ignore只在評分移除、不補候選。

Panel排名明定FP32 CPU dot與gallery原序stable ties，獨立於Google原生quicksort或paired GPU scalar table公式；保留完整parent encoder資格不表示panel分數是原論文重現。P2結果不可回流選COLA方法。P3另有未接線的qualification schema，這批不實作P3。

CPU合成負向／receipt stub僅用於測試真正adapter接線，stub置於自建暫存並清理，不會保存成正式PASS或P2plan。正式效果讀取須另凍結analysis PREREG、記錄原producer plan SHA／commit與實際讀取，保留原producer bytes。
