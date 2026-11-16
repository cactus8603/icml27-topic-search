# [codex] Native SigLIP2 缺失依賴的隔離修復

[codex] 2026-11-16（台北）。只讀中央 native-siglip2 的 import 失敗技術 log：peft→Bloom→flash_attn.bert_padding 缺 einops，尚未進入正式模型 forward。本修復沒有讀取模型效果，不會重試原輸出或自行執行 GPU。

原 `387d4fa5293967705b93aeba235cb9ffe643cc42` 的 native P0 plan、model_spec、完整 COCO／Flickr 原資料及分母、batch4／32、FP32、seed、TF32禁止、原 Google metrics／ties／來源與披露協定原樣保留。模型仍為原 native.models.Siglip2，沒有替換 class、processor、公式或權重；未量化、offload 或改 batch。

新 dependencies.json 完整保留原2,766檔與9個實際模組，只加入已凍結 `99431594893bba94337b3befd10a7a483ba7374f` 的 einops0.8.2 overlay32檔及第10個模組。所有原檔、overlay及來源 manifest 的 Git bytes、檔案 SHA、actual path/version/distribution 均核對；不修改原依賴、不重新安裝套件。新 source freeze 包含完整原 source、新 facade／qualification／launchers／本預登記。

新 plan 初始 REVIEW_REQUIRED／reviewed=false；需 root 獨立審查、commit、工作樹乾淨及中央 owner/task、恰一 assigned GPU 才可使用新輸出。中央 launcher 保持 FULL_COMMIT PLAN siglip2 OUT 四個參數。原失敗輸出保留；沒有自動 fallback、換 precision／batch、GPU重試或讀結果。

CPU驗證涵蓋全部 payload與10個actualimports、Bloom／flash padding 的 einops實際運算、隨機小型 SiglipModel 原 class 的 text／image forward，以及原 native.models.Siglip2.encode 路徑及真官方 processor。隨機模型不載入正式權重、CUDA不初始化，不產生 benchmark 效果或資格。

新增依賴會形成新的 runtime identity；不能借用舊 P0資格。本 qualification facade 對新 plan 的完整原 arrays／checkpoint／smoke／runtime、全部分母和獨立 Google對照計算重驗；非產生者需另接受原已披露的官方port／資料mapping限制。收據使用獨立 runtime-fix scope，舊 receipt 被拒，原 P2 gate也不自動接收新 scope；未來接線須另明確凍結。數值門檻仍為兩個 canonical benchmark各主指標±1pp，CPU通過不是數值PASS。

正式模型效果讀取須另凍結 analysis PREREG，記錄本producer plan SHA與commit；本producer bytes保留不改。本修復不授權本代理 GPU 執行、排程、重試或效果讀取。
