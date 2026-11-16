# [codex] Qwen2 einops 修復：獨立 CPU 啟動前審查

審查者：Codex root（非實作者）。時間：2026-11-16 22:39，台北。

結論：本次 P0 embedding 與原生 MCMR 的隔離 einops0.8.2 overlay 可進入凍結後中央 GPU smoke；沒有變更原作者模型、完整 DoRA 載入、processor、模板、精度、batch、資料或計分。這是技術啟動審查，不是 P0 重現資格或研究效果通過。

逐讀 contracts.py、runner.py、build_plan.py、兩個 launcher、preflight、依賴檢查與測試。contracts 逐欄比較原 frozen parent 的 models/datasets/settings/dependencies/phase/gate/fixed_baseline_test，逐 SHA 核舊依賴及新增32檔 overlay，核實際14個 module 的 path、version、distribution。原依賴 payload 不被覆寫，overlay 只加入 einops。入口強制中央 owner/task/單卡，拒絕舊失敗輸出位置；embedding 沿用原 Qwen2 class，MCMR 沿用原 NativeOfficialModel，沒有改成裸 base model 或更換 scoring。

root 以隔離 overlay、原 qwen2-transformers451/qwen2-flash/core-p0 路徑，CUDA_VISIBLE_DEVICES 空，獨立重跑 test_contracts 與原 test_adapters，共10 tests PASS（14.967秒）。實際 check_dependencies 對 MCMR plan PASS：2544個原依賴檔、32個新增檔、14個 actual imports、Bloom/Qwen2 classes 可載入，cuda_initialized=false。另核實實作者 CPU 技術收據（只含合成 tiny models/processor/DoRA roundtrip，不含 benchmark 效果）。正式完整權重與 FlashAttention2 仍須 GPU smoke，不將 CPU eager probe 視為正式替代。

技術收據 cpu-preflight-v3.json SHA256 abb4af17ca34b5b655a42ee92d4870b64d50797bd0c8fe51b1fb417898c29ad6；cpu-validation.json SHA256 9bcc39b9bd78cbabbb6c8824ed46eadb9be0a3dc20a615c868f385817e88f8bd。未讀主結果、未 GPU、未改舊失敗收據。root只將兩個新 plan 的 state 設為 FROZEN、reviewed 設為 true，無其他欄位或來源變動。

凍結 plan SHA256：

- p0-qwen2-einops-v1.json: `07a9b6859e17d12e99a00439327381655ec22b1369e2d27a9fa4a5f8a5ea3053`
- p0-mcmr-qwen2-einops-v1.json: `2e525b5698782f2abd89ded3042e3416886a3c666b4d470c77bf888fc8dbb9a0`

審查 source SHA256：

- PREREG.md: `9a7fec2661a3c2a3583f896fd740a55bd7785c7fbaf3828490da8d6f7f9e327c`
- README.md: `329d8da8dc963f558ec5ded7c8596a7ef0a6d2ba7e1ecdf404db64992a46d4e3`
- build_plan.py: `4ad39750ae5e4cd177021dfa9fe3df8951deedf0567c9be8b9333808d6b358c3`
- central_task.sh: `d603ddec31f8bc95423ccf8505ca6a8325e32e333f671fd5e332a9db68b42abb`
- check_dependencies.py: `e775b44282a77b0932ed062708c638581444c499d2f00e4a0769a9f5eb796053`
- contracts.py: `031046bf87c67c136d769e06d6299aa4d57ca65d9dbcc011493883c2c0c9cfd0`
- preflight.py: `5801859deb78485bbb46cef280fbc405b34b8e059b829ff6e117453f5d552859`
- run.sh: `ba7089ca60866ef7e94589333524934d333a94135100f5a42be25ae1faaeae9d`
- runner.py: `b0601934385372e27625947722b15377b24d77211bb5bcc5ca96303001e9fd30`
- test_contracts.py: `d1afefec94249c0ea26902dfda8946ea810efee1c0708d6261e08d08571377c9`
