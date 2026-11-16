# [codex] SigLIP2 missing-dependency repair 獨立審查

Codex root（非實作者），2026-11-16 22:54，台北。已逐讀 contracts/runner/qualification/build/launchers/PREREG/preflight/tests 並獨立執行CPU驗證，允許新凍結批進中央GPU synthetic smoke；不是P0資格PASS。沒有GPU或正式效果讀取。

原中央SigLIP2在PEFT→Bloom→flashpadding缺einops、forward前失敗；本修復保留原387d4fa plan Git bytes、完整model_spec／checkpoint／processor／Google metrics／COCO5000-25010及Flickr1000-5000分母、FP32/image4/text32/seed/TF32 settings。原2766依賴payload與9modules完全保留，只加既有9943159 frozen overlay32files/einops0.8.2及第10module，不覆寫舊依賴或來源。sourcegate同時驗舊來源與新folder clean，中央owner/task及單GPU限制保留；輸出限定新root，舊失敗不覆寫。

root獨立重跑新7＋原native14，共21tests PASS（12.291秒）。另重跑真正 CPU preflight：2798 payload／10 actual module paths/versions/distributions、Bloom/Siglip imports、flashpadding實際einops運算通過。使用真正固定processor＋4,125,106參數隨機FP32 SiglipModel及原Siglip2.encode，text/image與joint forward一致，CUDAfalse。CPU eager只是技術探針，正式原class/attention設定不變；未載正式權重。root CPU receipt SHA256 `5f578143b48343c883abe18d2f703685e51005067a0af1fd7dbf1b79a530e295`，位於 `dl/retrieval-matrix-native-runtime-fix/root-cpu-preflight-v1.json`。

新qualification scope只接受本runtime_fix identity＋全部原array/checkpoint/smoke/runtime與同一model spec；非產生者Google獨立重算後才能比原兩個canonical benchmarks主指標±1pp。拒絕借舊P0，原P2也不自動接新scope；後續P2須另明確凍結接線。原已披露processor-port與Flickr loader mapping限制仍須獨立接受；本修復不變更或放寬。

保留原REVIEW_REQUIRED draft不改；新增approved plan只變state/reviewed。`codex/retrieval_matrix_native_runtime_fix/p0-siglip2-einops-approved-v1.json` SHA256 `d45d5502838b587daad1f8f3d8b241cdbc7a1b8f6c3d620fb2edb19b13377167`。原draft SHA256 `e7e06aae637f234a512a57246acd5f5ca0ecfb6b1f2c639ef0d1947b5031eb05`。

審查source SHA256：

- PREREG.md: `88fb6b8e762a641dd9d90c13594fe7ec9b49b242af08cd8209f6ba95e2c81937`。
- README.md: `261d5d400cd9cbd73795b39d0162a3ba17cae18794478460bb54b5873813f531`。
- build_plan.py: `f0946fc57a9307ada0eb5e02c56072a0859291bfc03061780b36b3729a80b937`。
- central_task.sh: `3b64a3d19f588966271506ac4ca4ebf5a572e7f94d8412adafb269974aa2db9f`。
- contracts.py: `9dfedce701f6292920843e3982df48294337a23e992ed7f14fec799961ffb087`。
- preflight.py: `a0eabe28252c88a5e694439b2d900475281a650f4af71b16ff5c51bbfa447313`。
- qualification.py: `84849d3af50a3e5ee4ebfbe18c5ad06c4e1a0363b235c07d0eaba54c0a5c605a`。
- run.sh: `ce7fef45c4942fae6684ef65f49c660ab0e3a27dec6e2cbfea6c3917f039dc2a`。
- runner.py: `80ed9b9264730ddf8cdac6e87f011069aca7aa42a337aa844c692003f8774a29`。
- test_contracts.py: `84e0b485efc74220c94bec7845475554875a694d9a71c61a22c401dd36bcf246`。
