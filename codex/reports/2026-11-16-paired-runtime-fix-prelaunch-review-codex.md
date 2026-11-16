# [codex] CLIC paired runtime修復獨立啟動前審查

Codex root（非實作者），2026-11-16 23:12，台北。已逐讀 contracts/models/runner/qualification/build/launchers/PREREG/preflight/tests，獨立重跑31項CPU tests PASS（13.153秒）及完整新preflight PASS，CUDA false。允許新凍結版本進唯一中央GPU smoke；尚無P0實測資格，沒有GPU或正式效果讀取。

保留54f544c三個CLIC paired parent完整model_spec、兩benchmark7511/4757原題、FP32 batch1、raw captions、tokenizer、encoder class與formula、strict events與macro targets、seed/TF32/cudnn政策。只新增已凍結9943159的32檔einops0.8.2＋第10actualmodule，原2766payload/9module完全保留，不改任何frozen父來源或依賴。新Q以private paired engine驗原canonical科學規格；投影只用於驗原dependency/source欄位，真正array/scalar/smoke/runtime證據仍是新plan，沒有以Google scorer替代paired計分。

另外修復原transform repr含函式記憶體地址、跨smoke/index process會誤拒的收據問題。factory仍返回原PairedModel exact class，不改其encode、transform、tokenizer或model參數；只在新instance的signature方法刪除function repr address，函式名稱及所有其他transform設定保留。此policy有固定source SHA與runtime identity；後續P2須重用相同factory及policy。

root真CPU預檢直接import flash_attn.bert_padding並做pad/unpad運算，並核三plan的2798files/10actualpaths/versions/distributions；不依賴CUDA hidden會跳過GPU條件import的舊9-import測試。兩個獨立process構造CLIPS真transform/tokenizer，raw repr地址不同、正規化signature完全相同，且等於先前真權重CPU純色/空句helper收據。另重驗三個完整model spec、原modelsource、tokenizer/helper/checkpoint payload等同性鏈，不重新跑相同真權重forward。root preflight SHA256 `d32633f057afd5c0f2c9d44cd0a3d435315b48dd0235feffcdc84a97337372a9`，路徑 `dl/retrieval-matrix-paired-runtime-fix/root-cpu-preflight-v1.json`。

新qualification scope official-paired-runtime-fix-p0-qualified拒舊paired或Google資格。正式需原全部GPU scalar/vectors/batch/state/smoke/source/runtime與獨立CPU strict事件及metrics重算，兩canonical benchmark主指標仍±1pp；未讀主結果前不預先宣告PASS。原trainingseed映射／環境差異與cache順序限制仍明示。新輸出限定隔離root，不能覆寫原失敗、不fallback或減batch。

保留三個REVIEW_REQUIRED draft，新增approved plan只改state/reviewed：

- p0-clic_b32-einops-approved-v1.json: `f81ec57f7ea40bd913cb0a5f19774779498b17fee2e2599ea8340fb24ee430e5`。
- p0-clic_l14-einops-approved-v1.json: `efd8fbbe8bbd493cc473a52363cb0a9036dbe73b8c91414427ee0b1b42ff64ce`。
- p0-clic_clips_l14-einops-approved-v1.json: `5b5c4e33f06d0ec1c8b4df0f105a5c6f09a2ea896b669cd558e10044a2302e8b`。

審查來源SHA256：

- PREREG.md: `c50403ad08b1f5e43d1172e7c2b4528ef9f7509d907bf6ef447d367badec5a07`。
- README.md: `1429d11e14cd2a38908e194ec542184138f3e062fe8588cda35cc7dff1358635`。
- build_plan.py: `1fc5e8f2c18488c20ec1bfd59c2b9625ded07f8c1566a0db8c8d908f8cce66e0`。
- central_task.sh: `bd86785e68cc0b1e1f6a68c4c462feec2f20a5d39412e0ce03720f1d79bf5495`。
- contracts.py: `309842bf93a96bda1759436a4b1fd839dcaa0c543a45cfab80e54f33d9a26386`。
- models.py: `ef9f21bfc168f43db055f75ac765bb56d14956f3fa6603aa61cc838fdbed1e93`。
- preflight.py: `a4f1916507fcf96c055b928be166f8caea47098bbacf837f353fc0362f58f39e`。
- qualification.py: `19e4087b797ad7717712aff8c53a9c0fe300b3b9da4844ce1c62dd54bd6b3b5c`。
- run.sh: `6100dfacf2619453c768ee09f2d9a5d34d238f0266cdc9c9692d7b41aeff74fd`。
- runner.py: `7c0a3ad50972c3c2872cc408b71ac207201205c383ba7531ab0b3a0036da9298`。
- test_contracts.py: `b7934cb1f154024b600a6c18b71fbacf82e765f36aa613b86b623148dd031334`。
