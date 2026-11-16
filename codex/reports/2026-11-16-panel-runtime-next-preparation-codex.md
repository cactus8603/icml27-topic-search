# [codex] 修復後 MLLM P2 接線：CPU／source 交付

依root授權，在全新 `codex/retrieval_matrix_panel_runtime_next/` 完成最小兩family接線。沒有修改任何live／frozen folder、原模型、P0收據、排程或程序；沒有讀真Q／模型arrays／report／效果、GPU、正式P2plan或commit。原source阻擋與CPU重現另保留在2026-11-16-panel-runtime-contract-audit-codex.md。本報告是實作者交付，待root獨立source審查；不是正式P0數值資格、P2執行或研究判定。

唯一兩個family是core-qwen3-9943159（core2、core8、qwen2、qwen8）與qwen2-einops-9943159（gme2、vlm2vec2）。前者parentplan SHA0f74cb526e08952e05cc178e9fc2fba91d773701db7d92049a71661a142049b1，後者07a9b6859e17d12e99a00439327381655ec22b1369e2d27a9fa4a5f8a5ea3053；原parent與全部來源逐檔核99431594893bba94337b3befd10a7a483ba7374f Git bytes／SHA。CORE原plan沒有reviewed boolean；保留其原完整bytes與994獨立來源審查，不發明舊欄位。Qwen2保留原reviewed=true。

builder先呼叫真正原panel.Q.admit，再額外核回傳資格的exact parentplan entry／SHA／model／fullspec；不接受相同模型spec但另一parent或舊未修runtime的資格。隨後才讀已admit的三個原P0 report identity與各自processor.json，核processor payload SHA，要求三benchmark同signature／actualruntime與正確compat／einops修復marker。完成後才開dev-v2／COLA metadata，builder只回REVIEW_REQUIRED／reviewed=false候選。沒有真Q不建正式P2plan。

P2完整modelspec、原native settings（含seed宣告）、batch、processor、依賴、parentsignature與實際runtime都重新由真Q證據導出並binding。CORE使用同一個已凍結compat.CompatibleOfficialModel class，Qwen2使用同一個Qwen2OfficialModel／DoRA adapter class。私有matrix module用獨立model namespace，不import原panel／Qwen2 runner的頂層patch，canonical matrix module不變；若其他入口已patch shared model dispatch則明確fresh-process拒絕，不猜載入順序。建構後signature與已資格parent一致才可encode。

runtime重現994原函式語義：CORE保留原packages/modules/CUDA加compat marker；Qwen2保留PEFT／FlashAttention擴充、原完整environment checks與已釘einops overlay marker。執行時核原全部依賴SHA／distribution、overlay全payload與actual modules，actualruntime除中央GPU UUID外所有原欄位必須與qualified P0相同。launcher固定原隔離Python及依賴路徑，runtime再核sys.executable；只有中央owner/task單GPU可進。原P0未設定或記錄actual seed／TF32／matmul profile的界限明示於plan disclosure與PREREG；不悄改新預設，不補造原收據，也不宣稱此未記錄狀態已驗證。

source gate保留完整qualified parent source table，加原17e990ba6294b1962f67d0c9a33bccf291e43fbd panel/data/audit/qualification來源與本批七檔；每次重新真admit、完整Git/SHA與所有executing folder clean、依賴bytes、dev-v2獨立audit、exact四資料集entries均需通過。dev正式conjunction>=1000且20000全庫，negation僅診斷；COLA固定baseline test、原caption adaptation與threshold-group AP保留。排名FP32 CPU dot、gallery原序stable ties；K100主、50／200補充，ignore只評分移除不補K。P3明確不在本批。

16項新CPU synthetic tests PASS，7.560秒；原qualification／dataaudit／COLA共13項CPU tests PASS，0.002秒，合計29項。AST parse、bash -n亦PASS。新增測試包括六模型各自呼叫真正Q.admit邊界、無真Q先於任何audit／panel讀取即拒、另一parent／spec／seed／runtime／signature／依賴變更拒、三benchmark同processor/runtime、Qwen2 overlay、canonical globals無變與原class身分、載入次序明確failclosed、完整runtime receipt擴充語義與修復marker、同分排名及source／dataset缺漏。全部資料為臨時synthetic；成功admit為顯式stub，不能當資格。runtime測試只替換private closure的CUDA基底reader與假assignment，實際receipt-extension程式執行；canonical globals未patch，CUDA initialized=False。

```bash
CUDA_VISIBLE_DEVICES= PYTHONDONTWRITEBYTECODE=1 HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 \
TMPDIR=/ssd4/cactus8603/icml/hf_home/tmp OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1 \
PYTHONPATH=dl/retrieval-matrix-qwen2-runtime-fix/deps:dl/retrieval-matrix/deps/qwen2-transformers451:dl/retrieval-matrix/deps/qwen2-flash:dl/retrieval-matrix/deps/core-p0:. \
dl/retrieval3090/envs/retrieval3090/bin/python -B -m unittest \
codex.retrieval_matrix_panel_runtime_next.test_contracts -v
```

同一CPU環境另跑：`-m unittest codex.retrieval_matrix_panel.test_qualification codex.retrieval_matrix_panel.test_data_audit codex.retrieval_matrix_panel.test_cola -v`。

| 檔案 | SHA256 |
|---|---|
| contracts.py | 0fae908b6cefb2183d163e703c63c0ca07c45b34e794ec0c240b8ad593c9f940 |
| build_plan.py | 99adf9c85a3440efe8896ea7866d145a1162240eacd9ae876ed386fe4d4e88fd |
| runner.py | fecb98a5018768725108fa9eef004ac7cb9cd89567ec105ff74a97480834b72c |
| test_contracts.py | f73cd7ab663904d3b175b5f0ae1c155815b67bae3f328b22c646850d2982571f |
| run.sh | be411ec5107d66d4f284babce1197031587eb4d5ac1387c3b6c3cc60b99f2ebd |
| PREREG.md | cf92faf28baaf946a60b761e08f6e68c0db345eba03c7796c02dc5edbe5bfe7b |
| README.md | 20088acf44cbc946af8ccd439fff2d2118696111852bf78965cf361bf89d8a49 |

已停筆交root獨立review。尚無正式plan、真Q採用、P2 GPU或P3資格；未來完整P0 targetqualification與使用者定案仍不可由本次source／CPU通過替代。
