# [codex] CLIP panel next：獨立 source／CPU prelaunch 審查

審查範圍為p3_contract_review實作的 `codex/retrieval_matrix_clip_panel_next/` 七個來源檔及其呼叫的已凍結qualification、runtime、model factory與舊panel資料／排名介面。本人未實作本批P2接線。結論：目前未發現source／契約層的blocking問題，可交root獨立定案及source freeze；本報告不是任何模型的P0數值資格、P2結果或GPU啟動授權。

本輪只讀source、原plan/spec/來源commit等非效果metadata，並執行CPU合成測試。未讀真qualification收據、模型arrays／report／效果，未執行真正admit重算、建立正式P2plan、載完整模型或使用GPU、排程、程序signal、commit。合成stub全部在hf_home/tmp測試目錄，測完清理；沒有保存可誤用的資格PASS。雖本人實作過部分底層原runner，這次只審別人新寫的接線，不對原runner的數值資格作獨立判定。

資格入口為固定allowlist與完整來源SHA／Git bytes核對。paired-clic-einops-v1只接受5675ef59cee99fa6d28e12f07501eeb157c321bd的paired runtime-fix資格，三個模型為clic_b32、clic_l14、clic_clips_l14；siglip2-einops-v1只接受f3f787eeb193aad8dcc6f2ee43e9609ddf5b47eb的native runtime-fix資格與siglip2。兩個facade的實際檔案分別核SHA19e4087b797ad7717712aff8c53a9c0fe300b3b9da4844ce1c62dd54bd6b3b5c及84849d3af50a3e5ee4ebfbe18c5ad06c4e1a0363b235c07d0eaba54c0a5c605a，均與原commit逐bytes相符；import後還核實際module路徑。舊paired／舊Google scope、任意module與交叉family不能替代。

builder先真正facade.admit，再完整binding，才呼叫development audit與panel／COLA metadata reader；runner公開入口亦先admit，private load_plan與source_gate再次admit。真正facade會重建原完整P0、independent review及完整spec資格，不只是信任state字串。paired兩個不同benchmark與SigLIP2的COCO／Flickr資格責任保留在各自原gate，不以子集數或panel結果補資格。無真資格不能產P2plan；成功builder也只回REVIEW_REQUIRED／reviewed=false，須另審查、凍結與commit。

P0→P2完整model_spec digest、parent P0 plan entry、原settings、dependencies entry與各已資格benchmark的實際model signature全部綁定。保留原FP32／raw tokenizer、no quantization/offload、seed／TF32及batch：paired1／1、SigLIP2 4／32。實際construct沿用paired修復factory或原Siglip2 class；paired factory僅依已凍結同政策移除signature內volatile function repr地址，encode與原class不變。P2construct再核signature及全模型CUDA／FP32參數，runtime直接呼叫各自原parent函式，核actual module path/version/distribution與原依賴payload，重設原seed／TF32／matmul profile，paired維持cudnn benchmark設定。P2原GPU UUID／中央task可與P0不同，紀錄其當次分配；來源、依賴與數學profile不能因此變更。

source table保留完整已資格parent table，補入原panel資料／scaling／matrix／extra與本批全部七檔；舊panel四來源另釘4311a9d7b7356a7e59a5e42be659c52fde20d43b。執行時重核來源SHA／Git bytes、所有執行目錄clean、完整依賴payload及dev-v2 audit／dataset entries。private engine與legacy module重綁只影響私有instance，測試確認canonical matrix／extra model與gate globals未被改寫。單一卡中央owner/task保護與獨立新輸出根目錄皆存在。

資料契約沿用已凍結dev-v2全庫20,000、正式conjunction dev至少1,000 queries、negation僅診斷；COLA GQA／PACO只准固定baseline test，不供方法選擇。主K100、50／200補充，全庫排名後ignore僅評分移除不補K。Panel明列FP32 CPU dot與gallery原序stable ties，和Google原生quicksort／paired GPU scalar table公式分開披露，不能將panel分數宣稱為原論文表重現。run.sh只smoke／index四資料集，沒有自動analyze。正式讀效果仍須另凍結analysis PREREG／reading log；本次沒有授權讀取。

P3明列不在這批範圍。既有P3 validator期待另一qualification schema，這份新facade描述不能直接冒充既有P3 entry；source／README／PREREG均沒有假稱已接上或已取得P3 PASS。

獨立重跑12個新契約測試與9個原CLIP契約測試，共21 tests PASS，2.298秒：

```bash
CUDA_VISIBLE_DEVICES= PYTHONDONTWRITEBYTECODE=1 HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 \
TMPDIR=/ssd4/cactus8603/icml/hf_home/tmp OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1 \
PYTHONPATH=dl/retrieval-matrix-qwen2-runtime-fix/deps:dl/retrieval-matrix-extra/deps:dl/retrieval-matrix/deps/qwen2-transformers451:dl/retrieval-matrix/deps/qwen2-flash:dl/retrieval-matrix/deps/core-p0:. \
dl/retrieval3090/envs/retrieval3090/bin/python -B -m unittest \
codex.retrieval_matrix_clip_panel_next.test_contracts \
codex.retrieval_matrix_clip_panel.test_contracts -v
```

另加未修改實作的四模型合成負向探針：分別呼叫真正allowlisted facade，以明標synthetic scope的暫存stub執行builder，四者皆在data_audit.verify與legacy_builder被呼叫之前拒絕，確認沒有真資格不會讀panel。CUDA initialized=False。另AST parse及bash -n通過。CPU測試的成功路由使用明確臨時stub，不能當成真正P0 admit成功或GPU forward驗證。

| 審查檔案 | SHA256 |
|---|---|
| contracts.py | a7cc61cc57ca51101657224e270331db4dcb240b9901d52bfbca13986c41736d |
| runner.py | e9a40114220ad87a54778367ca55eb696033bec2b58967bbe15e0b3c0f03361b |
| build_plan.py | a768ff8eb0633b33161499bc87649cabe479750be3b5ee0ccda3ecc059526cef |
| test_contracts.py | edb5788921b4c1bc665f9bdfb98f5fb95af48b546f304011297a84cedcfafef0 |
| run.sh | 8f9689cb456840769506fe2d3aeceef82d8baf387d09b38be7ee7335556dc268 |
| PREREG.md | 738ff445647d7c648562ed081ffe0a2f0db3bc513569f4a3dc031a7483fed08c |
| README.md | de42b53c411bab8ce2337f03f9759b7f20b509b44425c9eee6be51f7295e2c12 |

已停止審查寫入；原實作未修改。後續仍需真native P0獨立數值資格，再產P2候選並另freeze。任何來源變更須使此source審查失效並重核。
