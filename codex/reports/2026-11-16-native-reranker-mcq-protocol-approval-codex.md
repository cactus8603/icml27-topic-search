# [codex] Native reranker MCQ source/protocol 核准

Codex root，2026-11-16 22:51（台北）。已讀非實作者 p3_contract_review 的最終 prelaunch report，另逐讀 contracts/adapters/runner/qualification/build/launchers/PREREG、固定 protocols 及原作者 loader/COLA/SC++/NegBench。root獨立重跑14個 CPU tests 全PASS（12.352秒）；只有合成 inputs／假 backend，未 GPU、未讀真效果。

已核固定 CORE paper v1 Table8原文：Qwen Swift2B targets COLA .686／SCPP .818／Neg .315；Swift8B .767／.856／.261；Jina-m0 .714／.842／.417。這些只是已發表數字。原released loader Qwen為 FP16/FlashAttention2/context1500/image1200/engine32；Jina原compute_score context2048/doc_type=image，COLA按原image-choice且2+2呼叫。source對應與完整固定 checkpoint／processor payload、CPU actual modules／template receipt、全部15912題／58891pairs／9425圖收據均保持。兩個方向不能挑較高者，五SCPP及兩Neg子集保留完整宏平均；同分全錯。

原始Python scalar／content／兩TrueFalse branch logprob皆留immutable raw；FP64只作無損容器。Swift storage32題維持32pairs邊界，Jina保留2+2/3/4原分組。raw先保存、checkpoint後提交，恢復不重算已存分數；qualify重驗原raw／array／state fingerprint和完整batch SHA、來源freeze、actualNumPy與非產生者重算。呼叫失敗或硬殺成本不當零。

核准 qwen2、qwen8、jina_m0 三個variant進入之後的凍結中央 GPU smoke；只把新 approved plan這三者model.verified_protocol設true並更新其完整model digest，plan設FROZEN/reviewed=true。原review draft及其source/probe均不覆寫。Qwen native2/8及通用MCMR TrueFalse三者仍verified=false且TARGETS_INCOMPLETE；不能借其他協定targets，也未授權本P0入口跑這三者。歷史paper exact revision／Swift build未公開的限制已明示，數值重現仍需獨立審查。

GPU記憶體尚未實測，Swift engine32不可逕套現行CORE request4的8GiB門檻；日後排程需保留原batch且選足夠餘量，smoke失敗保存，不fallback。此核准不是實測P0 PASS，不取消或更改任何現行作業。正式效果分析另經凍結analysis PREREG及非產生者重算。

Approved plan：`codex/retrieval_matrix_rerank_native_p0/plan.native-mcq.approved.v1.json` SHA256 `1af9cb84d317cff075a564cf7ad021bc9d872854d6caccc36227c8f37fe7b326`。原review plan SHA256 `40ad9abd8c0a7a19d2ac018a38c6bf4187dcfa69ee182c03c8c8b1dcd9a84c29`。CPU preflight SHA256 `28f24651b08112d68ff3b7312eacc91cccb0a4448dfadcdb7111659f53ef8e0d`。

審查來源SHA256：

- PREREG.md: `a53016257ad2fc1673aa1a965cd15f508532a068883848bf6d679beb1cc933eb`。
- README.md: `0028297bfda9e15d1afc9cb423f6c5769a36f5b2d94b2d0b644e105b18fed49c`。
- SOURCE_AUDIT.md: `0e38680c3d11e8a2d0ac7535355025ca997ea6bf2f8042e1f83505e91909222c`。
- adapters.py: `f00b20d2205e712bf53c478dd38292d7b63e262303f426883459ccdbaa64defd`。
- build_plan.py: `8f9e850811afb27081d327c53455d71cce208a8d2a8a8f23fa05be4c782b07ae`。
- contracts.py: `4fec5b30ef3c90999a38c13d325bcfd94811c5e5f1f30de9c6d66df19639042c`。
- cpu_probe.py: `b4ce1659af64632ba7f2db21bb655b5efbe69b6dd7fe36c7fc2a83cf893d4621`。
- qualification.py: `522ce979ed6f924c1087739e681f59862713742906467882ffea2510547b5ca6`。
- run.sh: `dc5119267822f2cfd13a9948aa6cd495fd8d23addabdbbb37e7e018a730a24a3`。
- runner.py: `fc6ff34fbd37642c750b7a7e1c15d0a287c85357bff131a34694fb83c214a0fc`。
- test_native.py: `f48d088ff1a6f6d98fcfb94c73dcf593c9d9bc2a41d3c0f93586ece3f2c26e4e`。
