# [codex] MCMR native v2 啟動前獨立審查

[codex] 2026-11-16（台北）。official_p0 非本模組實作者，只讀審查 `codex/retrieval_matrix_mcmr/` 與兩份 v2 plan，沒有修改其實作、讀模型效果或執行 GPU。結論：來源／計分與修正版 CPU 技術驗收通過，可由 root 凍結後排程；這不代表 GPU smoke 已通過，也不授予 strict CORE P0 資格。

全104,981商品已逐筆和釘版 MMEB-V3 官方 `build_mcmr_candidate_text`／圖片 resolver 比對，完整 id、path、文字、modality 與列序相同；3997 query 及 qrel 亦逐筆相符。103,176張互異圖中，有1421個共享圖片群、共3226個商品，1072群含不同文字，全部仍保留獨立商品列。checkpoint items 包含每商品 `{id,path,text,sha256,bytes}`；只允許重用同檔案的驗圖雜湊，不按圖片去重融合向量。完整 metadata 驗收證據為 `dl/retrieval-matrix/data/mcmr-mmeb-v3-native-v1/independent-prelaunch-review.json`；原103176張影像的獨立 byte/decode 收據沿用且釘SHA，本次未重複全量解壓或驗圖。

原生融合路線直接呼叫固定 CORE `fused_emb`，圖片及每商品自己的完整 metadata 都進 processor；`max_length=4096,truncation=False`。VLM2Vec載完整已核 DoRA，僅明示選用同一 generic Qwen2 的 gme dispatch 以補 released 名稱漏列。query及corpus均以 native BF16 normalization，再無損保存成float32；相似度恢復BF16、按原1000-query block作完整庫GPU矩陣乘法。修正版每列以一維 `torch.topk(k=100)` 呼叫，對齊 CORE `tasks/mcmr.py` 的形狀；沒有改成 float32 排序。平手 indices 本身不保證穩定，故保留相同呼叫路線及 native ordinal 診斷。[PyTorch 官方說明](https://docs.pytorch.org/docs/2.15/generated/torch.topk.html)

Recall是 relevant fraction，pytrec 在完整native top100 admission上算 Recall@1/5/10/50/100、nDCG@5/10、MAP@10；MRR@10另將native top10交 reciprocal-rank evaluator，3997題全列、零命中不丟。另報 native torch ordinal Hit@1、fractional Recall@1、MRR@10，揭露 pytrec 的分數平手/docid再排序，不挑較好數字。MMEB renderer 與 CORE 原轉檔未核相同的差異有明示；`strict_core_p0_match_eligible=false`、報告 `strict_CORE_P0_match_claim=false`，Table5只作差異對照，不能單靠此欄取得 strict P0 資格。

原 corpus 每商品checkpoint會反覆重寫累積state，104981步導致平方成長的寫入成本。v2改為storage chunk64、compute內仍依原序逐商品呼叫batch1；query batch32不變。獨立137商品測試在提交64商品後於下一chunk中途失敗，再續跑，結果與逐商品基準完全相同；每次forward確實只有一商品，未提交部分才重算，metadata改變會在forward前被拒絕。成本ledger保留失敗與resume invocation。9項模組CPU測試及獨立chunk/逐列BF16 top100合成比對全部通過，證據 `independent-prelaunch-v2.json`。

兩個正式環境的實際 dependency checks 通過：Qwen3為 torch2.14.0+cu126／transformers4.57.3（2734檔），Qwen2為 torch2.6.0+cu124／transformers4.51.3（2544檔）；模型完整spec與各自reference P0保持一致，只有事前明示的image batch1/query32差異。六個 actual processor 的CPU合成長輸入確認4096不是截斷硬上限：Qwen3為4106tokens、GME/VLM為4445tokens，未載權重或使用GPU。證據 `independent-processor-qwen3.json`／`independent-processor-qwen2.json`。

上述actual processor檢查揭露舊VLM影像token上界1280不足。修正以釘版runtime `max_pixels1310720` 與 Qwen2 factor28計 `ceil(1310720/784)=1672`；VLM完整保守上界4463，GME runtime上界4463／payload更保守19175，Qwen3仍4072，均低於context。已獨立逐欄確認全部原corpus/query tokenizer統計未變，只更正image bounds；舊audit完整另存，兩個CPU probe SHA及原source/config/tokenizer/formatter SHA皆納更正收據。新audit SHA為 `b9a179411e704930afeb447d149dd9e41327016c3a675e65250d5588205ba68e`。

保留限制：CORE未釋出其原始MCMR轉檔腳本，本批是固定官方MMEB原生文字加CORE相容schema；原論文exact-match仍未證明。正式GPU生產必須使用刷新後v2 plan與所有source SHA，先通過合成smoke，無量化、CPU offload或自動batch fallback。
