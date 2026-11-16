# [codex] 非 CORE 重排器的官方 MCQ 協定與真實缺口

[codex] 2026-11-16（台北）。這是來源核對及尚未執行的重現規格，沒有讀取本機模型效果。六個 variant 刻意分開：同一個權重名稱不代表同一種推論／計分協定。固定檔案、完整權重 SHA 與來源入口見 `protocols.json`、`backend_specs.draft.json`。

| variant | 實際來源／評分 | 可驗 MCQ targets | 狀態 |
|---|---|---|---|
| qwen2 / qwen8 | CORE release 的 Swift FP16、image1200、context1500、engine batch32、yes/no activated FP32 scalar | CORE Table8 的 COLA text-choice、SCPP 五子集宏平均、NegBench COCO/VOC 等權宏平均 | 可準備正式重現，仍須獨立協定審查及實測資格 |
| qwen_native2 / qwen_native8 | Qwen 作者 `Qwen3VLReranker.process`，BF16、image1800、context10240、先合併 yes/no weight 再 native sigmoid | 未驗明與此完整 native 協定相符的兩個 MCQ targets | TARGETS_INCOMPLETE；不能借 Swift 表數通過 P0 |
| jina_m0 | Jina 原生 `compute_score(max_length=2048,doc_type=image)`，BF16 auto、原 MLP／score token100／sigmoid(logit−2.65) | CORE Table8，COLA **image-choice**；SCPP／NegBench 各自原規則 | 可準備正式重現，三 benchmark 保留，不能換 COLA 軸 |
| qwen_instruct8 | MCMR 作者 unchanged `LabelScorer.score_pair`，True／False 分支各一次 forward，原 sequence logprob 與 FP32 branch softmax | 只找到 MCMR 一個 fused top50 NDCG benchmark；沒有這個 image-only MCQ 適配的兩個 targets | TARGETS_INCOMPLETE；不以 MCMR 多個 cutoff 冒充多 benchmark |

CORE [論文 v1 Table8](https://arxiv.org/html/2609.04083v1) 在固定 cache 的 SHA 為 `d2492de7d8894bf1b3ff55ae0cbfe191ce02f887cf4354487af9dce477f6c869`。以原模型列為準：Qwen2 的 COLA／SCPP／NegBench 為 .686／.818／.315；Qwen8 為 .767／.856／.261；Jina-m0 為 .714／.842／.417。它們是已發表 targets，並非本機重現結果。NegBench 必須另外保留兩個原子集，不能只用 COCO 驗宏平均。

## 為何 Qwen native 不能直接掛 Table8

CORE 原始 [loader](https://github.com/Alibaba-NLP/core-emb/blob/400cf919838ceeecf4c94e3ac178a4270abcc4c5/models/load_models.py) 對 qwen3vl reranker 實際回傳 `TransformersEngine(task_type=generative_reranker,torch_dtype=float16,attn_impl=flash_attention_2,max_batch_size=32,max_length=1500)`；`main.py` 固定 IMAGE_MAX_TOKEN_NUM1200。[COLA](https://github.com/Alibaba-NLP/core-emb/blob/400cf919838ceeecf4c94e3ac178a4270abcc4c5/tasks/cola.py)、[SCPP](https://github.com/Alibaba-NLP/core-emb/blob/400cf919838ceeecf4c94e3ac178a4270abcc4c5/tasks/scpp.py)、[NegBench](https://github.com/Alibaba-NLP/core-emb/blob/400cf919838ceeecf4c94e3ac178a4270abcc4c5/tasks/negbench.py) 的 native `.process` 分支已註解或棄用；實際走 `eval_ours*` 的 Swift request。

Qwen 官方 [native source](https://github.com/QwenLM/Qwen3-VL-Embedding/blob/393e2978d27852b0d0230d6994f37f9c15bed73c/src/models/qwen3_vl_reranker.py) 則使用 `<Instruct>/<Query>/<Document>` 模板、context10240、image1800，`weight_yes-weight_no` 先在權重精度合併後投影，再在 native dtype sigmoid。這與 Swift FP16 logits 分別投影後 FP32 sigmoid 不同。兩個 API 都可忠實呼叫，但不能在核對兩個 matching targets 前宣稱 native MCQ 資格。

Qwen2 pin `4bd860ac4f15ad1897a214615cccc700f8f71818`；Qwen8 pin `b212dc8c91a8164aef1ea2de9c1a867611e75c04`。CORE 論文沒有公佈歷史權重 revision／精確 Swift build；這裡固定官方 released checkpoints 和已審穩定 ms-swift4.0.0，揭露歷史 revision 不可證的限制。數值重現仍須真正執行與獨立審查，不能用來源一致代替實測。

## Jina 的方向與原呼叫邊界

官方 [model card](https://huggingface.co/jinaai/jina-reranker-m0/blob/94bfe0aeb2d4dd7978362699cddd5893d4e0adc8/README.md) 及 [source](https://huggingface.co/jinaai/jina-reranker-m0/blob/94bfe0aeb2d4dd7978362699cddd5893d4e0adc8/modeling.py) 固定在同一 revision。此 variant 是 2.4B Jina-m0，非 Jina-v3、API 服務或 GGUF。原 config 的 `_name_or_path` 保留歷史字串 v3，不能據此把公開 checkpoint 改名為另一模型；實際 AutoModel class 是 `modeling.JinaVLForRanking`，config 與全部 payload hashes 固定。

pair 順序始終為 `(image0,text0),(image0,text1),(image1,text0),(image1,text1)`。CORE/Qwen COLA text-choice 要求 s0>s1 且 s3>s2；Jina released image-choice 要求 s0>s2 且 s3>s1，皆為 strict。程式同時產另一軸的明確 diagnostic 名稱，但資格只認原 variant 的一軸。Jina 原呼叫為 COLA 每題2+2、SCPP 每題3、NegBench 每題4；維持 method 內部 default batch8 及原 padding，不統一攤平成 batch4。SCPP 兩正 caption 都須嚴格勝過負 caption；Neg caption0 嚴格勝另外三個。平手失敗。

## 通用 Qwen True／False 的邊界

MCMR 官方 [paper v1](https://arxiv.org/html/2603.01082v1) Table4 僅對一個 LLaVE top50 fused candidate pool 報 NDCG 多個 cutoff，這不足以滿足兩個不同 MCQ benchmark。其 [原始 pointwise code](https://github.com/EIT-NLP/MCMR/blob/8bf618a89cb41c1e12f1bad640eb12bcf0ef31a0/eval/rerank/qwen3_pointwise.py) 的 system 是 e-commerce judge，user 包含原 Query／Candidate，token 是大寫 True／False；每分支重新 forward，取原 log_softmax 的 label sequence logprob，最後對兩分支 FP32 softmax。不是 yes／no 單次生成，也不是可直接移植的 CORE Table8 權重。

Qwen3-VL-8B-Instruct pin `0c351dd01ed87e9c1b53cbc748cba10e6187ff3b`；MCMR code pin `8bf618a89cb41c1e12f1bad640eb12bcf0ef31a0`。MCQ 沒有商品 metadata，Candidate 明確保持空字串，原 e-commerce prompt 和兩次 forward 保留；這是一個已標明的 image-only 適配。不能把商品 fused 評分宣稱已在 COLA／SCPP／NegBench exact 重現。

## 資料與執行限制

沿用已凍結 CORE MCQ 原始配對：15,912題、58,891pairs、9,425圖；COLA210、SCPP788/1652/1406/666/245、NegCOCO5914/VOC5031。VOC 重複題保留，不把4951 unique images誤當5031題的分母。所有來源 metadata、query、image／ID／順序與 SHA 保留。

storage 每32題提交，Swift 的32題相當於96或128pairs，均整除 engine32，不改原有效 batch 邊界；Jina／native／pointwise 仍各依原規格分組。raw transaction 先保存、array/state 後提交，中斷可從原 raw 恢復，不重跑已保存模型分數。任何 image 缺失、非有限值、raw shape 改變、Qwen vision 被 NULL 替代或 GPU offload 都 fail；不複製原 silent skip／parse-error零分。

本次只準備可審 plan/code、官方 CPU processor/template probe 與合成驗收。所有 `verified_protocol` 初始為 false，plan 為 REVIEW_REQUIRED。只有獨立審查、完整 sourcefreeze、真正中央 GPU 指派和至少兩個 verified targets 的 variant 才可進 P0 GPU smoke；數值匹配還須非產生者獨立重算全部 arrays 才能形成資格。
