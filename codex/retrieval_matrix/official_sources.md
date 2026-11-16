# P0 官方來源與重現協定

[codex] 2026-11-16。只核對公開原始碼、論文及模型 metadata；沒有 GPU 推論、沒有開自建 test 或新主判準結果。機械設定與每個官方檔案的雜湊在 [official_protocols.json](official_protocols.json)。

## 可先執行的 CORE／Qwen3-VL 普通與否定 COCO

以 [CORE 官方 repo 固定版本](https://github.com/Alibaba-NLP/core-emb/tree/400cf919838ceeecf4c94e3ac178a4270abcc4c5) 的 release code 為依據。完整 Python／Markdown 程式已下載到 `dl/retrieval-matrix/sources/core/`，27 個檔案逐一符合 Git tree 中的 blob SHA-1，另外計算 SHA-256。程式未經修改。

從 `models.qwen3_vl.qwen3_vl_wrapper` 載入 `Qwen3VLForEmbedding` 與 `Qwen3VLForEmbeddingProcessor`；模型 bf16、eval。processor 為 `instruction_standalone=True`、`min_pixels=4096`、`max_pixels=1310720`、`total_pixels=4608000`、`num_frames=48`。`from_pretrained` 的 max_length 是 3024，但實際文字／影像呼叫傳入 `max_length=1024, truncation=False`，不可誤解為截斷 1024。影像由 qwen-vl-utils 先縮放，processor 的 `do_resize=False`。

**真正生效的 system 是 `Represent the user's input.`，文字與影像相同。** [embedding.py](https://github.com/Alibaba-NLP/core-emb/blob/400cf919838ceeecf4c94e3ac178a4270abcc4c5/embedding.py) 雖定義 caption／image instruction，卻對兩種輸入皆傳 `is_query=False`；[wrapper](https://github.com/Alibaba-NLP/core-emb/blob/400cf919838ceeecf4c94e3ac178a4270abcc4c5/models/qwen3_vl/qwen3_vl_wrapper.py) 因此不採用傳入 instruction。這與獨立 Qwen 官方 `.process()` 範例的預設模板不同，P0 不可直接替換。文字內容是原 caption，影像是 `PIL.Image.open(path)`；模板 `add_generation_prompt=True`，tokenizer postprocessor 加 `<|endoftext|>`。取最後一個有效 token 的 hidden state，`F.normalize(...,dim=-1)`；不截 MRL 維度。

普通 COCO 用 NegBench 的 `COCO_val_retrieval.csv`；否定用 `COCO_val_negated_retrieval_llama3.1_rephrased_affneg_true.csv`，皆含 `filepath,image_id,captions`，其中 captions 為 Python literal list。影像為 COCO val2017。採完整 CSV 的全部影像與 caption，不用 pilot。官方 main 的普通 COCO batch 是 64；NegBench 的 `eval()` 沒有傳遞 batch 參數給子任務，所以實際是 32。3090 若為記憶體調小 batch，須事先固定並在收據揭露。

已在 CPU 用未改官方 processor 實測空 caption：`transformers==4.57.3`、`qwen-vl-utils==0.0.14` 可正常產生 21 個 token，故原始空句仍納入官方分母。實際結尾有兩個 `<|endoftext|>`，是 checkpoint tokenizer 與 wrapper 各追加一次，按原碼保留。收據 `dl/retrieval-matrix/sources/core-empty-caption-cpu.json` 記錄完整 IDs、decoded prompt 與實際 import／distribution 版本，不載模型權重。原 torch214 環境的 `transformers==5.20.2` 在 processor 初始化因缺 `image_processor.min_pixels` 失敗；隔離安裝 4.57.3 後通過，不修改既有環境。

普通 COCO／Flickr 的 T→I 是 pytrec `recall_5/10`，每個 caption 對一個 image。**I→T 是全部參考 caption 的回收比例**，並非標準 COCO 常見的「前 K 有任一句正確 caption」成功率。否定檢索為所有 caption 中，原 image 是否出現在 `torch.topk(...,5)` 的平均。Table 5 的欄方向已核對，不應因 I→T 數值偏低而交換欄名。

## 論文與 release 的差異

[CORE 論文 v1](https://arxiv.org/html/2609.04083v1) 的 Appendix B.2 寫 image tokens 1000、text length 200；release 卻使用以上 pixels 與 `truncation=False`。本次先凍結照 release，不看結果後再選模板或解析度。Table 2／5 的六個 embedding 目標數字已逐項核對並寫入 JSON。

[Table 5](https://arxiv.org/html/2609.04083v1#S5.T5) 的 MCMR 指標是 **MRR@10**。release `tasks/mcmr.py` 只算 pytrec recall、NDCG、MAP，沒有 MRR；不可把無截斷 MRR 當成同一目標。

## GME 與 VLM2Vec 的必要關卡

GME 固定 [Alibaba-NLP/gme-Qwen2-VL-2B-Instruct](https://huggingface.co/Alibaba-NLP/gme-Qwen2-VL-2B-Instruct/tree/9cfa6413f704a7c1cf5064d240748e10c876b286)，3 個 FP32 safetensors 共 8.229 GiB。CORE release 使用自有 Qwen2 wrapper，與 GME model card 的 remote-code API 不同；model card 也明示 remote code 對 `transformers>=4.52` 有相容問題。不能在 Qwen3 環境中默默換成另一套 adapter 後宣稱是相同的 CORE 協定。

VLM2Vec-V2.0 的正確公開來源是 [VLM2Vec/VLM2Vec-V2.0](https://huggingface.co/VLM2Vec/VLM2Vec-V2.0/tree/e39ff079b8275ef876d3656da8c0bddbff3c4dde)，**不是** `TIGER-Lab/VLM2Vec-Qwen2VL-2B` 的 V1。V2 只有約 35.25 MiB adapter，adapter_config 明定 `use_dora=true,r=16`、基底 `Qwen/Qwen2-VL-2B-Instruct`。另外固定基底 revision `895c3a49bc3fa70a340399125c650a463535e71c`，其 safetensors 4.115 GiB。CORE generic loader 不顯式套 adapter，`fused_emb()` 的 Qwen2 分支甚至沒有包含 `vlm2vec` 名稱，不能直接以未驗證的 generic loader 開跑。

[VLM2Vec 官方 source](https://github.com/TIGER-AI-Lab/VLM2Vec/tree/f658ebeb49704784330ffe9e68a10796698d75e3) 的 source／範例也已固定於 `dl/retrieval-matrix/sources/vlm2vec/`，可供核對 adapter 載入、key coverage 與 pooling。先確認這些條件，才可把 V2 納入 P0；執行兼容修正應另列為明確差異。

此外，CORE `models/load_models.py` 只在 transformers >4.47 且 <4.57 分支 import Qwen2 wrapper；Qwen3 分支需 >=4.57。混用單一未核對環境會出現未定義名稱／舊內部 API，不是「只要改模型路徑」即可。

## MCMR exact conversion 尚缺官方證據

官方 `tasks/mcmr.py` 讀 `benchmark/MCMR-datasets/uni_mteb_eval_t2ti/`：

- `queries.jsonl`: `id,text`。
- `instances.jsonl`: `qid,pos`，pos 是 corpus id list。
- `corpus.jsonl`: `id,text,image`，image 可直接傳 PIL 載入。

商品以圖＋文字 jointly encode，`max_length=4096,truncation=False,is_query=False`。官方 `datasets.md` 提到 `make_candidates.py`、`make_query_and_instances.py`，但 35 項 Git tree 沒有這兩支檔案。因此 **商品文字欄位順序／標籤／空值格式尚無 CORE 官方可驗證實作**。本機既有 pilot renderer 不能因為合理就宣稱完全相同。下載／解壓公開資料可以先做，重現聲稱仍須揭露這項缺口。

## CORE reranker 的 MCQ 驗收

CORE release 的 reranker 走 ms-swift `TransformersEngine(task_type='generative_reranker')`，fp16、FlashAttention 2、max_batch_size 32、max_length 1500，main 設 `IMAGE_MAX_TOKEN_NUM=1200`。它沒有走 repo 內附的 Qwen `.process()` scoring 路徑。request 為 system instruction、user caption、assistant `<image>`，另附 image path；temperature 0，max_tokens 1500。score 取 `float(response.choices[0].message.content[0])`。COLA system 末尾無空白，SugarCrepe++／NegBench 有一個空白；JSON 已逐項保留。

NegBench 每張圖四個 caption，caption_0 正確，必須嚴格高於其他三個分數；平手失敗。**Table 1 的 MCQ 是 VOC 與 COCO 的 unweighted mean**，應使用 Table 8 的分項驗收：

| 模型 | VOC MCQ | COCO MCQ | 宏平均 |
|---|---:|---:|---:|
| CORE-Reranker-2B | 0.784 | 0.507 | 0.646 |
| CORE-Reranker-8B | 0.835 | 0.562 | 0.698 |

因此只跑 COCO 不得對 0.646／0.698。SugarCrepe++ 要兩個正 caption 都嚴格高於負 caption，再對五子集平均。COLA 要兩張圖各自把正 caption 排高；不是四張候選圖的一般檢索。

## 下載與驗證

[acquisition.py](acquisition.py) 只下載固定 revision 的公開推論所需檔案，不取 optimizer／訓練 state，沿用已存在的斷點續傳與 LFS SHA-256／Git blob SHA-1 驗證程式。Qwen3-VL Emb2／8、GME2、VLM2Vec V2 adapter＋base 共 67 檔、31.578 GiB（含 tokenizer／設定）。manifest 與逐檔收據在 `dl/retrieval-matrix/acquisition/`。CORE2／8 既有 .bin／safetensors 快取沿用，不重複下載；模型是否通過 P0 仍由實際完整資料推論與獨立重算決定。
