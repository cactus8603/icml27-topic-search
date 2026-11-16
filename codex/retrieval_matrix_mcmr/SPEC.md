# [codex] MCMR 原生 adapter v1：固定 MMEB-V3 官方 renderer

[codex] 2026-11-16（台北）。資料、CPU 稽核與獨立 runner 已備妥；未執行或比較模型效果，GPU 由 root 審查與封存後送出。

原始資料固定 `Lux1997/MCMR@12d53f4063804ee6a976024c7375ab2fbe4368c8`：3,997 queries、104,981 candidates、103,176 互異候選圖片。全量影像沿用已獨立驗收的 `mcmr-full/images`，不改 pilot。另完整下載 MMEB-V3 固定 revision 的 MCMR tar（4,576,936,692 bytes），驗官方 LFS SHA `03eefcd923b88897da4efd1d25b759af53bd581109eb161505031070c786ea9c`；archive 內 query／candidates 與 Lux 原檔 SHA 逐 byte 相同，證據為 `mmeb-mcmr/full-archive-identity.json`。不重複解壓圖片。

唯一事前選定的文字 renderer 為 `TIGER-AI-Lab/VLM2Vec@f658ebeb49704784330ffe9e68a10796698d75e3` 的 `src/data/dataset/mcmr_dataset.py`。直接呼叫已驗 SHA 的官方函式：

- query：`Find an image-text pair that matches the given query` 加一個空白及原始 query。
- candidate：依序 `title`、`description`、`features`、`price`、`Date First Available`；list 以空白合併，非空欄位為 `field: value`，以換行合併。renderer 不按字元裁切。
- 空文字／ID、未對到圖片或 qrel 不在完整 corpus 內時停止，不按模型效果排除。

輸出 `queries.jsonl`、`corpus.jsonl`、`instances.jsonl` 採 CORE task 可讀的 schema，但 manifest 必須保留 `CORE exact conversion unverified`。**CORE 發布版本缺少它引用的 make_candidates.py／make_query_and_instances.py；因此這是 MMEB-V3 官方原生文字、CORE 相容 schema，不是已證明逐 byte 相同的 CORE 論文轉檔。** 不以三種 renderer 掃模型效果追表。

原 MCMR 的 GME retrieval renderer 及 Qwen pointwise renderer 只作來源比較，從固定官方函式 AST 讀取／CPU 比較字串，不輸入模型。前者以 pipe 組合後補 `price/date`、截 2,048 字元；後者價格／日期前另加 pipe，也截 2,048 字元。二者均不取代本輪固定 MMEB renderer。

CPU token 稽核使用六個釘版模型本機 tokenizer，逐筆保留全部原文，`truncation=False`，納入實際官方 chat format／EOD 與影像 token 保守上界。裸文字最高 2,769 tokens；CORE／Qwen3 的融合保守上界 4,072、VLM2Vec **4,463**，GME 按 payload 的更大影像上界為 19,175，均低於各模型 context。runtime 影像上界必須由凍結 processor 設定計算：Qwen2 的 `max_pixels=1,310,720`、patch14×merge2，故 `ceil(1310720/28²)=1,672`，runtime 融合上界為 `2792+1672-1=4463`；GME 的保守欄仍取 payload／runtime 二者較大值。不裁文字、不按效果裁題。正式 GPU 前，root 必須另審查並 commit runner／batch plan；固定 CORE 官方 `fused_emb`、`max_length=4096`、`truncation=False`。

[codex] 2026-11-16 上界更正：舊版把 Qwen2 generic runtime 寫成固定 1,280 image tokens，低估了 VLM2Vec 的融合上界（舊 4,071 → 新 4,463），也把 GME runtime 上界寫低；舊 GME payload 保守上界 19,175 本身未變。獨立兩環境實際 processor 的純合成 CPU probe 確認 Qwen3 輸入 4,106、GME／VLM2Vec 輸入 4,445 tokens，皆未被 `max_length=4096, truncation=False` 裁切；這些是合成輸入長度，不冒充實際 corpus 最大長度。收據為 `independent-processor-qwen3.json`／`independent-processor-qwen2.json`，不載權重、不跑 GPU。原完整 token 稽核另存 `tokenlength-audit.original-v1.json`，新 `tokenlength-audit.json` 的 `image_bound_correction` 保留前版 SHA、舊／新上界與 probe SHA；只重算影像上界，全部逐 corpus／query tokenizer 統計及 chat 格式原樣保留，未重 tokenize。後續合成 GPU smoke 仍須驗完整 context 融合推論。

所有 corpus admission 只依固定原始 candidate ID、圖片存在性與 metadata，不使用 query、qrel 或 label 選取／裁切候選。圖片與 metadata 都必須進 fused encoder。VLM2Vec-V2.0 必須載入完整 adapter；CORE 官方 `fused_emb` 的 generic Qwen2 分支漏列 `vlm2vec` 名稱，如沿同一 generic GME 分支路由，必須在獨立 plan 揭露這個 dispatch 修正，不能以裸 base 代替。
