# [codex] 官方重排器與 MCQ 適配

這個模組提供 P0 選擇題資料、嚴格評分規則，以及可供父 runner 整合的官方 backend。尚未執行 GPU 或產生研究效果；不改已凍結的 `codex/retrieval_matrix/`。

`protocols.json` 固定來源、完整 Table 8 目標與差異。`backends.py` 只呼叫官方 engine／scorer；`mcq_data.py` 產生完整 pair 清冊；`metrics.py` 保留原題及嚴格不等號，僅在同一 benchmark 內平均。CORE／Qwen 的 COLA 是固定圖選 caption，Jina release 是固定 caption 選圖，兩種規則分名，不能混稱完全相同的統一指標。

環境在 `dl/retrieval-matrix-rerank/env`，唯讀繼承既有 torch2.6cu124 套件及 `deps/qwen2-flash`，沒有修改既有環境。新增依賴固定 ms-swift4.0.0、transformers4.57.3、peft0.18.0、datasets3.6.0、trl0.24.0、qwen-vl-utils0.0.14。install report 與完整 freeze 在同一 runtime 根目錄；CORE checkpoint 自身 `args.json` 記錄原 swift4.0.0.dev0，而沒有公開該 development commit。

CPU 核對及資料準備：

```bash
export HF_HOME="$PWD/hf_home" TMPDIR="$PWD/hf_home/tmp"
export PIP_CACHE_DIR="$PWD/hf_home/tmp/pip"
export HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1
export IMAGE_MAX_TOKEN_NUM=1200
CUDA_VISIBLE_DEVICES='' dl/retrieval-matrix-rerank/env/bin/python -B \
  codex/retrieval_matrix_rerank/cpu_probe.py
dl/retrieval-matrix-rerank/env/bin/python -B \
  codex/retrieval_matrix_rerank/mcq_data.py
dl/retrieval-matrix-rerank/env/bin/python -B -m unittest discover \
  -s codex/retrieval_matrix_rerank -p 'test_*.py'
```

原 MCQ 資料共 15,912 例、58,891 pairs、9,425 張不重複圖片。COLA210；SugarCrepe++五子集合計4757；COCO MCQ5914列／3926圖；VOC5031列／4951圖。VOC原始 test archive4952圖已核 torchvision 官方 MD5，所有 MCQ 引用圖齊。原 CSV 路徑改以 basename 對已驗證原圖目錄；重複 VOC 題列全部保留。

Jina 的 COLA 規則另產生清冊：

```bash
dl/retrieval-matrix-rerank/env/bin/python -B \
  codex/retrieval_matrix_rerank/mcq_data.py --cola-image-choice \
  --out dl/retrieval-matrix-rerank/data/mcq-jina-prepared
```

`build_specs.py` 核模型 payload 並建立 backend 規格。預設 `verified_protocol=false`，父代理完成獨立審查後才使用 `--reviewed` 產生另一份規格。GPU 前還須依共用協作規則凍結 preregistration、判讀設定、所有執行源碼與依賴並 commit；本模組不自行送 GPU 工作。

```bash
dl/retrieval-matrix-rerank/env/bin/python -B \
  codex/retrieval_matrix_rerank/build_specs.py --models core2 core8 \
  --out codex/retrieval_matrix_rerank/backend_specs.core.draft.json
```

首批交付為 `plan.core-mcq.v1.json`、`PREREG.md`、`SOURCE_MANIFEST.md` 及 `runner.py`／`run_core.sh`。計畫初始狀態 `REVIEW_REQUIRED`，不能執行正式推論。獨立 reviewer 核對後將計畫改成 `state=FROZEN, reviewed=true`，且兩個 model 的 `verified_protocol=true`，連同全部宣告源碼 commit。啟動時必須提供完整40字元 commit，runner 逐檔對 Git 與 SHA-256，並要求本模組工作樹乾淨。

```bash
CUDA_VISIBLE_DEVICES='' dl/retrieval-matrix-rerank/env/bin/python -B \
  codex/retrieval_matrix_rerank/preflight.py
bash codex/retrieval_matrix_rerank/run_core.sh FULL_COMMIT \
  codex/retrieval_matrix_rerank/plan.core-mcq.v1.json core2 \
  dl/retrieval-matrix-rerank/runs/core2-mcq-v1
```

`preflight.py` 僅核資料／來源／環境，不代表獨立批准。launcher 先用1280×960純合成圖檢查 batch1與4，再保存原始分數；合成 smoke 失敗不覆寫，PASS 可恢復。完整研究 score 以原批次邊界和逐批雜湊續跑。分析須另明確呼叫 `runner.py --stage analyze`，只在 `main/results/` 寫出逐題命中、子集與三個 benchmark 的官方目標驗收；不做跨 benchmark 總平均。輸出限制在上述 runtime `runs/` 下。

父 runner 使用 `BACKENDS[spec['backend']](spec)`，再呼叫 `score_pairs(rows,instruction=...)`。每個 row 有固定 `id,query,image`，回傳 `id,score,official_raw`；缺分數、NaN、圖片錯誤會失敗，不暗自補零或跳題。實際 score 與 raw 分支分數應一同保存。

- `swift_released`：CORE2／8 與 Qwen Reranker2／8 的 P0 路線。使用 ms-swift `generative_reranker`、fp16、FlashAttention2、max_length1500、IMAGE_MAX_TOKEN_NUM1200，保留官方 request、checkpoint args 所選模板及 float32 sigmoid。COLA instruction 末尾無空白，SCPP／Neg 有空白。完整 VOC／COCO 的宏平均才能對 Table 8 的 NegBench 平均。
- `qwen_native`：P3 獨立 Qwen 官方 `.process()`。預設 bf16、image1800tokens、max_length10240，投影與 sigmoid 精度和 Swift 路徑不同；不把兩套輸出混為同一 P0 協定。支援官方 image＋text documents。
- `jina_native`：固定 checkpoint 的 `AutoModel` remote code，`compute_score(...,max_length=2048,doc_type='image')`。保持官方 score head、2.65 logit bias 及 sigmoid；拒絕額外產品文字，避免悄悄丟失模態。
- `mcmr_pointwise`：直接匯入 MCMR 作者釘版 `qwen3_pointwise.py`，呼叫原 `load_qwen`／`LabelScorer.score_pair`。標籤是 **True／False**，每 pair 兩個 forward，保存 p_true、true_logprob、false_logprob。產品文字用作者原 renderer，最多2048字元。COCO等只有圖的情境需標為 image-only 適配；不宣稱它是 MCMR 原始完整資料重現。沒有將它替換成較便宜的 yes/no 單 forward。

新增權重由 `acquire.py` 下載，官方 LFS SHA-256／Git blob SHA-1 核驗，保留逐檔收據、`download.log` 與 stdout。CORE checkpoint 原有權重沿用，另補原作者 `args.json`；它是 ms-swift 自動辨識模型與模板的必要檔。其他模型若尚未下載完成，builder 會拒絕產生可用規格。

資料取得使用了 [Google Drive 技能](</home/cactus8603/.codex/plugins/cache/openai-curated-remote/google-drive/0.1.23/skills/google-drive/SKILL.md>)，讀取 NegBench 作者公開檔案；沒有寄送訊息、變更共享或修改遠端檔案。
