**[codex] Extra baselines 獨立事前審查，2026-11-16 21:38（台北）**

來源與權重準備可封存；目前不能宣稱 native P0 重現完成，也不能送研究或 P2 工作。清冊確實是 `PREPARATION_ONLY`，12 個模型全部維持 `P0_REQUIRED`，published targets 全標 `reported_only`。本輪沒有改 extra 程式、沒有啟動 GPU，也沒有讀 benchmark 模型效果或主判準結果。以下阻擋針對未來執行資格，不是下載失敗。

1. **UniME-V2 的原文方向與可執行 scorer 尚未建立一致映射。** 原 [Table2](https://arxiv.org/html/2510.13515) 的 `qi→ct`／`qt→ci` 順序與 registry 的數字對應相同，沒有發現 registry 抄錯；但作者釘版 [eval 函式](/ssd4/cactus8603/icml/dl/retrieval-matrix-extra/sources/unime_v2/Embedding/evaluation/eval_flickr_coco_sharegpt4v_urban1k.py:56) 以 `texts_image_index`（每 caption 一個 image index）建立 positive matrix，卻在第83行傳入 image features 為行、text features 為列。原生 5k／25k 輸入的索引長度不相容。本次直接 AST 載入原函式，僅給合成 2 圖／10 句，重現 `IndexError: shapes [2], [10]`；未載權重或產生指標。註解的反向呼叫才有正確形狀，而其輸出表格又把 image-retrieval 名稱印成 Image→Text。因此新 P0 必須事前記錄 paper 欄位、raw code 欄位與 conventional retrieval 方向的映射，不能照印出欄名直接資格通過。

2. **UniME-V2 不能沿用目前 CORE 相似度精度與 CLIC ties 當作原生協定。** 作者 [FeatureExtractor／evaluate_recall](/ssd4/cactus8603/icml/dl/retrieval-matrix-extra/sources/unime_v2/Embedding/evaluation/eval_flickr_coco_sharegpt4v_urban1k.py:36) 是 BF16 encoder／pooling，轉 FP32 features 後再 FP32 normalize、FP32 dot，scorer 使用 `torch.topk`。目前 [extra adapter](/ssd4/cactus8603/icml/codex/retrieval_matrix_extra/adapters.py:224) 另做一次 native BF16 normalize，共用 [producer](/ssd4/cactus8603/icml/codex/retrieval_matrix/runner.py:310) 又按 loader dtype 轉回 BF16 dot；[extra scorer](/ssd4/cactus8603/icml/codex/retrieval_matrix_extra/runner.py:84) 對所有 native-anyhit 使用 CLIC 的 `numpy.argsort`。新 native P0 必須按 family 固定 normalization、score dtype 與 ties，並明示修正；不可只把 encoder 改 FP32。作者 FeatureExtractor 本身也會 `.to(device,dtype=bfloat16)`，這部分沒有另列為偏差。

3. **「至少兩個 benchmark」尚不能只計名稱數量。** CS-CLIP 的 [三個 targets](/ssd4/cactus8603/icml/codex/retrieval_matrix_extra/build_registry.py:154) 全是 Half-Truth 的三個資料子集；若門檻指兩個不同 benchmark，這不等於通過兩個 benchmark。未來 qualification policy 須先定義 benchmark identity／子集如何計數，不能靠目前 `len(distinct names)>=2` 通過。CLIC L14／CLIPS 的 COCO target 另受缺原轉檔阻擋；SugarCrepe++ 的多個 replace／swap 欄位也不能直接算成多個 benchmark。

4. **目前入口只完成 retrieval P0 接口準備。** [extra load_plan](/ssd4/cactus8603/icml/codex/retrieval_matrix_extra/runner.py:43) 對非 GME 強制 native-anyhit retrieval，不能直接執行 CS-CLIP Half-Truth accuracy 或 SugarCrepe++ paired 協定；文件已有說明須另接 paired qualification。現有 [P2 factory](/ssd4/cactus8603/icml/codex/retrieval_matrix_panel/runner.py:39) 也只分 Qwen2／CORE，尚未載入 ExtraOfficialModel。這些缺口應由新 native P0／經資格審查後的 P2 接線解決；本目錄沒有可執行的新研究 plan，不能據本審查直接送 GPU。

本次獨立驗證：

- 27 項 CPU tests 全通過：extra 9、共用 metrics 7、共用 runner 11。測試涵蓋 any-hit 不等於 caption fraction、完整 caption 分母、ties、身份分離、精度／量化拒絕及續算邊界；不是 native benchmark 結果。
- 3,232 個互異實體檔、66,615,790,075 bytes 全 SHA256 通過：dependencies 2,766、模型 payload 109、官方來源 349、support configs／tokenizers 11、原文／modelcard 6；類別間可能重疊。另獨立重核 104 個 HF payload 對固定 revision metadata 的 LFS digest／Git blob，以及 307 個官方 source Git blobs。
- 9 個實際載入 module path／version 與 dependencies 完全相同；隔離 OpenCLIP 2.31.0、transformers 4.51.3、torch 2.6.0+cu124、PEFT 0.15.2、FlashAttention 2.7.4.post1 等。完整 SHA 證據在 [independent-prelaunch-integrity.json](/ssd4/cactus8603/icml/dl/retrieval-matrix-extra/independent-prelaunch-integrity.json)。
- 原有 12 模型 CPU meta／processor 收據 `cpu-preflight.json` 是 PASS，所綁 registry SHA 及每模型 identity 逐項與現版一致；本輪沒有重跑大型模型 meta 建構。key／shape 全覆蓋，UniME2 僅省略設定明示的 tied LM-head alias。SigLIP2 完整模型約 1.136B parameters，So400m 是視覺塔家族名，沒有把完整模型誤寫成400M。
- 舊 `cpu-synthetic-forward-final-two.json` 的 adapter SHA 與現版不同，不能無聲當成現版證據。已用**現版 wrapper** 重跑原 NegCLIP／Triplet OpenCLIP finetune 的純色圖／兩句合成文字 CPU forward：兩者 PASS、GELU、影像1×512／文字2×512、向量有限且單位化。新 [收據](/ssd4/cactus8603/icml/dl/retrieval-matrix-extra/independent-cpu-forward-final-two.json) SHA `e7f642c673807c7f39cbd57e724fcb4523d965920428715915220f8ed37302cf`，綁現版 adapter SHA；無 GPU、無 benchmark 輸入。

身份與協定邊界已核對如下；完整40字元 revision、權重檔 SHA 均留在 registry，不以同骨幹推定同模型。

| 項目 | 已確認身份與不能混用的協定 |
|---|---|
| SigLIP2 | Google `so400m-patch14-384@e8e48729…`；固定384、lowercase、Gemma tokenizer長64。Google官方 HF PyTorch port 與原 BigVision/JAX/TF 差異有揭露。COCO 是 TFDS Karpathy2014 test5k，原25,010 captions；Flickr用原Karpathy文字，不能換CORE HF引號版。原圖現已補齊。 |
| CLIC B32／L14 | 分別 `7b6ca58a…`／`57a65a05…`，都是CogVLM／LAION版，沒有套RedCaps較好列。retrieval原 `pre_caption(max_words=30)` 以固定原函式AST載入；paired SugarCrepe／++ 須另固定raw文字政策。 |
| CLIC-CLIPS L14 | `75717eef…`，作者 `clips` family，BERT tokenizer長80；與標準CLIP長77及CLIC CLIP-L14分開。 |
| 原 NegCLIP | 原作者Drive file ID＋原檔 SHA、作者 code commit `81fdd20d…`、tensor-only conversion SHA；是COCO finetune。其資格 targets 明示來自CLIC作者 reevaluation，不能稱原NegCLIP paper同一 protocol。 |
| NegCLIP CC12M | `TripletCLIP/CC12M_NegCLIP_ViTB12@4afcd9ac…`，TripletCLIP作者重訓的scratch baseline，與原 NegCLIP 分開。 |
| TripletCLIP CC12M／finetune | scratch HF `1f8b0192…` 對Table13；OpenCLIP finetune Drive對Table16，作者 code `abbf81c9…`。ViTB12指12層、實際patch32；不把兩版或CLIC reevaluation的數字互換。 |
| GME7 | `e54cb53a…`，完整權重約8.291B parameters；沿CORE Qwen2 wrapper／BF16與pytrec relevant-caption fraction。沒有宣稱是GME原生prompts／conventional I2T any-hit的同一資格。 |
| UniME-V2 2B／7B | `29f0ce56…`／`2ed1953a…`，分別2.209B／8.291B完整參數；作者MMEBModel、eos pooling、high1344 resize、原process函式。原版UniME Qwen2 2B／7B exact公開身份仍unknown，registry沒有以V2、Phi或LLaVA冒充。原作者指 `royokong/coco_test`／`flickr30k_test`，COCO取每row前5句，必須釘自己的HF資料revision與原文字。方向／精度問題見前述阻擋。 |
| CS-CLIP | 重用 `afd6e90f…`，沒有重下載不同權重；Half-Truth strict正確短句分數大於foil，需paired scorer。三子集與兩benchmark門檻見前述。 |

CLIC paper 的 COCO 明寫 Val2017，但作者程式下載 `coco_karpathy_test.json` 後讀未發布 `_test_2017.json`；缺口確實存在，registry／SOURCES均有保留。canonical Val2017轉檔可以另標 adapted，不能冒充 exact。原Karpathy COCO 與 CORE val2017只有593／5000圖交集；原Flickr caption與CORE HF版也有字串差異。這些都是未來plan的資料身份要求。

Drive沒有公開immutable revision，conversion程式以固定原SHA和file ID讀取 `weights_only=True`，並檢查逐tensor相等後才移除自己下載的optimizer原檔。此輪核對了程式、收據、現存safetensors完整SHA與header／模型身份；原Drive檔已移除，**沒有重新下載原檔獨立重做逐tensor比較**，不得把取得時的相等收據寫成本輪重做。可追溯限制已在既有文件明示。

審查快照：registry SHA `cc408a9cca74a16f7f377ea88a50e3e5313586ba034bcfd958e085709cefb902`；dependencies SHA `12d6d1b761179329243fc55734591efdbeea87cf77947472bb0f7dca95418f5a`；adapter SHA `02a57a608ffebeb2377b9aaa9d45d767b49f99c68b116973701c0531448026d5`；extra runner SHA `19cf2b5f3aa7c1a541695cb5d30c09f809a7581c99475548ddd496f4574cabbe`。如後續改動這些bytes，需要重核相應部分；本報告不替尚未凍結的native P0／P2授權。
