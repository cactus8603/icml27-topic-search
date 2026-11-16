# [codex] 官方來源與 baseline 身分

`registry.json` 綁定 checkpoint revision、所有推論 payload SHA、官方程式 commit/SHA、published target 的原文檔 SHA，以及原生 protocol。它是準備清冊，所有項目都尚待 P0 資格檢查；不代表已完成論文重現。

| 模型 | 作者來源 | 資格依據與身分限制 |
|---|---|---|
| SigLIP2 So400m/14@384 | [Google checkpoint](https://huggingface.co/google/siglip2-so400m-patch14-384)、[BigVision](https://github.com/google-research/big_vision)、[原文](https://arxiv.org/abs/2502.14786) | 原文 Table1 COCO/Flickr；[作者重現說明](https://github.com/google-research/big_vision/issues/166) 指定 lowercase/長64。使用 Google 官方 HF 轉換，與 JAX/TF 前處理差異揭露。 |
| CLIC ViT-B/32 CogVLM | [作者 code](https://github.com/AmitPeleg/CLIC)、[checkpoint](https://huggingface.co/nmndeep/CLIC-ViT-B-32-224-CogVLM)、[原文](https://arxiv.org/abs/2505.24424) | Table1/2 SugarCrepe++、COCO、Flickr。是 CLIC-LAION，不是 CLIC-RedCaps。 |
| CLIC ViT-L/14 CogVLM | [checkpoint](https://huggingface.co/nmndeep/CLIC-ViT-L-14-224-CogVLM) | Table3 CLIP 組 CLIC-LAION：COCO、SugarCrepe++。 |
| CLIC-CLIPS ViT-L/14 CogVLM | [checkpoint](https://huggingface.co/nmndeep/CLIC-CLIPS-ViT-L-14-224-CogVLM) | Table3 CLIPS 組 CLIC-LAION，與標準 CLIP L14 分開；使用原作者 `clips` 與 BERT tokenizer。 |
| 原版 NegCLIP | [作者 code](https://github.com/mertyg/vision-language-models-are-bows)、[作者 Drive](https://drive.google.com/file/d/1ooVVPxB-tvptgmHlIMMFGV3Cg-IrhbRZ/view)、[原文](https://openreview.net/forum?id=KRLUvxh8uaX) | 原版 COCO finetune。暫存資格 targets 用 CLIC Table2 的公開 reevaluation；不能宣稱它是原 NegCLIP paper 的同一套下游 protocol。 |
| TripletCLIP CC12M | [作者 code](https://github.com/tripletclip/TripletCLIP)、[checkpoint](https://huggingface.co/TripletCLIP/CC12M_TripletCLIP_ViTB12)、[原文](https://tripletclip.github.io/static/docs/TripletCLIP__Camera_Ready_.pdf) | 原文 Table13 CC12M 列，COCO/Flickr。從頭預訓練，不是 OpenCLIP finetune。 |
| NegCLIP CC12M | [checkpoint](https://huggingface.co/TripletCLIP/CC12M_NegCLIP_ViTB12) | TripletCLIP 作者重訓的 baseline；原文 Table13 CC12M NegCLIP 列，與原版 COCO finetune 不同。 |
| TripletCLIP OpenCLIP finetune | [作者 Drive](https://drive.google.com/file/d/14mupW26LMh6U4FQPa74FOIMEg8MndxCh/view) | 原文 Table16 COCO/Flickr。不能套 Table13 或 CLIC paper 的 TripletCLIP 列。 |
| GME7 | [checkpoint](https://huggingface.co/Alibaba-NLP/gme-Qwen2-VL-7B-Instruct)、[GME 原文](https://arxiv.org/abs/2412.16855)、[CORE source](https://github.com/Alibaba-NLP/core-emb) | CORE Table5 COCO/Flickr，沿用 CORE 未修改 Qwen2 embedding 路徑；不是 GME 原作者原生 prompts 的資格宣稱。 |
| UniME-V2 Qwen2-VL 2B/7B | [作者 code](https://github.com/GaryGuTC/UniME-v2)、[2B checkpoint](https://huggingface.co/TianchengGu/UniME-V2-Qwen2VL-2B)、[7B checkpoint](https://huggingface.co/TianchengGu/UniME-V2-Qwen2VL-7B)、[原文](https://arxiv.org/abs/2510.13515) | 用 V2 原文 Table2 COCO/Flickr。與 CORE 引用的原版 UniME Qwen2 2B/7B 身分尚未建立，禁止直接套 CORE UniME 數字。 |
| CS-CLIP | [既有固定 checkpoint](https://huggingface.co/kbora/CS-CLIP/tree/afd6e90f39fe49137fe15d6a462d994eae4811f5)、[作者 code](https://github.com/kargibora/CS-CLIP) | 重用已下載權重；modelcard 三個 Half-Truth sets 作原生資格 targets。仍需自己 protocol/metric freeze。 |

資料與協定缺口：

- SigLIP2 原生 COCO 是 [TFDS coco_captions/2014 的 Karpathy test5k](https://www.tensorflow.org/datasets/catalog/coco_captions)，不是 CORE val2017。原始 Karpathy archive 已在共用 data cache；缺影像由 root 排程補齊。Flickr 的原 Karpathy caption 與 CORE HF CSV 也不同。
- TFDS builder 直接讀 `caption_datasets.zip/dataset_coco.json`，完整保留 `sentences[].raw`；Karpathy test5k 有10張含6句，因此 SigLIP2 原生 COCO 分母是25,010句。UniME-V2 則在 `data/eval_data_download.sh` 明確指定 `royokong/coco_test`、`royokong/flickr30k_test`，COCO eval 再取每 row `text[:5]`；它必須依作者 HF 原文字與順序，不可假設和 Karpathy raw 或其他 HF CSV 文字相同。
- CLIC 的原文明寫 Val2017，但官方 `COCO_Retrieval` 下載 Karpathy JSON、讀取未發布的 `_test_2017` 轉換檔。這個缺口不以自行轉換冒充 exact；可優先凍結其他官方 benchmark，或把明示 adapted 的 Val2017 驗證另列。
- `unime-public-search.json` 記錄公開 HF 搜尋。找到的作者 Qwen2 checkpoints 都是 V2；原版 UniME 的官方公開首頁主要提供 Phi3.5/LLaVA，不能以骨幹相同推定 checkpoint 相同。這是來源身分未解，不是執行相容性失敗。
- Drive 沒有公開 immutable revision；完整原 SHA、固定作者 code commit、file ID、轉檔 SHA 與逐 tensor 相等證明都保留在 `weights/conversion-receipt.json`。轉檔只保留推論 tensors。

`runner.py` 的現有下游 scorer 支援完整 caption retrieval any-hit 與既有 CORE metric。SugarCrepe++、Half-Truth 的 paired benchmark execution 應接 root 的共用資格框架；模型 wrapper 已可計算影像與文字向量，本目錄未宣稱 paired P0 已執行。

CLIC `pre_caption(max_words=30)` 是作者 retrieval 專用。SugarCrepe / SugarCrepe++ paired protocol 須在獨立 frozen model spec 把 `native_protocol.text_cleaner` 設為 raw tokenizer profile，不能沿用 retrieval30-word 清洗。CLIC Table1/3 另外保留 SugarCrepe 的 add/replace/swap 列，可和 SugarCrepe++ 作兩個原生 benchmark 資格測試，避免把缺轉檔 provenance 的 COCO adapted run 當 exact。

授權以各固定 modelcard/source 的實際宣告為準。Google/GME/UniME-V2 modelcard 宣告 Apache-2.0；UniME-V2 與原版 NegCLIP code 另有 MIT。此次取得的 CLIC/TripletCLIP code/model release 未找到明確 license 檔，清冊如實標未宣告，不推定授權。
