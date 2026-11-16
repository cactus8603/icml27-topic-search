# [codex] Additional official baseline preparation

本目錄是來源、權重、執行介面的事前準備。未產生研究結果，未啟動 GPU。`registry.json` 每項模型均為 `P0_REQUIRED`；下載、CPU import、形狀對齊及純色輸入成功均不構成 P0 qualification。

各 checkpoint 必須先凍結自己的完整模型設定、依賴檔 SHA、資料原始版本、published target、指標方向、容差及失敗處理，再由 root 排程官方資格測試。至少兩個官方 benchmark 的資格與獨立審查通過後，才能由共用 qualification gate 納入 P2。不得按結果更換模型變體、模板、精度、批次或資料切分。

`adapters.py` 提供 `ExtraOfficialModel(spec, device=...)`、`encode(items, image=False)`、`signature()`，和既有 producer 相同。`runner.py` 只接事前資格 P0；沿用已凍結 producer 的續算、收據及失敗處理。原生 conventional any-hit Recall 另在 CPU 計分，不使用 CORE 的 I2T caption-fraction 代替。P2/P3 由 root 的新 panel 模組接線，這裡不自動授予資格。

固定區分：

- SigLIP2 是 Google `so400m-patch14-384`，不是 NaFlex 或 256/512 解析度。So400m 是視覺塔家族大小；完整模型的參數數量依 CPU meta/權重鍵驗證記錄。
- CLIC `ViT-B-32`、CLIP `ViT-L-14` 和 CLIPS `ViT-L-14` 分開。CLIPS 用作者 `clips` 實作及 BERT tokenizer（長80），不是標準 CLIP tokenizer（長77）。三者都是 CogVLM/LAION 版，不能套 RedCaps 最佳列。
- TripletCLIP CC12M 與其 OpenCLIP finetune 版分開。HF 名稱 `ViTB12` 表示12層；實際 patch=32。
- 原版 NegCLIP 是作者 COCO finetune 的 Drive checkpoint。TripletCLIP 作者發布的 `CC12M_NegCLIP` 是重新預訓練 baseline，不是同一 checkpoint。
- UniME-V2 2B/7B 用原作者 `MMEBModel.load`、`Qwen2_VL_process_fn`、`process_image`、eos pooling，另有自己的 Table2 targets。CORE 引用的原版 UniME Qwen2 2B/7B exact 公開 checkpoint 尚未核實，不能用 V2、Phi3.5 或 LLaVA 權重冒充。
- GME7 沿用 CORE 官方未修改的 Qwen2 wrapper，與已凍結 GME2 相同 family/processor/i2t protocol。模型名7B與完整權重實際參數數量分開記錄。
- CS-CLIP 重用既有 `afd6e90f...` 權重，補下載同 revision 的設定與 modelcard。

來源與轉接差異：

- Google 原生 SigLIP2 target 使用 BigVision/JAX/TF；原作者 issue #166 指定 lowercase、Gemma bos=no/eos=sticky、長64。此 wrapper 是 Google 官方 HF PyTorch 轉換，固定長64 padding 與 lowercase；PIL/HF 影像處理差異須在資格測試揭露。
- Google `coco_captions/2014` 的 test 是 Karpathy5k。它與 CORE val2017 僅593張交集，不可互換。Flickr 原 Karpathy 與 CORE HF caption 字串也不能無聲互換。
- CLIC paper 明寫 COCO Val2017，官方 code 卻下載 `coco_karpathy_test.json` 而讀未發布的 `coco_karpathy_test_2017.json`。來源尚有缺口；自行以 canonical Val2017 captions 補檔須標 adapted，不可宣稱 exact native qualification。
- 作者 Drive 權重無公開 immutable revision。以釘版作者 source commit、file ID、原檔 SHA 和每 tensor 完全相等的 safetensors 轉換收據識別。只移除自己下載、已驗證轉換的 optimizer/scaler 原檔，保留推論 tensors；從未開啟任意 pickle 執行。
- UniME `process_image` 和 CLIC `pre_caption` 以原始碼 AST 直接載入未修改函式，避免為單一前處理函式匯入訓練資料依賴。函式來源在 SHA 表內。
- OpenCLIP/HF factories 以相同 config 建立模型，將網路檔名改成已驗證的本機路徑。不得新增 fallback、量化、自動減 batch 或更換 activation。

CPU 驗證僅使用固定純色圖及合成句子，不開 benchmark dataset、labels 或主結果。`preflight.py` 檢查完整權重 SHA、原作者 processor、所有模型 tensor 鍵與形狀、已宣告 tied LM-head alias；大型模型只建 meta。CPU meta 的 eager attention 只用於形狀建構，正式模型仍依作者 FlashAttention2，沒有 CPU offload。

獨立依賴放 `dl/retrieval-matrix-extra/deps`：`open_clip_torch==2.31.0`；使用已隔離的 Qwen2 transformers4.51.3 / PEFT0.15.2 / FlashAttention2.7.4.post1 與 torch2.6.0+cu124，不修改前兩批環境。所有正式執行仍須綁實際 module path/version 與檔案 SHA。

CPU 執行方式：

```bash
CUDA_VISIBLE_DEVICES= PYTHONDONTWRITEBYTECODE=1 TRANSFORMERS_OFFLINE=1 HF_HUB_OFFLINE=1 \
PYTHONPATH=dl/retrieval-matrix-extra/deps:dl/retrieval-matrix/deps/qwen2-transformers451:dl/retrieval-matrix/deps/qwen2-flash:dl/retrieval-matrix/deps/core-p0 \
dl/retrieval3090/envs/retrieval3090/bin/python -B codex/retrieval_matrix_extra/preflight.py
```

GPU smoke 入口（root 完成獨立 plan、commit、排程後）：

```bash
codex/retrieval_matrix_extra/run.sh --stage smoke --plan FROZEN_PLAN \
  --model MODEL_KEY --dataset DATASET_KEY --out OUTPUT --freeze-commit FULL_COMMIT
```
