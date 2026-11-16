# [codex] GME2／VLM2Vec-V2 首批啟動前獨立審查

[codex] 2026-11-16 20:33（台北）。root未實作Qwen2獨立adapter、producer entry或其PEFT測試；本次只讀審查與重跑CPU驗證，沒有GPU推論或新模型效果讀取。

**可以凍結執行六格：GME2／VLM2Vec-V2各自的完整COCO、NegBench、CORE指定Flickr。** 這是啟動前條件通過；GPU synthetic smoke、完整推論、±1pp論文重現與結果獨立驗收仍未完成。

## 審查證據

- 計畫 `codex/retrieval_matrix_qwen2/p0-qwen2-batch1.json` 使用已凍結首批P0的三資料集、官方CSV順序、原始全分母、native BF16 matmul、逐row torch.topk與pytrec語義。主模組未修改，獨立entry擴充模型constructor與PEFT／Flash實際版本收據。
- 新模組16個宣告來源SHA均符合目前檔案。CPU preflight收據 `dl/retrieval-matrix/qwen2-cpu/preflight-batch1.json` 的plan SHA與候選計畫相符且為PASS。
- 重新執行獨立依賴驗證：2,544個檔案PASS；實際torch2.6.0+cu124／Transformers4.51.3／PEFT0.15.2／FlashAttention2.7.4.post1，沒有改CORE/Qwen3首批的torch2.14依賴。
- 重新跑shared runner/metrics與Qwen2 adapter的21個CPU tests全通過。包含真正PEFT小模型的完整模型DoRA載入、保存、重載與merge後分數一致，亦包含V1／裸base不得冒充V2的拒絕測試。
- V2固定官方rank16 DoRA與Qwen2-VL-2B基底；CORE evaluation class與VLM2Vec native class的meta coverage均420/420，missing／unexpected／shape mismatch為0，含140個magnitude vector。這是所有保存tensor的鍵／形狀證據，不誤稱完整base在GPU的數值等價。
- 真實執行adapter在完整base上用PEFT載入；merge前再對每個保存tensor逐值核對其載入值，merge後要求全部BF16且在單一選定GPU，不能把未合併adapter或CPU offload藏在設定內。

## 明示的相容修正與限制

1. CORE generic loader遺漏V2 DoRA的套用步驟。本模組補上官方adapter及上游推論merge方式，保留CORE原本的Qwen2 processor、i2t_emb、pooling與normalization。這是明示修正，不能聲稱直接照有缺陷的generic loader就已重現。
2. V2 preprocessor存舊 `min_pixels/max_pixels` size鍵，獨立spec傳4.51.3所需的 `shortest_edge/longest_edge`；數值仍為CORE覆設後的4096／1310720。沒有按效果選解析度。
3. VLM2Vec現行來源列Transformers4.52.3，本模組使用與CORE/GME Qwen2 wrapper相容的4.51.3；CPU API／adapter coverage已過，GPU相容性及數字仍待驗。模型與資料版本、環境差異全部保留，不能將CPU測試當成已成功取得研究效果。

模型payload SHA與下載收據已由plan builder逐檔核對；本審查未再重讀全部13GiB payload。正式constructor仍會重新驗所有聲明檔案。後續若需變更batch／dtype／attention或處理器，必須另行凍結技術修訂，保留原失敗紀錄，不按test效果調參。
