# [codex] SigLIP2 native P0 qualification preparation

本模組僅執行固定官方 baseline 的原生 P0；目前沒有模型效能結果，不能據此宣稱 P0 PASS 或接納 P2/P3。GPU 工作由 root 在本目錄、plan、來源與相依檔案凍結後，交既有中央排程器處理。本模組不另建佇列、不自行送 GPU，也不自動重試失敗工作。

模型為 Google 官方 `google/siglip2-so400m-patch14-384` 固定 revision `e8e487298228002f3d8a82e0cd5c8ea9c567f57f`，視覺 So400m 家族，完整模型共 1,136,008,498 參數。使用官方 HF PyTorch conversion、完整 FP32，無量化或 offload。文字全部 lower-case、max length 64、padding max_length、truncation true，圖片使用該固定 checkpoint 的官方 HF processor；normalize 在 FP32。固定 image batch 4、text batch 32，技術 smoke 比較 batch1 與此固定 batch，cosine >= .9995、max abs <= .01；失敗保留收據，不依主結果調 batch。

COCO 用原始 Karpathy test5000、全部 25,010 句 raw captions（10 張有第六句，不能截成 25,000）；Flickr 用原始 Karpathy test1000、5,000 句。loader 逐列與原始 JSON 核 image IDs、caption 原文、sentence IDs，核完整 image byte/SHA 收據；圖庫保存原始 metadata，模型只接收路徑或文字。沒有標籤過濾、ignore、開發／測試混用或缺圖排除。CORE val2017 與 COCO Karpathy test 只有 593 張相同，不得互代；CORE HF Flickr 文字與原始 Karpathy 有 574 句不同，不得互代。

計分使用固定 Google BigVision `image_text_retrieval.py`（revision `0127fb6b337ee2a27bf4e54dea79cff176527356`）原函式：CPU FP32 `numpy.dot(images, texts.T)`，對負相似度 `numpy.argsort` 預設 quicksort，保留原始 Karpathy 列序與句序；同分不改成穩定排序或 torch.topk。雙方向皆 conventional any-positive Recall@1/5/10；主要資格只比較 Table1 的雙方向 R@1。每個資料集的所有主要方向都需距目標 <= 1 個百分點，至少 COCO 與 Flickr 兩個不同 canonical benchmark 同時達標；同一資料集的兩個方向或不同子集不能算兩個 benchmark。所有事前宣告的 benchmark（包含失敗）一律保留。

差異事前揭露：官方 HF PyTorch port 與 JAX 不是同一數值路徑。HF PIL bilinear resize384 與 BigVision TF bilinear resize（antialias=False、之後 cast uint8）亦非逐像素相同。TFDS 列序可能與 Karpathy 原列序不同，故 exact ties 有差異。Google 作者重現留言明指 COCO TFDS coco_captions；原 paper 只標 Flickr，公開留言沒有發布其 Flickr loader。因此獨立審查者必須明確檢查並接受／拒絕此 port 與 Flickr split mapping；未審查時資格不能 PASS，也不能稱為 exact JAX reproduction。必要修訂須於任何真正模型結果產生前另行凍結，不能看效能後選前處理。

完整 plan 記錄官方 weight/tokenizer/config payload SHA、來源 SHA、相依程式 payload SHA、完整分母與 primary metric 的方向／定義／variant／原文表格與來源 SHA。GPU stages 僅 smoke/index；index 復用既有 retrieval_scaling 的逐 batch checkpoint/SHA 收據，不重造續算協定。audit/analyze/review/qualify 是獨立 CPU stages。每個 benchmark 報告繫結原始 FP32 embedding arrays、plan/model/data/protocol/source hashes；不能用 BF16 score matrix 代替原始 FP32 arrays。

獨立審查必須由非產生者執行 `qualification review`，使用另一個 CPU 實作由原始 arrays 重算全部雙方向 R@1/5/10，與原 scorer 完全一致，並核原始資料／完整分母。收據明記 reviewer/producer、所有 report/array SHA、重算程式 SHA、差異審查。`qualification qualify` 機械比較目標；`admit` 重新核所有證據與來源才允許後續接線。P3 還須一個已合格 CLIP 與一個已合格 MLLM；兩個 MLLM 不足。

操作：先執行 `python3 codex/retrieval_matrix_native_p0/build_plan.py` 產生未執行 plan。CPU audit 用 `bash codex/retrieval_matrix_native_p0/run.sh audit --plan codex/retrieval_matrix_native_p0/p0-siglip2.json --out dl/retrieval-matrix-native-p0/preflight`。未凍結與中央派工前不執行 smoke/index。root 未來分別派 smoke、兩個 index，再於 CPU 執行兩個 analyze。資格 review/qualify 由不同身分操作，且 `--accept-disclosed-protocol` 是審查者明示接受上述差異，不能因指標達標自動加上。
