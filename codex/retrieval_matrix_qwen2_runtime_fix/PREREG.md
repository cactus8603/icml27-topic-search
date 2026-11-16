# [codex] Qwen2 缺少依賴的獨立修復批

2026-11-16（台北）。原 GME2／VLM2Vec2 在任何模型 smoke forward 前，PEFT→Bloom→flash_attn.bert_padding 因缺少 einops 中止。只檢查技術失敗 log，未讀任何模型效果。

此新批的 embedding mode 從原 commit `1c328a2c35c486f1768e9e9fea1c51f98dbb226d` 完整 P0 plan 複製；MCMR mode 從 `65f4f4850a82adf283b34dfddc319b72657562e6` 的 Qwen2 MMEB-v2 plan 複製。各自 models、datasets、settings、dependencies、精度、批次、模板、權重、DoRA 載入、計分公式及判準均保持相同；MCMR 的 image+metadata、104981 products、3997 queries、模型batch1/storage64/query32及native top100語意不變，仍不具 strict CORE P0資格。唯一新增執行依賴為 isolated einops0.8.2 overlay，來源是已存在的獨立 rerank 環境；不改任何舊來源或依賴 payload。完整 overlay bytes、module paths／versions 與新 source table 必須凍結。

只做 CPU imports、原 processor、tiny random model text/image forward 與 DoRA roundtrip/merge 技術驗收；CPU tiny model 使用 eager attention 是技術探針，不是 production fallback，正式仍為原 BF16/FlashAttention2。這些驗收不產生官方重現數字或替代真 GPU smoke。

正式 plan 必須經非實作者獨立審查、設 FROZEN/reviewed=true、commit 並保持工作樹乾淨，交 root 中央排程於 `dl/retrieval-matrix-qwen2-runtime-fix/runs/` 下全新輸出位置執行；入口拒絕舊 run 路徑。未經授權不自行 GPU、重試、排程或改舊失敗收據。若正式 smoke 再失敗，保留失敗，不改設定或自動 fallback。

主結果讀取另凍結 analysis PREREG、記錄來源 plan SHA／freeze commit及實際讀取時間；本批 producer bytes 保持凍結。
