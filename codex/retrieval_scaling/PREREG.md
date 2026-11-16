# CORE 2B／8B 固定成本檢索與剩餘空間

[codex] 2026-11-16（台北）。使用者要求實際量測：將目前 CORE50 換成大模型後，候選召回與排序的改善空間是否已消失。本輪只使用既有已曝光開發圖庫，不開啟封存面板。本文、機械設定及程式在新增研究推論或讀取新指標前提交。

## 固定問題、來源與比較

- 資料固定為 `dl/retrieval3090/runs/platform-torch214-20261116/panel.json` 的 8,191 圖、136 query，原圖逐檔對照舊 manifest JPEG SHA。原始查詢、正例、群、圖片順序不修改。
- 這是已曝光、由 COCO 類別／面積規則建立的 development panel；不能冒充獨立確認或人工完整真值。後續公開 NegBench 開發作業保持原狀，本輪不更動其 query、流程或輸出。
- 四格固定為召回／重排 `2B/2B`、`2B/8B`、`8B/2B`、`8B/8B`。每格每 query 50 對。兩個召回模型都先排序整個圖庫，前 50 交給指定 reranker；選入集合放前面，未入池圖片保留該召回模型原排序。平手依召回原名次。
- 使用作者官方公開完整權重。2B revision 沿用原平台；8B embed 固定 `123aef9e5e9e76fb686420f077fce3ef3b580c2b`，8B reranker 固定 `d95d36a34300e34b87cc1793827ac798642c4488`。8B embedding 為 4,096 維，不能和 2B 的 2,048 維快取混用。
- 延用既有凍結的模型封裝 `2daedf522c6d64b860277657d3067b61da3a0f2b:killtests/compositional-retrieval/scripts/headroom_models.py`，相同 image-only 提示、resize、token 上限、BF16、SDPA、`use_cache=False`，重排分數是 yes logit 減 no logit。不量化、不訓練、不 CPU offload。
- 主要環境為既有 torch214。影像 batch8、文字 batch32、cross batch4。兩個 8B 模型依序載入，技術 smoke 只檢查載入、形狀、有限值與記憶體，不能推定研究效果。若記憶體不足，停止原配置，先另行提交技術修訂，再重新計算受批次變更影響的 2B 控制；不得隱式改精度、像素或 batch。

## 重用、實際成本與完整性

- 原平台 2B 影像／完整句向量、完整召回順序與前 50 分數只在 panel、檔案 SHA、形狀、模型／處理設定與 batch 完全符合時重用。重新計算向量內積排序，核對保存的 full order。歷史 H200 指標不作本輪 baseline。
- 8B 重新建立整庫索引；2B reranker 另算 8B 入池的 6,800 對；8B reranker 對兩種入池各算 6,800 對。每格可部署成本仍為 50 次；離線為拆解機制另外付出的實驗成本分開登記。初版不跨臂重用 pair，以保持原始批次與順序。
- 每 batch 記錄來源指紋與完成進度，輸出目錄與舊作業分離，不覆寫舊 cache。完整配置、程式／plan／模型／資料／環境 SHA 綁定；不符合禁止續算。完成收據包含各 canonical 輸出 SHA。
- 記錄模型載入、實際階段 wall time、batch 有效推論時間與峰值 GPU 記憶體。索引建置及全實驗時間不冒充線上單題延遲；相同 50 calls 不等於相同實際成本。
- 顯卡需至少 20 GiB 空閒、utilization ≤20%，連續兩次確認並取得既有合作排隊鎖才啟動。純色 smoke 通過後，每個主階段載入前還須確認空閒記憶體不少於該模型 smoke 實測 peak reserved 加 1 GiB，且仍至少 20 GiB；不足就停止並保留快取，重新排隊，不改 batch／精度。無卡就等，不取消或干預他人程序。磁碟已見約 1.2 TiB 可用，本機沒有 `quota` 指令；新增模型約 31.5 GiB。

## 固定評估與判讀

- 每格報全圖庫 mAP、固定 Recall@50、無正例題數、候選固定 oracle mAP，以及 oracle 減實際 mAP 的排序差距。Recall@50 的容量上限按各 query `min(50/R,1)` 報出。
- Oracle 僅利用評估標籤，把目前入池 50 張中的正例放最前，尾序保持該召回模型原順序。因此相同召回的兩格 oracle／Recall 必須一致；換 8B 召回後重算，不沿用舊 oracle。
- 報 `2B/8B−2B/2B`（同池重排增益）、`8B/2B−2B/2B`、`8B/8B−2B/2B`、`8B/8B−8B/2B`；同池排序差距回收比例為前者除以原 2B oracle gap，另報其分子／分母。總增益不全歸因單一階段，並報兩階段交互差。
- Bootstrap 固定 seed `2026111668`、10,000 次，以「同原群或共享任一正例」的連通 query group 為抽樣單位，對 query 平均作配對群重抽；報 95% 雙側區間。不得把候選圖片數當獨立測試樣本。
- 事前固定描述性實質差距 epsilon=2 個 mAP 百分點。對 8B/8B 的剩餘排序 gap，以及 `100−8B oracle` 分別報單側 95% 界：下界 >2 表示該成分仍有可辨的實質差距；上界 <2 表示該成分已小於本輪參考差距；其他為不確定。兩成分上界都 <2 才描述為本面板兩種空間均已很小，不推廣成所有任務無空間。這不是停止整題的自動裁決。
- `100−oracle` 需要改變候選成員或尾序才能回收，不能全稱純召回損失；oracle gap 不是可實作方法的效果保證。只換大重排器若接近原 oracle，不能因此說候選漏找已解決。
- 原先最佳 M1 與本輪 8B 的比較只作既有開發觀察，不以曾經選後的 M1 代表事前固定新方法。最終判讀須經未實作對應模組的獨立審查後交使用者。

## 分工與讀取紀錄

- root：資產取得、plan、凍結、排程、紀錄與結果整合。
- scaling_runner：只實作 runner 與其技術測試。
- oracle_audit：只實作 CPU 分析與合成資料測試。
- sota_check：官方資產與輸入、程式／結果獨立核對，不修改執行實作。
- 凍結前已讀既有平台報告與先前公開指標；只對舊 cache 取 SHA／形狀作來源查核。未產生或讀取本輪新的模型指標，未碰封存資料。研究輸出首次讀取後在此追加具體時間與路徑，判讀規則不隨結果變動。

- [codex] 2026-11-16 17:32:02（台北），完整 GPU 作業 completed/0、COMPLETE 十檔收據通過後，依凍結版分析機械讀取 `dl/retrieval3090/runs/core-scaling-20261116/index/{2b,8b}_order.npy` 與 `cross/{e2_r2,e2_r8,e8_r2,e8_r8}.npy`，產出 `results/analysis/report.json`／`per_query.json`。未修改判讀設定；主代理尚未檢視數值。
- [codex] 2026-11-16 17:32:42（台北），root首次檢視 `dl/retrieval3090/runs/core-scaling-20261116/results/analysis/report.json` 全四格與區間；原判準維持。已授權未實作主分析的 sota_check 進行獨立重算，讀取時間另記。
- [codex] 2026-11-16 17:32:39（台北），sota_check 在 root 明示授權後首次讀取本輪 `results/analysis/report.json`、`per_query.json` 與 canonical 陣列，以不引用主分析函式的 CPU 驗證器獨立重算；原判準不變。
- [codex] 2026-11-16 17:35（台北），root 首次檢視 `dl/retrieval3090/runs/core-scaling-20261116/results/independent-review.json`；17:36 閱讀 `codex/reports/2026-11-16-core-scaling-review-codex.md`。將已核對機械輸出原樣保存到兩個 Codex 模組的 `results/` 供版本追溯；原判準不變。
