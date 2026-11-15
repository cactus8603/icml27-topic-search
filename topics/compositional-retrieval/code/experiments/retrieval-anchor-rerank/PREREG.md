# 固定 CORE 重排名單中的 anchor 增量

原 Half-Truths 入池訊號來自單張影像上的三句排序。第一次新庫排名 `26450` 以 CS-CLIP、開發集選出 `λ=0.125`，在固定索引未達預登記增益。`26568` 固定 CORE 前 50 名的交叉編碼使用完整查詢，未計算 anchor。本次在全新候選池測試把既有 anchor 調整疊加在固定 CORE cross-encoder 上。 [原訊號拆解](../retrieval-anchor-audit/REPORT.md) · [原跨圖評估](../../killtest/PREREG_prospective_rank.md) · [固定重排協定](../retrieval-rerank/PREREG.md)

## 面板與排除

- 用固定 COCO train2017 標註、影像種子 `2026111541` 雜湊排序取 8,192 張。排除已登記 Half-Truths、COCO val2017、`26534` 開發與測試、`26568` 全部候選及已記錄排除圖相同的來源 ID 或可核對 SHA，並排除 Controlled Images B 及池內的完全相同圖檔。這次檢查涵蓋 ID 和逐位元同一性。重壓縮近似圖及模型預訓練重疊未量。 [COCO 官方資產](https://github.com/cocodataset/cocodataset.github.io/blob/master/dataset/download.htm)
- 非 crowd 物件原標註面積至少全圖 2%。各查詢由兩個官方類別構成，兩類皆有為相關，須各有一個只含單類的硬負控制。排除 E2、`26450`、`26534` 和 `26568` 的類別查詢，包含 `26450` 曝光的二類 foil 或 anchor。由指定種子固定每查詢刪去哪一類，另一類為真 anchor。打亂從本池其他 query 的 anchor 做一對一 donor 配對，donor anchor 類別不得出現在接收 query。無分數選擇或權重校準。
- 驗證下載內容為可完整解碼的常見點陣圖。非 JPEG 或與原標註尺寸不同則記錄 ID、SHA、資料格式與期望／實得尺寸後排除。下載不完整、非影像或無法解碼則中止。保留原 roster，排除後無補抽。最後門檻為至少 8,110 圖、100 查詢、85 群、每個查詢一個合法打亂 donor。正式新池依 `panel.json` 的 manifest SHA 凍結。

## 排序與判讀

- 三個固定權重模型分批依序載入。CORE-EMB-2B 全句分數決定整池排名與每查詢前 50 名。四臂共享候選成員及尾序。CORE-RERANKER-2B 的原整句 `yes−no` 是同榜強基線 `R`。CS-CLIP 只對 shortlist 唯一影像與三組文字計算餘弦。令 `F` 是 CS-CLIP 整句，`A` 是真或打亂 anchor，`U` 是同一前 50 名的平均名次百分位，方向由低至高。
- 固定四臂為原 `R`、`U(R)+0.25 U(F)`、真 anchor 及打亂 anchor。兩個 anchor 臂都是 `U(R)+0.25 U(F+0.125(F−A))`，唯一替換為 anchor 的身分。檢查 `U(R)` 維持原排名，所有分數同分依原前 50 名順序。只重排名單內；整池 AP 分母包含尾端的正例。秩融合權重從本輪前固定，所有面板資料只用於驗證。
- 主量為真 anchor 相對另外三臂的配對全池 mAP 差。以固定正例影像作群，10,000 次配對群集重抽，同一抽樣涵蓋三個比較。Bonferroni 中央 `1−0.05/3` 區間。三個差點值皆 ≥+2pp 且區間下界皆 >0 為 PASS。任一區間上界 <+2pp 為 FAIL，其餘為未定。另報固定 shortlist 的標註 oracle、recall@50 和沒有 shortlist 正例的查詢數。區間只描述這個固定候選池，結論限於本次二物件 conjunction 與固定融合。

## 執行

使用既有 COCO、CS-CLIP 與 CORE snapshots。公開圖檔初估約 1.24 GiB，累計下載上限 2.5 GiB。無額外模型下載。H200 一張、8 CPU、48 GiB RAM；先估 5–12 分鐘，作業上限 30 分鐘。先前同 pipeline 的 CORE-only 26568 實測 6.16 GiB、3 分 23 秒。24 GB 逐模型載入有資源基礎，須以相容的本地 CUDA 與封存版本跑小 smoke 後確認。

凍結面板有 8,191 張圖、136 個新查詢、118 群，105 個查詢只有一個標註相關候選。下載收據共 1,330,217,839 bytes，按原逐位元去重規則排除一張。固定評分 6,800 組原前 50 名查詢影像對。

```bash
shared=/work/ChihChungHsu/cactus8603/icml
stage=codex/experiments/retrieval-anchor-rerank
python="$shared/envs/vllm033/bin/python"
"$python" -IB "$stage/prepare.py" plan
"$python" -IB "$stage/prepare.py" fetch
"$python" -IB "$stage/prepare.py" verify
"$python" -IB "$stage/test_evaluate.py"
"$python" -IB "$stage/evaluate.py" --models-only
bash "$stage/run.sbatch" FULL_FREEZE_COMMIT --check-only
```

在本 repo 重建 `verify` 後，將 receipt 的 `manifest_sha256` 填入 `panel.json`，複核兩份一致再凍結。執行入口只輸出一份正式聚合 `summary.json` 至叢集路徑 `runs/retrieval-anchor-rerank/<job-id>/`。

## 讀取紀錄

- [codex] 2026-11-15 11:52／11:54（台北），獨立子代理與 Codex 分別讀取 `26601/summary.json`，凍結版本 `2daedf522c6d64b860277657d3067b61da3a0f2b`。依收據訂正面板排除原因為逐位元重複，候選名單及判讀規則依凍結版。
