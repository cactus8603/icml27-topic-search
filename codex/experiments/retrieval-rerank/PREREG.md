# 固定 CORE 前 50 名重排

`26534` 的兩個融合規則都在開發集選出 0 權重。原整句 CORE 在該測試池的標註 oracle 若只改前 50 名可提高 40.93 個百分點。這次用既有 CORE cross-encoder 在另一個池實測這個固定重排限制。[上一輪判讀](../../reports/2026-11-15-topic-status-codex.md)

## 候選、查詢與標註

- 從已鎖定 SHA-256 的 COCO train2017 標註，用種子 `2026111534` 對影像 ID 雜湊排序，取前 4,096 張。排除 `26534` 的兩個池、全部 COCO val2017 和 Half-Truths 的 COCO ID 與已知逐位元相同影像，再排除 Controlled Images B 及池內重複檔案。互斥檢查涵蓋來源 ID 和精確檔案內容。
- 只收兩個不同 COCO 類別的整句。每類取非 crowd、標註面積至少佔全圖 2% 的物件。兩類都成立為相關，兩邊各須有一個只有一類的對照圖。排除 E2、`26450` 和 `26534` 的主要查詢，以及 `26450` 作為 foil 或 anchor 評過的雙類查詢，同樣依無序類別配對。Controlled Images B 的空間關係查詢與本次類別並列查詢分屬不同定義。納入所有符合條件的查詢。
- 同一個相關候選可能支援數個查詢。每個查詢取種子 `2026111535` 雜湊最小的相關影像做群組。固定下限為 4,056 張、60 查詢、45 群。下載失敗或無法完整解碼為一般點陣格式時中止。可解碼但非 JPEG 或尺寸與標註不符的官方原圖排除，記 ID、SHA-256、格式、標註與原檔尺寸。保留既定抽樣名單，排除後無補抽。其餘 JPEG 用精確雜湊排重。完整池、標籤與檔案校驗清單只放叢集，這裡以 `panel.json` 綁定。這組標註會漏掉未標註或小於門檻的物件。不同編碼的近似影像和模型預訓練重合未量測。 [COCO 下載與使用條款](https://github.com/cocodataset/cocodataset.github.io/blob/master/dataset/download.htm)

## 測量與門檻

- 基準是凍結的 CORE-EMB-2B 對整句與候選的 cosine 排序，同分依影像 ID。凍結的 CORE-RERANKER-2B 用官方 prompt 與 `yes`、`no` logit 差只評原前 50 名的查詢影像對。按差值由高至低重排，同分按基準排名，後方保留原順序。兩端共用前 50 名成員及完整候選池，沒有訓練或調參。
- 主要統計為 reranker 減基準的全池配對 mAP，含名單外正例。以正例影像群集重抽 10,000 次，種子 `2026111536`，計算單一對比的中央 95% 百分位區間。點估計至少 +2 個百分點且下界大於零為 PASS，上界小於 +2 為 FAIL，其餘為灰區。PASS 的區間只要求排除零，不能聲稱效果的下界已達 +2。群集區間對目前固定語料與群組定義解讀。
- 同時報固定前 50 名的標註 oracle、recall@50、前 50 名全無正例的查詢量。Oracle 只表達該名單對此標註的天花板。最終只寫一份彙總收據。

## 資源與入口

最終面板 4,095 張、104 查詢、93 群，91 條查詢只有一個相關候選。官方檔案 `320612` 為可解碼的 PNG，尺寸符合原標註，依前述格式規則排除。4,096 個原始圖檔共 666,772,611 bytes，含重試與保守預留的下載上界為 803,389,577 bytes。

共用已緩存的兩個模型，依序載入。固定評分 5,200 個查詢影像對。一張 H200、8 CPU、48 GiB RAM，估計 10–45 分鐘，作業上限 90 分鐘。此估算尚待本面板 GPU 實測。無新模型下載，影像下載累計最多 2.5 GiB，CUDA 結果只寫聚合 JSON。先在原 repo checkout 執行 `prepare.py verify` 重建來源互斥清單，凍結後從封存程式驗證已綁定的面板和本地模型。

```bash
shared=/work/ChihChungHsu/cactus8603/icml
stage=codex/experiments/retrieval-rerank
"$shared/envs/vllm033/bin/python" -IB "$stage/test_evaluate.py"
"$shared/envs/vllm033/bin/python" -IB "$stage/evaluate.py" --preflight
"$shared/envs/vllm033/bin/python" -IB "$stage/prepare.py" verify
bash "$stage/run.sbatch" FULL_FREEZE_COMMIT --check-only
```

## 讀取紀錄

- [codex] 2026-11-15 11:02（台北），Codex 已讀 `26568/summary.json`。凍結版本 `53027bd`。
- [codex] 2026-11-15 11:02（台北），獨立子代理已獨立讀取並複核同一份彙總。
