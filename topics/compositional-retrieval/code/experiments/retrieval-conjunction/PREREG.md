# 兩項條件各自成立的固定索引檢索

同圖三句排序已通過。原 anchor 差分在新庫排名與獨立圖像對照均未達門檻。這次用不依賴 anchor 的兩項證據分數，直接比較全池 mAP，另量只改短名單順序的可達範圍。 [原判讀](../../../../../codex/reports/2026-11-15-three-frozen-readbacks-codex.md#多條件檢索)

## 獨立候選池

- 用已鎖版 COCO train2017 標註，從未用於前面排名試驗的影像均勻雜湊抽樣。排除 val2017 和原 Half-Truths 的 COCO 影像 ID，再排除與所有本地 val2017、三個 Half-Truths 子集或另一 split 逐位元相同的 JPEG。分別用固定雜湊分派影像和查詢，每類標註面積至少佔圖 2% 且非 crowd。
- 每個查詢由兩個 COCO 類別組成。兩個類別都標註才屬相關，兩邊各至少有一個只含單一條件的候選。排除既有 E2 及 `26450` 的全部相同類別 keys。依既有群組門檻，先做無模型分數的標註可行性檢查。初始 1,280／2,560 張面板在 DEV 群數不足，改用 4,096／8,192 張。固定圖冊與所有 JPEG 雜湊後才可執行模型；未預測近鄰影像或模型預訓練重合。
- 影像逐張取自 COCO 官方發布，未使用模型或檢索結果決定下載對象。驗證 JPEG 尺寸與原標註相符及可解碼，候選的本地 SHA-256 綁入 `panel.json` 指定的校驗清單。下載容量上限 3 GiB，沒有產生新模型或 checkpoint。 [COCO 官方發布與使用條款](https://github.com/cocodataset/cocodataset.github.io/blob/master/dataset/download.htm)

## 訊號與上限

- 同一 CORE-EMBED-2B 影像索引對各整句查詢與單類別文字評分。每個文字在同一池用分數平均名次換成經驗百分位。整句為 `F`，兩單項為 `U`、`V`。主要分數 `(1−w)F + w min(U,V)` 要求較弱條件也有證據。比較原整句與同樣文字、以 `(U+V)/2` 替代 `min` 的控制。兩種新規則都只用 DEV 各自從 `{0,.25,.5,.75,1}` 選最高 mAP，平手取較小權重。程式必須驗證整句轉百分位仍維持逐項名次，排序同分依 COCO ID。
- 單一主量為 TEST 查詢宏平均 AP。以固定指定的正例影像為群組，對兩組差同一批 10,000 次群集 bootstrap，使用中央 97.5% 百分位區間。兩個配對差點估計均至少 +2 個百分點且區間下界都大於零才通過。一個差區間上界低於 +2 即本規則未過，其餘為灰區。收據報告 recall@50 及單正例／多正例 mAP 分層供解讀。標註缺漏或門檻外物件會限制外推。
- 上限只改 CORE 整句的前 K 名，K=10、50、200，主要參考 K=50。固定短名單成員與名單外排序，用正式相關標註把名單中所有正例放前面，重算全池 AP。報告可回收 mAP、漏在 shortlist 外的正例比例和完全沒有正例的查詢量。這是指定候選名單對已標註相關性的事後上限，不能保證可學到的增益。

## 執行

沿用既有離線權重與固定 `params_headroom.json` 存檔。一張 H200，50 分鐘上限。上一組相同索引評測用時不到兩分鐘，池擴大後以 6–18 分鐘作暫估，VRAM 以本次實測為準。按正常流程凍結，再用相同入口預檢和提交。所有影像、標籤與作業分數落在共享叢集，Git 只保留程式和小型綁定。

```bash
shared=/work/ChihChungHsu/cactus8603/icml
stage=codex/experiments/retrieval-conjunction
"$shared/envs/vllm033/bin/python" -IB "$stage/test_evaluate.py"
bash "$stage/run.sbatch" FULL_FREEZE_COMMIT --check-only
sbatch --parsable "$stage/run.sbatch" FULL_FREEZE_COMMIT
```

## 讀取紀錄

* [codex] 已讀叢集 `26534/summary.json`，2026-11-15 09:08（台北）。凍結版 `6aa7eb5b97d50dff77af07074f51163813ae863d`。
* [codex] Astra 獨立複核已讀同一彙總，2026-11-15 09:09（台北）。
