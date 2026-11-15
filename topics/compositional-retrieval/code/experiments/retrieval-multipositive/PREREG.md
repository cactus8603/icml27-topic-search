# 多正例左右關係的 anchor 排名驗證

用新圖組檢驗 Half-Truths 的同圖文字排序訊號能否帶來跨圖 mAP 收益。先前固定前 50 名的多物件合取試驗中，真 anchor 的額外增益未定。這輪固定小候選組，把正例完整納入，同時將目標明確限定為左右關係。[原同圖訊號](../retrieval-anchor-audit/REPORT.md) · [四臂固定排序](../retrieval-anchor-rerank/PREREG.md)

- 使用官方 COCO train2017 標註，排除先前圖池的 COCO 與 Flickr ID，以及已核對的歷史左右圖組物件對。每個物件類別須只有一個非 crowd 實例，標註面積各至少全圖 2%。水平框互不重疊，間隔至少圖寬 2%，垂直重合至少矮框高度 25%。固定雜湊及稀有組優先的規則產生 **123 組、984 張圖、246 條查詢**。每個物件對左右方向各四張，兩個相反查詢各把同一八張全數排序。metadata SHA-256 `33363955507ec1c7cad716c3294f41c7892b12d944760631ce42fb85894295bb`。評分前核對來源、檔案大小、JPEG 原始尺寸、解碼與可核對的重複 SHA，影像收據另行凍結至 `assets.json`。固定圖名單無替換。源頭重壓縮、近似圖與模型預訓練重合仍可能存在。[官方 COCO 下載來源](https://github.com/cocodataset/cocodataset.github.io/blob/master/dataset/download.htm)
- 原同圖風格診斷以 CS-CLIP 比較同圖正確方向全句、只保留兩物件的真 anchor、以及反向全句。另報正確全句勝過反向全句的比例。anchor 刪除關係，保留兩個名詞。打亂 anchor 使用同句型，按固定雜湊、跨物件對一對一配置，且兩種 donor 名詞均不同。打亂臂是語意不匹配的控制。文字消融也會改變句長與分數校準。
- 沿用固定 CORE 交叉編碼 `yes−no`、CS-CLIP 與秩轉換 `U`。四臂為 CORE 原全句 `R`、`U(R)+0.25 U(F)`、`U(R)+0.25 U(F+0.125(F−A))` 的真及打亂 anchor 版。`F` 是 CS-CLIP 全句，`A` 是相應 anchor，同分按圖 ID。1,968 組 CORE 查詢圖對均計分，兩模型依序載入，無召回截斷或開發調權。
- 主量為真 anchor 相對 CORE、全句融合、打亂融合的三個配對 mAP 差。每個物件對的兩個方向一起重抽，沿用 10,000 次重抽與 Bonferroni 中央 `1−0.05/3` 區間。三個點估皆 ≥+2 個百分點且校正下界皆 >0 為 PASS。任一校正上界 <+2 為 FAIL，其餘為未定。以已知標籤把每組四個正例全排在前的 **100% mAP** 為同組事後參考。這個參考僅涵蓋固定可判斷的八張，完整圖庫召回及實際使用場景需另行量測。

使用既有權重和官方圖檔，下載取原官方 S3 bucket 的 HTTPS 端點，累計到貨 **145,472,785 bytes**，下載保留額上限 **512 MiB**。GPU 路徑配置一張 H200，依先前兩個模型循序執行的實測，估 **3–7 GPU 分鐘**，上限 25 分鐘，記錄實際顯存及耗時。舊同模型實測 **6.19 GiB**，本輪 24 GB 卡的可行性屬估算。圖檔與推論摘要保留於叢集。

```bash
stage=codex/experiments/retrieval-multipositive
bash "$stage/run.sbatch" FULL_FREEZE_COMMIT --check-only
sbatch --parsable "$stage/run.sbatch" FULL_FREEZE_COMMIT
```

## 讀取紀錄

- [codex] 2026-11-15 15:17（台北）已讀 `icml/runs/retrieval-multipositive/26679/summary.json`。Slurm 成功，摘要 SHA-256 與執行紀錄相符，凍結版 `a84de2f`。
