# 多條件影像檢索｜目前結果

**現象**　原同圖測試通過入池門檻，但最強固定索引模型 CS-CLIP 的嚴格三句排序只有 58.15%。新的左右關係面板上，CS-CLIP 對正句和方向顛倒句的判對率是 49.59%。[同圖判準](code/killtest/PREREG_headroom.md) · [左右關係測試](code/experiments/retrieval-multipositive/PREREG.md)

**訊號**　同圖的完整句勝過半真句達 92.34%。在獨立圖庫用現有 CORE 逐張看圖與句子、重排前 50 名，mAP 增加 15.79 個百分點。把只有兩個物件的短句分數額外加入重排，在新的左右面板未達預設增益。[獨立重排](code/experiments/retrieval-rerank/PREREG.md) · [短句對照](code/experiments/retrieval-multipositive/PREREG.md)

**改善空間**　左右關係小面板的 CORE 為 87.13% mAP，距同組標註上限 12.87 點。另一個大圖庫固定前 50 名的差距是 19.50 點。31/136 條查詢的前 50 名沒有任何標註正例，要救回這些題得先改進召回。[大圖庫四臂對照](code/experiments/retrieval-anchor-rerank/PREREG.md)

**最新結果**　作業 `26806` 以同樣 50 次重排完成。交錯候選的 Recall@50 多 **0.25 點**，全庫 mAP 低 **0.44 點**（95% 區間 −3.84 至 +2.93 點），依預登記判為未定。[獨立複核](../../codex/reports/2026-11-16-retrieval-recall-budget-codex.md)

## 信心、潛力與資源

- **證據信心 4/5**　兩個評估面板都量到可改善的差距，獨立圖庫也量到現有重排的實際收益。
- **研究潛力 4/5**　如果同樣重排成本能找回原先漏掉的相關圖片，且在新圖庫重現，就能提出可實際部署的改進。
- **下一個實驗與資源**　`26806` 實測 1 張 H200 跑 6 分 15 秒，約 0.10 GPU 小時，不含排隊。沿用約 1.3 GiB 的圖庫快取，新輸出小於 256 KiB。新圖庫的獨立查詢及標註清冊已封存，後續須取得候選圖並驗收。[資源估算](code/experiments/retrieval-recall-budget/PREREG.md) · [硬體實測](../../codex/HARDWARE.md)

## 深入調查

Half-Truths 已提出同一張圖上比較完整、基準與錯誤描述的測法。CORE 也已把部分符合與完全符合區分開來，讓檢索模型學習更細的排序。

我們要驗證的是首階段能否找回原本沒進候選名單的相關圖。兩組都只精排 50 張，再換一座獨立圖庫確認收益，並計入額外搜尋成本。 [檢索文獻深查](../../reports/2026-11-14-46-deepcheck-compositional-retrieval.md)

[回題目首頁](README.md)
