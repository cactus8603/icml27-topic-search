# 研究題目排程

[題目狀態、介紹與設計](README.md) · [共同判準](criteria.md) · [作業紀錄](killtests/SUBMITTED.log)

## 主攻（2026-11-17）

1. [多條件檢索](topics/compositional-retrieval/)　第一主攻，本機 3090。
2. [多保真度資料歸因](topics/multifidelity-attribution/)　H200 第一順位跑多種子期望真值停損（約 8–10 H200 小時）；現象與空間都 PASS 後升為 H200 主線，排在 Λ 與文生圖之前。

## 已通過入池判準

- [多條件檢索](topics/compositional-retrieval/)的同圖三句排序通過。跨圖的新召回方法由各自的實驗設計驗證。

## 優先完成現象與訊號

1. [用訓練前的 Λ 預測防禦漏出](topics/inoculation-lambda/)
2. [文生圖罕見失敗的機率估計](topics/t2i-rare-failure/)

## 暫停

- [RL 稀疏決策蒸餾](topics/rl-sparse-distill/)　2026-11-17 使用者決定暫停：完整論文約需 500–800 H200 GPU 小時，時間不夠（[報告 63](reports/2026-11-16-63-full-paper-resources-and-allocation.md)）。

後續探針與判讀更新在各題資料夾。GPU 作業依協作規則與當前佇列安排。
