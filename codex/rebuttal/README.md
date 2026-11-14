# ICLR 2027 Rebuttal 案例

這份筆記整理六篇和擴散語言模型解碼、推論效率或組件消融相近的 ICLR 2027 投稿。每個案例都能對上公開的原評語、作者回覆與 reviewer 說明。實驗數字來自作者回覆，未獨立重跑。分數是 11/06 21:54 到 11/14 20:12 的討論期間快照，時間為台北時間。每篇的評分向量由高到低排列，個人初末分只有 reviewer 明說時才逐一標示。 [Radar 固定快照](https://github.com/dipta007/iclr-2027-rebuttal-radar/tree/a67aefa2c4c2d231e471abab4c36cb771c672ec9/docs/data) · [ICLR 官方時程](https://iclr.cc/Conferences/2027/Dates)

| 疑慮 | 建議先讀 | 作者實際補了什麼 | Reviewer 結果 |
|---|---|---|---|
| Baseline 不公平 | [CoCom #20666](iclr2027-cases.md#cocom-20666) | 重跑同 block length 的 Native，揭露原本的 GSM8K 優勢在對齊後反轉 | 品質相近而 forward 次數大幅下降，升到 4 |
| Novelty 像多個 heuristic 疊加 | [TACG #32163](iclr2027-cases.md#tacg-32163) | 固定跨任務參數、限制每輪額外提交數、做同 block 實測延遲 | 因診斷與實用效率加分，現為 3，仍追問核心新意 |
| 機制與 baseline 混在一起 | [Sample the Order #50735](iclr2027-cases.md#sample-the-order-50735) | 拆開 token sampling 和 unmasking 排序的溫度，發現短答的強 baseline | 表示加分，現為 3，要求把引言的主張收窄 |
| Ablation 少了貼近先前工作的對照 | [RestoreKV #28487](iclr2027-cases.md#restorekv-28487) | 固定八個 restore states、同最終 cache 預算，只改訓練時是否與保留的 KV 聯合 | 指出對照實質強化區別於既有方法的貢獻，現為 3 |
| 局部消融不足、步數不等於成本 | [Dynin-Omni #11132](iclr2027-cases.md#dynin-omni-11132) | 跑完整 EOS 排程與等預算的合併後訓練，另外量 FLOPs、實際延遲及配對信賴區間 | 兩位明說升到 3，分別點名完整路徑、對照與成本分析 |
| 理論澄清了，重現仍有疑慮 | [Rethinking Soft Tokens #23776](iclr2027-cases.md#rethinking-soft-tokens-23776) | 遮住被評估位置、修正 SLERP 的幾何解釋，後續補更新設定檔 | 明說 1 → 2，當時仍質疑設定檔與 baseline 重現 |

## 回覆可以怎麼寫

**Baseline 公平性。** 開頭直接列原先兩個設定哪裡不同。重跑後把舊 baseline、修正後 baseline、方法結果並列，再交代結論剩下哪部分。如果省的是 forward passes，寫成 NFE 降幅。若主張延遲下降，另列同 GPU、batch size、輸出長度與後端的實測。

**Novelty。** 先承認最接近的既有工作已做到什麼，再用一句話說清楚自己改的是哪個決策、訓練目標或適用條件。接一個反事實對照，例如「相同模型、相同提交數、幾乎相同 NFE，只拿掉這個訊號」。主要改善集中在哪個場景，就把主張寫在那個場景。

**Ablation。** 對照要能區分兩個可能解釋。若兩個因素一起改，分開控制並報結果。若主張跨階段排程或合併後效果，跑完整路徑並固定後續訓練量。若主張一組參數可泛化，共用設定和每任務調參要分列，並寫清選參資料、隨機種子和信賴區間的單位。

這是一段可改寫的英文骨架，括號內用本次有證據的設定取代。

> **W2 — Fair comparison with [baseline].** We reran [baseline] and our method on the same [model, split, block length, and output budget]. The matched comparison is [baseline metric] vs. [our metric], at [cost A] vs. [cost B]. On [exception], [baseline] performs better on [metric]. We revised [section/table] to limit our claim to [supported regime].

[逐篇原評語、作者回覆和 reviewer 跟進](iclr2027-cases.md)
