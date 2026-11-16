# [codex] P2 面板與 P0 准入接線

[codex] 2026-11-16 20:54（台北）。此模組接續報告64，不更改已排隊的 P0 程式與預登記。

P0資格由已凍結模型身分、全部首批COCO／Flickr／NegBench官方計分報告，以及非產生者的獨立重算收據共同確認。三欄都必須提交並保留，至少兩個不同資料集的主目標誤差≤0.01。COCO與Flickr只用官方native-dtype、pytrec的T2I R@5；NegBench用官方native top5。CPU stable-order診斷不可替代。MCMR renderer尚有未驗證差異，不借其兩個指標當兩個資料集。

資格是資料與身分核對，不自動作研究停損判斷。未完成的模型不填0、不標PASS。只有具原始報告及獨立重算證據、模型規格與P0一致的資格收據，才可建立P2正式執行計畫。不同權重／模板／batch不能沿用資格。

P2後續開發使用 `DEV_SPEC.md` 固定的COCO development v2：連言dev-full≥1,000為主；small保持完整20,000圖庫；否定版若不足1,000只作診斷。保留v1歷史結果與曝光說明。自建test不作方法選擇；固定基線若需量測test須另外明示凍結授權。

COLA只作測試。官方single-object mAP只在每類hard_list選出的圖上評分，分數平手按完整threshold group計算AP；不能改成任意逐圖平手次序。檢索候選仍取自完整庫，gold/hard/ignore不進模型與取K。每個資料集獨立報告，seen/unseen若公開來源缺少split清單，留為不可驗證，不自行猜分。

主K100，補充50/200；完整庫取K後計分去ignore、不補K。正式效果須先登讀取時間，機械輸出後由非該模組實作者重算，再交使用者判讀。

## 讀取紀錄

- [codex] 2026-11-16 20:54：僅讀公開協定、資料數量／ID及既有凍結plan，未讀本輪P0主判準結果。六模型18格仍在等待空卡。
