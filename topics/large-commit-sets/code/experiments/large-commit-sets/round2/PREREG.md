# 第二輪，大型提交集合的等探測預算比較

[codex] 2026-11-15，台北時間。設計時已讀第一輪彙整，使用另一批鎖定題目，尚未取得第二輪結果。凍結範圍涵蓋本目錄及 `paths.sh` 指定的第一輪共用程式。 [第一輪彙整](../../../../../../codex/reports/2026-11-15-rebuttal-pilot-results-codex.md)

## 估計對象

在 LLaDA-8B-Instruct 的自然 confidence 解碼軌跡上，搜尋固定第 2、10 輪的首個合格狀態。固定可重現的 BF16 快照、零樣本提示、256-token canvas、block 64、每輪 quota 8、32 次模型 forward、第一個 EOS／EOT 截斷和嚴格 `####` 數字解析。每題只介入一次，候選池沿用原 confidence 集合及兩個近 confidence、位置距離受限的一 token 交換。候選值均取自同一介入前狀態的 argmax。所有候選使用相同後續策略和 forward 上限，答案評分在完成三條軌跡後才取用。如果兩個狀態皆不合格，所有策略回退原規則，該題保留於答題分母。

以 parquet row index 為 doc_id，按 `SHA256(seed + ":" + split + ":" + doc_id)` 排序。測試排除第一輪確定性抽樣的 240 題及與其規格化後文字完全相同的題目，再將其餘在 test 內去重，固定取 1,000 題。訓練從官方 train 排除所有 test 的規格化完全重複、官方 train 前兩題（第一輪 smoke）並去重，固定保留排序最前兩題做第二輪 smoke，再取後 480 題做唯一的擬合集。所有題組均互斥，沒有用 test 結果挑控制量、超參數或做決策。去重涵蓋大小寫及空白差異，語意同題沒有自動比對。

## Probe 數配平

每臂先共享介入前 logits 與精確 singleton。signed margin 為預固定候選 token 的 logit 減去當下最佳替代 token 的 logit，目標位置始終保留為 mask。

- **Joint**。每個候選集合對每個目標揭露其餘 7 個位置。同一揭露輸入合併一次 forward，該輸入可同時對多個 masked targets 計分。令當前狀態去重後共有 `J` 個額外 forwards。
- **Doubleton**。從三個候選集合內實際可達的兩位置組合，挑選恰好 `J` 個不同揭露輸入。每步在候選集合／目標的未遮蔽配對上，以 `1/(1+已覆蓋次數)` 作權重，優先補足低覆蓋者，同分取較近位置後再按位置 ID。每次 forward 至少測到一個候選／目標；每一候選／目標均至少取得一筆。用 signed margin 的 doubleton 減兩個 singleton 加原始 margin 求二階增量；對每個目標以已測二階增量平均乘組內 `choose(7,2)`，加到 singleton 的加總預測。這是固定可及配對的外推值。
- 兩臂每題有相同數量、同長度、batch 1、同模型的完整 forwards，關閉模型 KV cache。兩個探針的 GPU 記錄順序按 doc_id 奇偶固定交換。報告的策略秒數為實測前綴、共享及各臂 probes、各臂探測計畫、CPU 排序與所選候選續跑相加的重組估計。兩臂分支分數從共同診斷記錄整理，沒有取得各臂串行部署時的總時間，秒數不作淘汰判準。

三個 learned 策略皆以同樣的擬合集與目標。對每同一 anchor 的正誤不同配對，用原候選次序取差，擬合無截距、L2=0.1 的 pairwise logistic regression。基準控制平均及最低 log-confidence、最左／平均／最右位置、相對 A 的 swap 距離、是否 swap，以及 singleton additive 集合分數。Doubleton／Joint 在共用特徵上各加一欄各自的集合分數，分數均為各目標 signed margin 的平均 log-sigmoid。特徵按訓練集加權 RMS 縮放，每題總權重相同；依驗證前鎖定的擬合結果獨立選擇測試候選，並列取原候選次序。金標只用於訓練和事後評估。第一輪聯合規則按 raw joint score 排序，另列為 current selector。此設計估的是此固定、非窮舉 doubleton／joint 預算的比較。

## 三項判準

最低覆蓋為 200 道 test 合格題。兩側 95% 區間下界達預定量記 `supported`，上界仍低於預定量記 `below_floor`，其餘 `inconclusive`。未達覆蓋條件記 `inconclusive`。

| 判準 | 估計量 | 下界要求 |
|---|---|---|
| 現象 | A 集合在單點加總及 joint LOO 之間至少一個目標跨過 `log(9)` 的合格 prompt 比例，Wilson 區間。另回報原狀態固定最佳替代 token 後，其 logit 差仍有至少 1 nat 高階殘差的跨線事件，以及動態最強替代 token 換位的比例。 | 15% 合格題 |
| 訊號 | 同 anchor 正誤不同配對，learned Joint 比 learned Doubleton 多正確排序的比例。同分按 0.5，每題內平均後，按 test 題成對 bootstrap 5,000 次。需至少 50 道 train、100 道 test 有正誤不同配對。 | +5 個百分點 |
| 改善空間 | 全 1,000 道 test 中，learned Doubleton 答錯而固定同池存在答對候選的題數占比，Wilson 區間。另列相對 confidence、第一輪聯合規則及 learned Joint 的同池 oracle gap。 | +4 個百分點 |

全測試答題率與配對差另報 learned Joint 對 learned Doubleton、兩臂各對原 confidence 規則。此外列配對救回、反向退步與 +4 pp 的參考目標。實際答題差提供後續研發依據，三項判準仍按上表獨立判讀。條件式排序改善可以出現在有限比例的正誤分歧題，無法直接換算全題增益。區間固定擬合政策，反映 test prompt 抽樣波動。若全題不一致率約 0.25，1,000 道的 accuracy 差雙側 95% 區間半寬約 3.1 pp，觀察差約須達 7.1 pp 才可證成下界 +4 pp；本設計未承諾分辨接近門檻的效應。

三項都支持時，推進等成本 joint 訊號研究。訊號低於門檻時，暫停此預算下的 joint 特徵，保留實驗其餘現象和空間的證據。改善空間低於門檻時，降低此固定候選池的優先順序。現象低於門檻時，先檢查切點及固定競爭 token 的機制量。有判據不確定時，後續驗證對準區間太寬、合格題覆蓋不足，或訓練／測試的正誤分歧題太少的來源。全題準確率差獨立報告供選擇後續方法。

## 產物

整段使用已凍結 checkpoint 和兩個已核對 parquet。2 題 smoke 只驗證程式與資源；正式 train/test 產物驗證精確樣本 ID、權重 SHA-256、資料和程式雜湊、固定 quota 及預算一致性。各作業從已凍結 commit 取出執行程式。依序啟動 smoke、train、test 及 CPU scorer，單 H200 的 train 和 test 上限分別為 2h、3h。CPU scorer 同時讀入完整 train、test；只用 train 擬合。 [執行說明](README.md)

## 讀取紀錄

- [codex] 2026-11-15 06:09 已讀 CPU 評分作業 `26443` 的 `results/verdict-26443.json` 機械彙總。凍結版沿用 `c979eba`。
- [codex] 2026-11-15 06:11 獨立子代理已讀同一機械彙總並核對凍結判準，未讀逐題收集檔。
