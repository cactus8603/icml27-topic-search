# 大型提交集合的同狀態診斷

[Codex] 2026-11-14，台灣時間。凍結範圍為本目錄的程式與 `thresholds.json`。尚無主判準資料。

## 問題與實驗單位

在 LLaDA 同一次去噪、同一個已到達的狀態、同樣 8 次 token 寫入下，直接聯合探測能否捕捉單 token 影響加總漏掉的訊號，而且讓後續答題有改變的空間？對照已有的同狀態交換實驗與局部相依診斷。 [SetCommit 附錄 A/F](https://arxiv.org/html/2610.31120v1) · [Measured Coupling 附錄 J/K](https://arxiv.org/html/2611.08307v1)

- 使用公開 [LLaDA-8B-Instruct 固定快照](https://huggingface.co/GSAI-ML/LLaDA-8B-Instruct/tree/08b83a6feb34df1a6011b80c3c00c7563e963b07)；原版 bidirectional forward、BF16、固定 256 位置，循序處理 64-token blocks，每輪提交同 block 的 8 個最高 softmax confidence 的 argmax。全序列 32 次 forward，不因 EOS 提早停止。寫入不可逆。獨立計算探測全部 NFE 與秒數。
- GSM8K main/test 共 1,319 題。依 `thresholds.json` 的 seed，用 `SHA256(seed + ":" + doc_id)` 排序後取前 240 題；`doc_id` 為 parquet 原順序的 0-based 列號。以預先指定角色模板的零樣本 prompting，最後一行 `#### <數字>`。讀最後一個可辨識的同格式行，按十進位數值比對參考答案；無格式則得分 0。先在第一個 EOS／EOT 截斷。每題的樣本權重相同。
- 在原 confidence trajectory 的第 2、10 輪（0-based，各為前兩個 blocks 的第 3 輪）保留同一狀態。若 top-8 原候選有特殊 token，或末四個入選位置與接下來四個未入選位置間無法形成兩個普通 token 的交換，記錄不合格原因。每個交換按 confidence 差、位置距離、位置 ID 依序排序，最多取兩個；log-prob 折損最多 0.25 nat、替換位置距離最多 24。每個集合仍有 8 個普通 token，所有集合中的 token 值來自分支開始前同一次 forward 的 argmax。
- 每個 eligible anchor 的三個集合皆跑同 checkpoint、同 quota、同貪婪後續規則，到相同總 forward 上限。A 是原始 confidence，B 是第一個匹配交換，另外一次匹配交換增加後設候選覆蓋。C 依精確 singleton additive 排序，D 依精確 joint leave-one-out 排序，兩者共用三個候選集合及同一 set score。若同分保留候選次序。A 沿用原 trajectory，其他集合各自複製介入狀態續跑。C、D 結果取其選中集合的已配對續跑。選擇函式只接觸模型輸出和位置；原始 rollout 完成後才解析正解。
- 用固定目標 token 的 logit 減去該狀態下最佳替代 token 的 logit 得到 signed margin。C 的預測為原始 margin 加各其他入選位置單獨揭露時的 margin 差。D 對每一個候選集合中的目標位置，一次揭露集合中其他 7 個候選位置，目標保持 mask，再直接量其 margin。C、D 的集合分數都為平均 log-sigmoid(margin)。這是固定 pool 的完整 forward 探針，Measured Coupling 的 fused、線上閾值選擇與 SetCommit 的完整解碼器均未納入。

## 獨立判準

各判準回報估計與雙側 95% 區間。區間下界達最低效果記為 `supported`，上界低於最低效果記為 `below_floor`，其餘記為 `inconclusive`。若合格題少於 60，三項一律 `inconclusive`。D 對 C／A／B 的 paired accuracy 為描述量，三判準由獨立審查後讓使用者決定下一步。

| 判準 | 主量測 | 預先設定 |
|---|---|---|
| 現象 | 合格題中，有任何原 A 集合中已提名 token 在加總與直接 LOO 的信心門檻兩側。每題最多算一次，兩個固定 anchors 合併。另描述 token-level 翻轉和 D 的 argmax 推翻率。 | margin 門檻 log(9)，對應 margin proxy 的 0.90，最低 15% 合格題，Wilson 區間。 |
| 訊號 | 同一 anchor 中正誤不同的候選配對，比較兩個保留整題的預測器。基準使用低成本訊號及 C 分數，擴充只加 D 分數，主量為 held-out 排序正確率增益。 | 至少 25 題有正誤不同的配對，最低增益 5 pp，4,000 次按題成對重抽。 |
| 空間 | 原 A 答錯而任一 eligible anchor 的任一替代集合答對的題數，占全部 240 題。這是有限候選的事後可救量，正解只在此處和一般評分使用。 | 最低 4 pp 全部題，Wilson 區間。若低於門檻，只涵蓋這組 prompts、quota、anchors、pool。 |

按 doc_id 及設定檔固定 seed 的 SHA-256 取餘數分成五 folds。同題兩個 anchors 及所有配對分在同一 fold。正誤不同的配對依原候選順序取特徵差，目標為前者是否答對。每個候選的控制量為平均和最小 log-confidence、最左／平均／最右位置、相對原集合的替換距離（原集合為零）、是否交換，以及 C additive score。擴充模型使用相同控制量，並加 D joint score。位置與信心一律來自介入前的狀態及候選值。

兩模型均為無截距 logistic regression，使用固定 L2 係數 0.1、訓練 folds 的各特徵加權 RMS（低於 `1e-8` 時採 1）縮放、不中心化。訓練時每題總權重相同。保留 fold 的每個配對計排序正確，零分（絕對值至多 `1e-12`）算 0.5，先在題內平均，兩模型的題級差以固定的 out-of-fold 預測按題成對 bootstrap。區間衡量評估題抽樣波動。原 D／C 分數直接排序與第一個 eligible anchor 的 D−C 答題正誤均作描述。

一般性診斷另外記下 EOS 截斷、續跑步數、候選選擇差異與 probe 時間。精確探測本身的成本尚未攤入任何 selector 的部署延遲比較。此試驗為固定 A trajectory 上一次介入，沒有由此推斷逐輪介入或跨任務效果。缺乏信心門檻所需樣本時保留不確定性。

## 執行上限

Smoke 固定取 train parquet 前兩題，使用同一模型及解碼設定。只跑程式檢查和計時，不解析答案、不留結果；兩題內至少要走過一個 eligible anchor 的真模型探測。主跑單 H200、240 道 test，Slurm 上限六小時。如有中斷只留下 `.partial`，判讀入口要求完整資料。GPU 送出與 freeze 依共享協作規則記載。
