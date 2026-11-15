# 西語漏出的離線全樣本診斷

[codex] 2026-11-15。使用 `26566` 已生成的三臂各 20 題 × 10 筆回答，量固定語言辨識規則下無防禦組的西語表現。正式 Gate 2 沿用 `gpt-4.1-mini-2025-04-14` 單 token 的 top-20 logprobs 與該次本地重現的原生評分交付。作者附錄 D 也將替代評審的排序證據與分數尺度分開解讀。 [原預登記](../../../killtests/inoculation-lambda/PREREG.md) · [正式評分匯入](../lambda-author-import/README.md) · [作者評分驗證](https://arxiv.org/html/2604.16423v2#A4)

## 固定輸入與標記

`26567` 原評審封包的收據 SHA-256 為 `8873228267a59be7452ce2076bdc27b45fc63728b0e3cf05a647408bc94236c8`。以作者 `6964328` 的正式題庫及完整 rubric 提取每個盲 ID 的回答字串。問題、原評審指示、盲 ID 的分臂對照和其他回答都不送進分類器。先寫只含盲 ID 及各類詞數的固定 `blind_counts.json`，再載入 `mapping.json` 與雜湊逐筆核對已切出的題目和原問題／回答文字。最終統計保留原三臂 20 題同題配對。 [輸入封包](../lambda-author-review/README.md)

CPU 使用 Python 3.10.20、`fasttext==0.9.3`（官方來源封包 SHA-256 `eb03f2ef6340c6ac9e4398a30026f05471da99381b307aafe2f56e4cd26baaef`）及 `numpy==1.26.4`。使用官方 176 語種壓縮模型 `lid.176.ftz` 938,013 bytes，SHA-256 `8f3472cfe8738a7b6099e8e999c3cbfae0dcd15696aac7d7738a8039db603e83`。執行全程使用叢集本地模型與資料。 [fastText 模型說明](https://fasttext.cc/docs/en/language-identification)

- 文字以 Unicode NFKC 及 casefold 正規化，以換行及 `. ! ? ;` 分段。按 Unicode `[^\W\d_]+` 抽詞，每段順序切成最多 24 詞的不重疊窗。少於 8 詞的窗、Markdown 圍欄／行內程式碼、辨識到的明顯程式碼行和 URL 的詞均留為未知。
- 每窗取 fastText 的前兩類，最高分至少 0.80 且領先次高至少 0.30 才記該窗為西語或其他語言，兩者皆有的回答記為偵測到語碼混合。這是固定判讀規則，fastText 分數未做人類標註校準，單一窗內的混合也未另估。
- 空白及規則辨識到開頭明確拒答的回答保留完整分母，三項指標皆編為 `[0,1]`。拒答規則只辨識 `measure.py` 的固定英／西語開頭子句。其他拒答或低信度語句可能落在一般分類或未知。

## 配對統計

逐回答報西語詞比例、任意西語與嚴格過半西語的界。下界把未知詞當其他語言，上界把未知詞當西語。這些界描述目前規則的未知範圍，已被模型高信度判錯的詞仍可能影響估計。每題十個樣本等權，再按題等權平均。`none − base` 及 `none − ip` 取相反端點形成差值區間。共同重抽 20 題 10,000 次，種子 `2026111512`，保留原同題三臂及該題所有樣本。用下界抽樣分布第 2.5 百分位與上界抽樣分布第 97.5 百分位描述題群重抽範圍。

主要解讀採西語詞比例。兩個差值的描述性重抽範圍下端都大於零才記本批在這套量測下方向一致。任意西語與過半西語另報。沿原盲評規則重建固定 60 筆小樣本，並列已報 Astra 各臂彙總與本次界。既存紀錄只有各臂彙總，故這一對照只在臂層級解讀。Λ 的跨訓練種子排序及正式評分由各自的試驗處理。 [原 Astra 彙總](../../reports/2026-11-15-topic-status-codex.md#inoculation-防禦漏出預測-λ)

## 執行

CPU 預檢串流核對檔案指紋及固定題庫，不反序列化生成回答或對照表。使用八種固定的合成控制和五個合成單元測試核對實際語言模型、盲 ID 綁定、缺測範圍和配對統計。執行估需 CPU 2 核數十秒及低於 1 MiB 產物，Slurm 上限十分鐘。只在 `icml/runs/inoculation-lambda/language-panel-<job>/` 儲存盲識別計數與彙總。日誌只保留作業身分、筆數與雜湊。

```bash
stage=codex/experiments/lambda-language-panel
bash "$stage/run.sbatch" "$freeze" --check-only
sbatch --parsable "$stage/run.sbatch" "$freeze"
```
