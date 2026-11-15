# 組合檢索的 anchor 排序拆解

本次是凍結作業完成後的 CPU 診斷，沿用 Half-Truths 三個子集。報告 60 的現象、訊號、空間判準及 26393 的機械判讀照原結果。所有實驗檔只讀。[補量預登記](../../killtest/PREREG_headroom.md) · [程式](audit.py) · [逐模型、子集、干擾類型及讀檔紀錄](results/anchor_audit.json)

## 結果

CS-CLIP 在 COCO(Qwen) 的 509 列有 296 列符合 `full > anchor > half`（58.15%）。470 列的 `full > half` 已經正確（92.34%）。這 470 列裡，78 列的 anchor 太高，96 列的 anchor 太低，兩種失敗互斥，沒有平手。另外 39 列的 full 與 half 已倒序。若事先知道每列答案並可逐列自由調整 anchor，最多可加回 174 列，即 34.18pp。

| CS-CLIP 整句 cosine | COR | `full > half` | 在 `full > half` 中 anchor 太高／太低 | 逐列 anchor 上限差 | 單一 offset 的同子集事後最佳 COR |
|---|---:|---:|---:|---:|---:|
| COCO(Qwen) | 58.15% | 92.34% | 78／96 | 34.18pp | 58.35% |
| COCO(Mistral) | 50.44% | 92.95% | 72／121 | 42.51pp | 51.54% |
| CC3M(Qwen) | 42.19% | 86.08% | 90／118 | 43.88pp | 43.25% |

COCO(Qwen) 的全樣本中，`anchor ≥ full` 有 96 列，`half ≥ anchor` 有 120 列，交集為 3 列。限制在端點排序正確的列後，交集為零。按 foil 類型拆解，`attribute_change` 的 59 列有 14 列端點順序錯。端點順序正確的 45 列中，又有 4 列 anchor 太高、17 列太低。`antonym` 的 162 列正確端點，則是太高 36、太低 32。完整四種 foil 及 entity／relation 分組在 JSON。

條件 PMI 整句版本在 CS-CLIP COCO(Qwen) 把 COR 降到 30.84%，`full > half` 降到 79.76%。在仍能排對端點的 406 列，anchor 太高／太低為 43／206。採 COCO 5,000 張影像做正規化，兩數變成 55／191，仍以 anchor 太低為主。這些量只描述原探針。

## 簡單解釋的持出檢查

把同模型的 anchor 分數全部加上一個常數。每次在另外兩個子集選常數，排除所有與評估子集共享影像 SHA-256 的校準列。對每個校準列，能修正排序的 offset 形成開區間 `(half − anchor, full − anchor)`。掃描區間端點間的中點可求得同一常數的精確最佳覆蓋數。CI 以目標子集的 image SHA-256 為群集配對重抽，呈現所選常數的持出差異。

| CS-CLIP 評估子集 | 持出 COR | 較原分數的變化與 95% CI | 修好／弄錯 |
|---|---:|---:|---:|
| COCO(Qwen) | 57.96% | −0.20pp [−1.18, +0.79] | 3／4 |
| COCO(Mistral) | 50.88% | +0.44pp [−0.66, +1.54] | 4／2 |
| CC3M(Qwen) | 42.62% | +0.42pp [−0.63, +1.48] | 4／2 |

同一常數即使准用目標子集標籤選擇，COCO(Qwen) 也只多排對一列。這支持 anchor 定位誤差因列而異。CLIP B/32 的持出 COCO(Qwen) 則從 18.86% 升至 22.79%，增益 3.93pp [0.59, 7.47]，其低 anchor 錯誤原本遠多於高 anchor 錯誤。這是看過既有資料後提出的診斷，CI 只反映本次固定 offset 在持出樣本的變異。

## 驗證與後續

- CLIP B/32 與 CS-CLIP 各三份向量檔的 row ID、image SHA、triplet type 與 foil type，均和固定 Parquet 的列序一致。整句 cosine 和邊際 PMI 的逐列 COR 旗標完全重現 Stage 1。整句 PMI 與 COCO pool 版本的 COR 及 `full > half` 重現 Probe 2，CS-CLIP 整句數值也重現 26393。CORE-EMBED、SigLIP2、CLIC 和 CORE-RERANKER 只保留彙總，將其端點／COR 差及總平手數另列，無法從現有輸出還原錯誤方向與 foil 子類。
- 後續確認可用新查詢與固定候選影像，在測試前凍結一個會隨 `(影像, 查詢)` 改變的證據分數。對每個查詢用固定規則抽與目標影像無關的 anchor，在開發資料決定權重，在未見查詢及影像上比較 CS-CLIP 與 CORE 的成對正確率及 Recall@50，並報告 k=1 到 4 的稀釋。只加上依文字決定的常數，對固定查詢的候選影像排序沒有影響。
- 26393 的 `B_fix=58.15%` 低於 75% 入池門檻，`R=30.45%` 未觸發 `R≥90%` 的限定。這次 CPU 診斷提供原排名錯誤的分解，不新增候選池門檻。

重現指令（從 repo 根目錄執行）

```bash
OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 PYTHONDONTWRITEBYTECODE=1 \
  ../icml/envs/vllm033/bin/python -I codex/retrieval-anchor-audit/audit.py
```
