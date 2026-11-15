# [codex] 本機檢索數值重現診斷

審查時間：2026-11-16 04:43（台北）。本輪唯讀三份技術診斷 JSON 與 `parity_probe.py`；未執行 GPU、未讀方法成效、新全池結果或 sealed panel。兩份 GPU 探針均由 root 排程，凍結程式版本為 `4c5c20188e3db6f9aeef5e1f9ef77c4dd7f9eb52`。

## 已確認的證據

比較基準是本機 `dev-20261116-v2` 既有 CORE checkpoints，並非原 H200 的向量。原本機三份 checkpoint 的 SHA、manifest SHA、候選順序 fingerprint，在兩個 GPU 探針與先前 CPU 診斷中相符。

| 技術檢查 | 結果 |
| --- | --- |
| 同一既有向量，CPU BLAS 1 與 4 threads | 136 × 8,191 的分數及排名逐項相同；最大分數差 0；top-50 順序 SHA 相同 |
| PyTorch 2.6 重跑 16 圖、32 文字 | 與原本機 checkpoint 逐項相同；兩類向量最大差均為 0 |
| PyTorch 2.6 重跑原 top-50 前 8 pairs | 原與新 reranker scores 逐項相同，最大差 0 |
| PyTorch 2.14 重跑相同 16 圖 | 最大向量元素差 0.0045284231，最小逐列 cosine 0.999799996 |
| PyTorch 2.14 重跑相同 32 文字 | 最大向量元素差 0.0073189288，最小逐列 cosine 0.999780605 |
| 新文字向量 × **舊**全圖庫向量的技術排名 | 2.6：全同；2.14：29/32 條完整排名改變、14/32 條 top-50 集合改變，最大分數差 0.0040833354 |
| PyTorch 2.14 重跑原前 8 pairs | 最大 reranker score 差 0.125，平均絕對差 0.046875 |

兩次 GPU 探針使用同一張 RTX 3090（UUID `GPU-9a253dde-91d0-8c96-abe7-ddf4e3ee8a89`）、相同模型 revision、模型設定 SHA、原始來源與 freeze。Transformers 5.20.2、NumPy 2.3.5、Pillow 11.3.0 相同；兩者的 TF32 matmul 均關閉、matmul precision 均為 highest，BF16 reduced-precision reduction 均開啟。變更的是整套執行環境：PyTorch 2.6.0+cu124 / torchvision 0.21.0 / CUDA 12.4 / cuDNN 90100，對比 PyTorch 2.14.0+cu126 / torchvision 0.29.0 / CUDA 12.6 / cuDNN 91002。

首批 8 圖的五個輸入 tensors（含 `pixel_values`、`image_grid_thw`）與首批 32 文字的三個輸入 tensors，**shape、dtype、SHA 全數相同**。兩個 processor 的類別、設定與 torchvision backend 也相同。因此，已檢查的前處理輸入沒有觀察到差異；純文字輸出也改變，故影像前處理無法單獨解釋全部向量差異。

## 判讀界限

1. 目前證據支持「同一張 3090 上，兩套軟體執行環境產生不同數值」，尚未分離 PyTorch、CUDA、cuDNN、SDPA 或其他運算的個別原因，更未證明原 H200 與 3090 的硬體差異是歷史基線不合的原因。PyTorch 2.14 探針並未證明更接近原 H200 輸出。
2. 原批次被保留：image 8+8、text 32、cross 4+4；涵蓋固定前綴，不能外推整個圖庫。輸入 trace 是額外一次 CPU prepare 呼叫（`parity_probe.py:172–178,237–238`），未攔截實際 forward 的那次 tensors；第二批 8 圖及 reranker 的輸入沒有 trace。
3. 新文字 × 舊全圖庫是混合向量的技術排名，不是完整新模型的 Recall、AP 或 oracle。32 × 16 比較的 membership cutoff 為 16，取滿全部候選，故 membership 改變為 0 沒有辨識力；該矩陣在 2.14 下其實有 17/32 條排名改變（44 個 rank entries）。
4. 舊 checkpoint state 沒有存完成時的 array SHA，探針已重算並記錄現在的 SHA；這能綁定此次比較讀取的檔案，但不能補造歷史完整性收據。

## 下一步最小測試

root 已開始的 PyTorch 2.14 全池重算，應先按原規則機械檢查 CORE 候選 Recall / no-hit / fixed-50 oracle，再檢查完整 top-50 rerank 基線。兩者都維持原重現門檻；未通過時不能執行方法比較或額外 150 候選。Recall / no-hit / oracle 不需 cross scores，若可在新增 cross 前檢查，能更早停止無效重算。

若仍需定位數值來源，下一個小測試只需固定的一批 32 文字：攔截實際 forward 前的 CPU tensors，在同卡、同權重、同 dtype 下由兩環境直接重放相同輸入，各重複兩次以檢查不確定性。必要時再增加一批 8 圖與原 pair batch 4。這可進一步分離前處理和 inference stack；不依已讀主指標挑選 backend 或放寬容差。要確認是否恢復原 H200 行為，仍須原環境的同樣本輸出或原固定基線閘門提供證據。

## 本輪讀取來源

- `dl/retrieval3090/runs/dev-20261116-v2/parity-cpu-threads-20261116.json`（CPU 診斷建立於 04:25）
- `dl/retrieval3090/runs/parity-torch214-20261116/probe.json`（GPU 探針建立於 04:37）
- `dl/retrieval3090/runs/parity-torch26-20261116/probe.json`（GPU 探針建立於 04:41；本審查首次讀取 04:42:57）
- `codex/retrieval3090/parity_probe.py`

原本機 reference top-50 順序 SHA：`2bd4601ee313ce6a2b17ead967121225e48498706c7f3b34b13cd4aaff7b2310`；manifest SHA：`8f43eedbcef31fe5057c26705e97bf9f213f162612d9a15b53da11bb9bacf78d`。
