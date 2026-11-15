# ASDiv-A 同絕對點風險的事後策略省費上限

在同一封存 ASDiv-A 面板比較直接決策樹和可買強判決的決策樹。兩者的保守絕對誤收與誤拒點上界各限 11%，採相同的名目暖機攤銷 0.135，付費強判決與專家的價格比沿用 1:10。每個暖機已見 `(資料集, 作答槽位, 弱判決)` 可用原有限決策樹的隨機混合。沒有弱判決或暖機未見的狀態直接轉專家。付費強判決不可用時再付專家費。未知金標留在成本分母與保守風險計算。 [原 ASDiv-A 協議](../asdiv-replication/PREREG.md) · [固定決策樹與原始／對偶檢查](../empirical-policy-ceiling/PREREG.md)

固定每筆都交給無誤的專家作為零風險參照，成本含相同的名目暖機攤銷。原線性規劃介面中的配對容許設為 0.5，任何通過 11% 絕對風險的解均自動滿足。若某一真類已知筆數為 `n`，已知錯誤機率總和為 `e`，未知金標的錯誤機率總和為 `u`，絕對風險約束等價於 `e + 0.89u ≤ 0.11n`。因為 `e,u ≥ 0`，相對無誤專家的保守配對增加 `(e+u)/n` 至多為 `0.11/0.89 < 0.5`。回報兩個個別最小成本及兩者差額，並用直接樹與全專家各作分母表示增量。比較採用相同的未知真值處理與同一個原始／對偶驗證。

固定來源為正式作業 `26707`，程式凍結 `885ac701b9b4ed313c3ffb7d6619b70953f0422f`，原摘要 SHA-256 `e12d6896b08d2cc563d2ee11a4aac20b860e1acb0365b1846d7e6ea8ec9b8d9c`，組裝清冊 SHA-256 `86d86ac5658db34f07f47554612f7d00bfaac3737b801b8542a6bdf8ee1ae496`。用原摘要核對候選和獨立金標指紋、四份評分收據及資料覆蓋，重建原比較的成本和點風險並列出違反的條件。兩種最優策略直接用留出金標求解。結果只適用於這個面板和固定的二元決策樹，屬於事後點上限。正式 20% 省費門檻及信賴界仍由原分析判定。

- [codex] 2026-11-15 16:04（台北）。本規格制定前已讀正式作業 `26707/summary.json`，讀取已登記於原協議。 [原讀取紀錄](../asdiv-replication/PREREG.md#讀取紀錄)

預檢只執行合成測試與設定驗證。正式分析先查核來源作業成功。Slurm 版使用一個 CPU 核、4 GiB 記憶體、分析程序上限 120 秒、作業上限 5 分鐘。本機備援用相同封存程式，限單核與 30 秒。兩入口各寫一份 SHA-256 可核對的彙總 JSON，程式標準輸出只包含作業及檔案身分。依協作規則登記新摘要的讀取，再交獨立複核。

```bash
bash codex/experiments/fallible-verifier/asdiv-point-risk-ceiling/run.sbatch FULL_FREEZE_COMMIT --check-only
sbatch codex/experiments/fallible-verifier/asdiv-point-risk-ceiling/run.sbatch FULL_FREEZE_COMMIT
bash codex/experiments/fallible-verifier/asdiv-point-risk-ceiling/run.sbatch FULL_FREEZE_COMMIT --local
```

## 讀取紀錄

- [codex] 2026-11-15 16:17（台北）。核對 `26712` 為 `COMPLETED / 0:0`，摘要 `runs/fallible-verifier/asdiv-point-risk-ceiling/26712/summary.json` 的 SHA-256 為 `34eded9d78b587b55c39dae74569264d54201ac1ac811dd2d419c670f2523a85`，與執行收據、凍結版本 `580772dcb825f11af836b4f3d966d4c444ba0379`、正式來源及面板摘要吻合。直接與雙端的每候選最小成本為 0.83015134 與 0.72045133，相對直接最優額外省 13.2145%。兩種解的保守絕對誤收與誤拒點風險均達 11%，原始與對偶成本差小於 `7e-15`。Astra 獨立核對比例與證據範圍。Slurm 實測 5 秒、MaxRSS 約 76.93 MiB，CPU 無 GPU，摘要 5,363 bytes。
