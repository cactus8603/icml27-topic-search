# ICML 2027 選題與實驗

2026-11-15（台北）。目前有 **10 題已建立實驗**，其中 **1 題 GO、6 題待判、3 題已完成先導並送出第二輪**。另有 [7 題備案](codex/reports/2026-11-15-topic-status-codex.md#備案與舊題)。

## 目前題目

| 題目 | 進度 | 現象要確認什麼 | 訊號與改善空間要量什麼 |
|---|---|---|---|
| [多條件檢索的單調證據聚合](codex/reports/2026-11-15-topic-status-codex.md#多條件檢索) | **GO**，新排名試驗已送出 | 完整、基準、半真描述是否排錯 | 完整描述能否穩定勝過半真描述，以及調整基準分數能否改善固定候選圖的排名 |
| [條件擴散的兩個時鐘](codex/reports/2026-11-15-topic-status-codex.md#條件擴散的兩個時鐘) | 第 2 週正式評估已送出 | 條件模型是否先進入樣本偏差期 | 用負控制辨別 CFG 是否提早或放大偏差，以精確分數量差距 |
| [Inoculation 防禦漏出預測 Λ](codex/reports/2026-11-15-topic-status-codex.md#inoculation-防禦漏出預測-λ) | 數值檢查後，待訓練驗證 | 不同防禦設定下的漏出是否有可重現差異 | 初始梯度內積能否排序漏出，現有預測量距信度上限多遠 |
| [多保真度資料歸因](codex/reports/2026-11-15-topic-status-codex.md#多保真度資料歸因) | 階段 0 訊號待釐清，新查詢已送出 | 廉價歸因能否預測重訓後的效果 | 少量精確歸因能否補回廉價方法的差距，同軌跡與跨種子各量一次 |
| [Neural Chameleons 防禦](codex/reports/2026-11-15-topic-status-codex.md#neural-chameleons-防禦) | 待核對作者基準重現 | 觸發條件是否降低單層監測器的檢出率 | 跨概念的訊號是否能補回多層集成漏掉的案例 |
| [RL 稀疏決策蒸餾](codex/reports/2026-11-15-topic-status-codex.md#rl-稀疏決策蒸餾) | 待建立可比配對 | RL 能否提高單次答對率，同時降低多次取樣的覆蓋率 | 少量 token 決策能否改善 pass@1／pass@64，及其相對權重混合的空間 |
| [文生圖罕見失敗機率估計](codex/reports/2026-11-15-topic-status-codex.md#文生圖罕見失敗) | 待驗證篩選器與 CPU 代理 | 固定條件下是否有可量測的罕見失敗 | 篩選器陽性有多少是真陽性，以及尾端估計在同誤差下能省多少抽樣 |
| [大型平行解碼提交集合](codex/reports/2026-11-15-topic-status-codex.md#大型平行解碼提交集合) | 先導完成，第二輪已送出 | 同輪提交多個 token 時是否出現可觀測的交互作用 | 等成本比較聯合探測與兩兩探測的排序，以及同候選池的可改善上限 |
| [強驗證器與少量金標](codex/reports/2026-11-15-topic-status-codex.md#強驗證器與稀疏金標) | 先導完成，第二輪已送出 | 強弱驗證器是否有大量共同誤收 | 強判決能補多少資訊，維持相同風險時能否省下金標費用 |
| [Attention gate × optimizer](codex/reports/2026-11-15-topic-status-codex.md#attention-gate--optimizer) | 先導完成，第二輪壓力校準已送出 | 更新是否造成可重現的單步損傷 | 注意力寫入能否預測損傷，同一 checkpoint 有多少可回收空間 |

[逐題進度、結果與下一步](codex/reports/2026-11-15-topic-status-codex.md) · [共同判準](reports/2026-11-14-60-phenomenon-signal-headroom-criteria.md) · [Slurm 提交紀錄](killtests/SUBMITTED.log)

## 硬體與執行

目前有 GPU 的實驗各以 **1 張 H200** 送出，部分評估與先導重播可用 CPU。逐題硬體表列出實際量到的 **GPU 峰值 VRAM**、Slurm 申請的 **主機 RAM**、執行時間與磁碟用量，也標出尚未跑完的階段。框架預留的 VRAM 會計入取樣峰值，各配置的最低可用卡型仍需實測。

[逐題硬體、VRAM 與磁碟](codex/HARDWARE.md)

## 實驗與歸檔

- 各題的 `PREREG.md` 記錄凍結的判準，GPU 作業在 `killtests/SUBMITTED.log` 連到對應提交。已判讀的證據可從逐題進度頁找到。
- 模型、checkpoint、資料下載、大型中間輸出留在 H200。GitHub 保存程式、預登記、排程與小型結果。〔[協作規則](COORDINATION.md)〕
- 本輪以前的篩選、取捨與交接可查閱歷史紀錄。〔[候選總表](candidates.md) · [決策紀錄](log.md) · [早期報告索引](reports/INDEX.md) · [交接報告](reports/2026-11-14-61-claude-handoff-to-codex.md) · [初版資料來源與操作說明](https://github.com/cactus8603/icml27-topic-search/blob/6d8e1f6ba597dd0c758e13bc408a35b9cd4891cd/README.md#資料來源)〕
- 舊 ICML 2027 總覽保留 15 題的歷史快照與原 repo 連結，Neon 語言版也保留試跑紀錄。〔[舊題歸檔](https://github.com/s990093/ICML2027/tree/5a7d66f531710f6397882cd743832066edfa70cc/old-topics) · [Neon 中斷紀錄](reports/2026-11-14-58-neon-lm-killtest-interrupted.md)〕
