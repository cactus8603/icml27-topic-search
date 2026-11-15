# ICML 2027 選題與實驗

2026-11-15（台北）。目前有 **10 題已建立實驗**，其中 **1 題 GO、6 題待判、3 題已完成先導並送出第二輪**。另有 [7 題備案](codex/reports/2026-11-15-topic-status-codex.md#備案與舊題)。

## 目前題目

前七題沿用 Claude 的前期工作，Codex 依交接紀錄跟進。後三題由 Codex 發展具體實驗、實作先導並送出第二輪。[Claude 交接](reports/2026-11-14-61-claude-handoff-to-codex.md) · [Codex 三題深查](codex/reports/2026-11-14-rebuttal-topic-deepcheck.md)

| 題目 | 前期分工 | 目前接手 | 下一步 |
|---|---|---|---|
| [多條件檢索](#多條件檢索) | Claude 做第一階段與第二個探針 | Codex 完成 anchor 對照、新庫排序及前 50 名上限診斷 | `26568` 用新留出池比較實際重排器與原 CORE 整句排名 |
| [條件擴散的兩個時鐘](#條件擴散的兩個時鐘) | Claude 做第一週訓練與評估 | Codex 完成正式評估與三臂共同網格 | `26559` 固定母體、條件策略與更新量，隔離新資料的唯一本數 |
| [Inoculation 防禦漏出預測 Λ](#inoculation-防禦漏出預測-λ) | Claude 做初版 Phase A；Codex 在交接前做小步長診斷 | Codex 備妥作者環境、離線複核和 GCD 信度程式 | `26562`／`26563` 重跑作者訓練，通過後生成並準備盲化與作者評分材料 |
| [多保真度資料歸因](#多保真度資料歸因) | Claude 建立第 0 階段訓練、重訓與精確歸因流程 | Codex 完成精確診斷及新方向對照 | `26535` 用獨立訓練種子重測最佳便宜方法與精確參照的排序落差 |
| [Neural Chameleons 防禦](#neural-chameleons-防禦) | Claude 實作判準、完成第一次激活抽取 | Codex 核對乾淨基準與作者訓練入口 | 核對作者歷史探測器、初始化紀錄與基座版本 |
| [RL 稀疏決策蒸餾](#rl-稀疏決策蒸餾) | Claude 實作判準並送出配對 A | Codex 完成 B 的 16K 重現、新題模板對照與配對 A 同輸出重評 | 核對生成規格和候補權重的原作依據，短預算資格暫緩 |
| [文生圖罕見失敗](#文生圖罕見失敗) | Claude 實作判準、吞吐測試與篩選閘門 | Codex 完成尾端代理與 CLIP 新種子重驗 | 定案是否採用已核對的修復憑證，再評估正式生成與盲標 |
| [大型平行解碼提交集合](#大型平行解碼提交集合) | Codex 深查並實作 LLaDA 先導 | Codex 完成第二輪與零額外 forward 的新題測試 | 暫停此版規則，保留固定候選池上限和成本作參照 |
| [強驗證器與少量金標](#強驗證器與少量金標) | Codex 深查並用公開資料做 CPU 重播 | Codex 凍結官方 GSM8K 獨立留出及開發診斷 | `26565` 重跑真實開發候選的格式與排序診斷 |
| [Attention gate × optimizer](#attention-gate--optimizer) | Codex 深查並實作強基線先導 | Codex 完成 12 個保留種子的壓力試驗 | `26521`／`26522` 用 48 個全新種子重驗訊號 |

[逐題進度與結果](codex/reports/2026-11-15-topic-status-codex.md) · [兩個時鐘、平行解碼、檢索正式複核](codex/reports/2026-11-15-three-frozen-readbacks-codex.md) · [共同判準](reports/2026-11-14-60-phenomenon-signal-headroom-criteria.md) · [Slurm 提交紀錄](killtests/SUBMITTED.log)

## 各題的現象、訊號與比較上限

各題的上限可能是已知真值、事後選擇的最佳表現、經驗前緣或局部代理。下列說明各自成立的條件，量測值與判讀放在逐題報告。

### 多條件檢索

- **現象**看多條件圖片檢索能否把完整、基準、半真三種描述排對。
- **訊號**看完整描述能否勝過半真描述，再檢查基準描述的分數偏高或偏低。
- **現有方法**比較 CS-CLIP、CORE-EMBED-2B、SigLIP2、CLIC 的固定索引分數，另以 CORE-RERANKER-2B 作跨編碼比較。
- **上限**同圖比較可事後調整基準描述分數，最高三句排序率等於完整勝半真的比例。跨圖檢索另用已標註的相關圖設定前 50 名內的理想順序，固定候選成員和名單外排序。[逐題證據](codex/reports/2026-11-15-topic-status-codex.md#多條件檢索) · [新庫預登記](codex/experiments/retrieval-conjunction/PREREG.md)

### 條件擴散的兩個時鐘

- **現象**看不同訓練樣本得到的條件模型，是否在測試損失最低之前就開始反映各自的樣本偏差。
- **訊號**用每類樣本數正控制和單類別負控制，區分條件樣本變少與 CFG 提早或放大偏差的作用。
- **現有方法**比較同一模型的無引導與 CFG 分數，並用配對控制對齊資料與隨機種子。
- **上限參考**由 belief propagation 算出階層合成資料的精確分數，量現有 CFG 對訓練和留出樣本的誤差差距。[逐題證據](codex/reports/2026-11-15-topic-status-codex.md#條件擴散的兩個時鐘) · [預登記](killtests/biased-gen-cfg/PREREG.md)

### Inoculation 防禦漏出預測 Λ

- **現象**看不同防禦設定下的特質漏出差異，能否在重訓與不同種子間穩定重現。
- **訊號**看訓練開始前的梯度內積 Λ 能否排序訓練後漏出，並檢查移除任一防禦家族後相關是否仍在。
- **現有方法**比較 Grant 梯度餘弦、Wichers 誘發強度、Tan 損失降幅與 PHM persona 距離。
- **上限**用 Λ 和漏出在不同種子間的信度估計可達相關，與最高的既有預測量比較。覆蓋不足時依預登記另作判讀。[逐題證據](codex/reports/2026-11-15-topic-status-codex.md#inoculation-防禦漏出預測-λ) · [Phase B 預登記](killtests/inoculation-lambda/PREREG.md)

### 多保真度資料歸因

- **現象**看便宜的歸因分數在多大程度上能預測移除資料後的實際重訓結果。
- **訊號**量便宜分數在已知精確方向之外，能否跟上精確歸因的反應。
- **現有方法**比較梯度內積、TracIn、TRAK、EK-FAC 和相同精確方向預算的 Spell 式延伸。
- **上限參考**是同一訓練軌跡的完整精確歸因，以實際移除重訓的排序驗證。跨種子要另量。[逐題證據](codex/reports/2026-11-15-topic-status-codex.md#多保真度資料歸因) · [新查詢協定](killtests/multifidelity-attribution/exact_probe/PROTOCOL.md)

### Neural Chameleons 防禦

- **現象**先重現作者的單層監測器，再檢查固定觸發條件下的檢出率落差。
- **訊號**檢查跨概念的留一正控制，或穩健監測器能否比多層集成找回漏檢。
- **現有方法**同一公開 layer-12 checkpoint 上比較作者單層探測器、八層集成 ENS8 與穩健監測器。
- **上限參考**是 ENS8 在乾淨條件下的檢出率。閾值由乾淨負例設定，固定後量觸發條件下的檢出率落差。[逐題證據](codex/reports/2026-11-15-topic-status-codex.md#neural-chameleons-防禦) · [預登記](killtests/chameleon-defense/PREREG.md)

### RL 稀疏決策蒸餾

- **現象**看同預算、截斷率相近時，RL 模型的單次答對率和 base 模型的多次取樣覆蓋率是否各有優勢。
- **訊號**測少量 token 決策替換能否保留 RL 的單次增益，同時改善多次取樣成果。
- **現有方法**以 base、同引擎的 RL 和 WiSE-FT 權重內插為對照，另量回應層級混合。
- **空間參考**是在相同 pass@1 下，稀疏替換或回應混合可達的 pass@64，與 WiSE-FT 實測前緣比較。估計只涵蓋可以內插的共同範圍。[逐題證據](codex/reports/2026-11-15-topic-status-codex.md#rl-稀疏決策蒸餾) · [預登記](killtests/rl-sparse-distill/PREREG.md)

### 文生圖罕見失敗

- **現象**看固定模型、提示與事件定義下，失敗率能否被穩定量到。
- **訊號**用盲標確認稀有陽性有多少是真的，並檢查篩選器的正負控制。
- **現有方法**以直接 Monte Carlo 抽樣為基準，和 subset simulation 等尾端估計方法比較。
- **上限參考**用已知機率的高斯半空間，估計達到同樣相對誤差時能省多少抽樣。真實生成的加速幅度仍待量測。[逐題證據](codex/reports/2026-11-15-topic-status-codex.md#文生圖罕見失敗) · [預登記](killtests/t2i-rare-failure/PREREG.md)

### 大型平行解碼提交集合

- **現象**固定同一解碼狀態與提交數，檢查單點加總與整組探測是否對同一 token 給出落在信心門檻兩側的分數。
- **訊號**在留出題量聯合探測是否比等次數的兩兩探測更常排對候選。
- **現有方法**沿用原 confidence 選擇、單點加總和等額 forward 的已擬合兩兩探測作對照。
- **上限**固定同一狀態和候選池，事後選能答對的候選。可回收空間以兩兩方法答錯、池內至少有一個答對的測試題計算。[逐題證據](codex/reports/2026-11-15-topic-status-codex.md#大型平行解碼提交集合) · [第二輪預登記](codex/experiments/large-commit-sets/round2/PREREG.md)

### 強驗證器與少量金標

- **現象**量強驗證器誤收錯解的比例，以及多少也是弱驗證器共同誤收。
- **訊號**在弱判決、資料集與作答模型已知時，檢查加入強判決還能補多少預測資訊。
- **現有方法**在相同暖機預算下，比較只買專家金標與混買強判決、金標的政策。
- **上限參考**用完整暖機金標訓練有無強判決的兩類政策，在同一鎖定測試集比較等風險的部署成本。這個參考受固定政策類別限制。[逐題證據](codex/reports/2026-11-15-topic-status-codex.md#強驗證器與稀疏金標) · [第二輪預登記](codex/experiments/fallible-verifier/round2/PREREG.md)

### Attention gate × optimizer

- **現象**看競爭力較強的 optimizer 在壓力下，是否出現可重現的單步驗證損失上升。
- **訊號**在 QK、激活、梯度和更新幅度等診斷已知時，量 attention 寫入變化能否改善對損傷的預測。
- **現有方法**比較 StableAdamW、Muon、AdamW 加 sandwich RMSNorm 的配對正常學習率分支。局部介入另和同量縮步、正交調整比較。
- **上限**同一 checkpoint 回退整步至更新前損失，界定這一步最多能回收的損傷。持續訓練的時間收益仍待量測。[逐題證據](codex/reports/2026-11-15-topic-status-codex.md#attention-gate--optimizer) · [第二輪預登記](codex/experiments/gate-optimizer/round2/PREREG.md)

## 硬體與執行

目前有 GPU 的實驗各以 **1 張 H200** 送出，部分評估與先導重播可用 CPU。逐題硬體表列出實際量到的 **GPU 峰值 VRAM**、Slurm 申請的 **主機 RAM**、執行時間與磁碟用量，也標出尚未跑完的階段。框架預留的 VRAM 會計入取樣峰值，各配置的最低可用卡型仍需實測。

[逐題硬體、VRAM 與磁碟](codex/HARDWARE.md) · [24 GB 與 H200 安排](codex/HARDWARE.md#下一步用-24-gb-還是-h200)

## 實驗與歸檔

- 各題的 `PREREG.md` 記錄凍結的判準，GPU 作業在 `killtests/SUBMITTED.log` 連到對應提交。已判讀的證據可從逐題進度頁找到。
- 模型、checkpoint、資料下載、大型中間輸出留在 H200。GitHub 保存程式、預登記、排程與小型結果。〔[協作規則](COORDINATION.md)〕
- 本輪以前的篩選、取捨與交接可查閱歷史紀錄。〔[候選總表](candidates.md) · [決策紀錄](log.md) · [早期報告索引](reports/INDEX.md) · [交接報告](reports/2026-11-14-61-claude-handoff-to-codex.md) · [初版資料來源與操作說明](https://github.com/cactus8603/icml27-topic-search/blob/6d8e1f6ba597dd0c758e13bc408a35b9cd4891cd/README.md#資料來源)〕
- 舊 ICML 2027 總覽保留 15 題的歷史快照與原 repo 連結，Neon 語言版也保留試跑紀錄。〔[舊題歸檔](https://github.com/s990093/ICML2027/tree/5a7d66f531710f6397882cd743832066edfa70cc/old-topics) · [Neon 中斷紀錄](reports/2026-11-14-58-neon-lm-killtest-interrupted.md)〕
