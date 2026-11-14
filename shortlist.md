# 精選清單：準備跑停損實驗的題目（2026-11-14 晚）

依據：放寬後的條件（使用者原話整理）——不限方向、不要求沒人做過、不必贏 SOTA、要有研究價值、不能是純負面分析，目標信心中～中偏高。
資源：最多 4 張 H200（h200q，單一 8 卡節點與他人共用，需排隊）。第一梯表內的成本是初估，已執行階段的耗時與 VRAM 見 [硬體紀錄](codex/HARDWARE.md)。本輪 10 題的逐題進度與證據見 [現行實驗](codex/reports/2026-11-15-topic-status-codex.md)。
信心欄是深查或 Fable 校正後的評級；「自評」是輕量掃描回報自己給的，通常偏樂觀。

## 目前 GO 題目（2026-11-15，台北）

依[現象、訊號、空間的共同判準](reports/2026-11-14-60-phenomenon-signal-headroom-criteria.md)與各題凍結版複查。本區的 GO 表示通過候選池的三項門檻，可以深化研究。

| 題目 | 三項複查 | 下一個有辨識力的實驗 |
|---|---|---|
| 多條件檢索的單調證據聚合（`compositional-retrieval`） | **GO。** 適用於原同圖三句排序判準。[凍結判準](killtests/compositional-retrieval/PREREG_headroom.md) · [正式補量](killtests/compositional-retrieval/results/headroom/26393/verdict_headroom.json) | 新查詢的 anchor 校準未達固定候選庫的排名門檻。若續做，先查標註與錯排，並採 CORE-EMBED-2B 全句作較強基線。[正式複核](codex/reports/2026-11-15-three-frozen-readbacks-codex.md#多條件檢索) |

## 仍待判定的交接題目

| 題目 | 現象／訊號／空間 | 最值得先釐清的因素 |
|---|---|---|
| 兩個時鐘 | 正式評估完成。現象與放大訊號通過，提前時機未確立，G3 改善空間灰區。 | 先查每類樣本數正控制的第一格觸底。依預登記暫緩 CIFAR，等使用者決定是否入池。[正式複核](codex/reports/2026-11-15-three-frozen-readbacks-codex.md#兩個時鐘) |
| Λ | 限步長且等權的 v2 數值閘門通過。訓練後的現象、訊號、空間待測。 | 跨到真實訓練後，無防禦與 IP 能否重現，以及五個 arm 三個 seed 的漏出信度是否達 0.80。[v2 判讀](killtests/inoculation-lambda/VERDICT_phasea_v2.md) |
| 多保真度歸因 | 第 0 階段現象與同軌跡空間通過，訊號灰區。新查詢診斷指出十方向覆蓋偏低。 | 先驗證代表性更好的精確方向。跨種子正式補量仍受原門檻約束。[新查詢診斷](codex/reports/2026-11-15-topic-status-codex.md#多保真度資料歸因) · [第 0 階段](killtests/multifidelity-attribution/VERDICT_stage0.md) |
| Chameleon | 基準重現閘門未過，三項未能正式判讀。同 checkpoint 的 ENS8 幾乎飽和。 | 逐例核對作者探測器的資料、層索引與池化，隨後確認 ENS8 是否有可回收的失誤。[預登記](killtests/chameleon-defense/PREREG.md) |
| RL 稀疏蒸餾 | A 配對的重現與截斷門檻未過，B 尚未測。三項待測。 | 確認同預算、可比截斷的 pass@1／pass@64 交叉。先查 A 的既有輸出，再分開 B 的原作重現和短預算配對。[A 配對](killtests/rl-sparse-distill/results/aggregate_gate0_A.json) |
| 文生圖罕見失敗 | CPU 加速代理通過。CLIP 篩選未過，正式現象與訊號仍待測。 | 查篩選分數同分的成因。大批生成需要可信的篩選。[CPU 診斷](codex/reports/2026-11-15-topic-status-codex.md#文生圖罕見失敗) · [預登記](killtests/t2i-rare-failure/PREREG.md) |

## 第一梯：建議現在就跑停損（依優先順序）

| 優先 | 題目 | 信心 | 第一個停損點（成本） | 過關後 | 被搶先風險 | 回報 |
|---|---|---|---|---|---|---|
| 1 | **條件擴散模型偏差泛化的「兩個時鐘」**：條件分支依每類樣本數 n/K 提早進入偏差期，CFG 放大樣本特定成分；據此做條件模型專用的單模型偏差訊號與早停／guidance 規則 | 中 | 第 1 週 15–30 GPU-h：重現原作 CelebA 偏差曲線；階層資料用 BP 算條件／CFG oracle 曲線 | 第 2 週 55–85 GPU-h；第 3 週新訊號贏過 loss-gap／TSC 即升中偏高 | 低 | 53（原掃描 52 候選 2） |
| 2 | **inoculation 類防禦漏出的通用預測量 Λ**：防禦情境訓練梯度與非誘發情境特質 log-prob 梯度的內積，跨防禦類型排序漏出 | 中 | 第 1–3 天約 5 GPU-h，不訓練：用公開程式算 Λ，對照兩篇論文已報結果，至少一組 Spearman ≥ 0.6 | 第 4–12 天 60–80 GPU-h（約 84 次 LoRA）；判準過即升中偏高 | 高 | 54（原掃描 50 候選 A） |
| 3 | **文生圖罕見失敗的可校準機率估計**：雜訊空間 subset simulation／CE-IS，估暴力抽樣看不到的尾端，並處理偵測器可被對抗的問題 | 中 | 第 1 週約 30 GPU-h＋人工檢查：SD v1.4＋5 個抹除器，確認存在 10⁻⁵–10⁻³ 的罕見格且偵測器陽性為真 ≥ 60% | 第 2 週約 25 GPU-h；估計器在 2 倍內且省 10 倍樣本 | 高（2611.10106，11/06） | 55（原掃描 52 候選 1） |
| 4 | **多保真度資料歸因**：便宜歸因當 control variate，少量精確 metagradient 校正，加逐查詢可靠度估計 | 中 | 第 1 週約 45 GPU-h：CIFAR 上便宜方法 LDS ≥ 0.8 就停 | 第 2 週約 15–30 GPU-h（GPT-2） | 中 | 45 |
| 5 | **RL 稀疏決策的混合目標 on-policy 蒸餾**：pass@1／pass@k 前緣 | 中（Fable 校正中偏低） | 第 1 週推論為主：兩個公開 1.5B base／RL 配對上，≤10% 位置拿回 ≥70% pass@1 增益，且 pass@64 勝 RL | 第 2 週蒸餾須勝 WiSE-FT | 中偏高 | 44 |

## 第二梯：備案，或可問其他人要不要做

| 題目 | 信心 | 一句話 | 回報 |
|---|---|---|---|
| 組合泛化／屬性綁定 | 中 | 扣掉感知誤差後仍有 11–15 pp 只在未見組合出現的落差；因子化讀出＋子空間重組。原作程式資料未公開 | 47 |
| Drifting 小算力去偏 | 中 | drift 欄位的有限樣本偏差與記憶庫過期校正，主張同 GPU 小時下的效率 | 51 候選 1 |
| 合成影像能量分布與視覺預訓練 | 中偏低～中（自評中偏高） | NeurIPS 2026 oral 延後的等量預訓練因果驗證；原作團隊可能自己補 | 49 候選 1 |
| 合成資料選擇理論推廣到多類別與條件偏移 | 中偏低（自評中） | ICLR 2026 oral 點名的未來工作；需隨機矩陣工具 | 48 候選 1 |
| 多峰座標推理 RL | 中 | NeurIPS 2026 oral 的連續座標動作改多峰，修正混合動作 GRPO | 49 候選 2 |
| Neural Chameleons 防禦 | 中 | 利用逃避子空間做穩健激活監測器 | 49 候選 3 |
| PoPE 推廣到視覺 | 中 | ICML 2026 spotlight 的位置編碼推到 ViT／DiT 解析度外推 | 52 候選 3 |
| VLM「再看一次」真的重看 | 中 | VisualSwap 配對當可驗證獎勵做 GRPO | 52 候選 4 |

## 已淘汰（本輪）

擴散 LM 平行解碼量尺（43，已被做掉）、RLVR 稽核（42，2610.35716 已做）、Muon 與 warmup（飽和）、masked diffusion LM 記憶（核心已有人回答）。

## 排程建議

- 第一週同時送出第 1、2 名（合計約 20–35 GPU-h），第 3、4 名接著排；第 5 名有空卡再跑。
- Neon 語言版的試跑屬舊紀錄，已因磁碟配額中斷。〔[中斷紀錄](reports/2026-11-14-58-neon-lm-killtest-interrupted.md)〕
- 每題的停損判準在對應回報裡已事先寫定；跑之前由 Sonnet 實作、Opus 審查程式與判準，再送 slurm。
- 目標：11 月 28 日前依判準選定主線。
