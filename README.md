# ICML 2027 選題與實驗

我們用小規模實驗確認研究現象、可用訊號，以及相較現有方法的改善空間。點進題目資料夾可看介紹、設計、結果、程式，以及資料集、環境和執行方式。

本 repo 是**中樞**：放協作規則、選題、GPU 送件紀錄與跨題目報告。進行中與候選題目都有題目 repo（`icml27-<題目>`），**題目 repo 是正本**；本 repo 對應的 `topics/`、`codex/retrieval*/`、`paper/` 只是封存。暫緩與淘汰的題目不建 repo，正本留在本 repo 的 `topics/`。本機 checkout 放在 `/ssd4/cactus8603/icml27-repos/`。規則見 [協作規則第 0 節](COORDINATION.md#0-repo-結構中樞與題目-repo)。

## 進度總表

[claude] 2026-11-17 更新。內容取自各題 repo 的 `VERDICT.md`（正本）和 Slurm 佇列。判準是報告 60 的「現象、訊號、改善空間」三項。符號說明：✅ 已成立；◐ 部分成立或未定；⏳ 量測中；— 尚未量。

| 題目 | 現象 | 訊號 | 改善空間 | 證據／潛力 | 目前作業 |
|---|---|---|---|---|---|
| [多條件檢索](https://github.com/cactus8603/icml27-compositional-retrieval) | ✅ 已入池。CS-CLIP 嚴格三句排序只有 58.15%；左右顛倒句只判對 49.59% | ✅ CORE 重排前 50 名，獨立圖庫 mAP +15.79 點 | ✅ 距標註上限 12.87 點，固定前 50 名差 19.50 點；31/136 查詢前 50 名沒有正例 | 4/5、4/5 | 基線矩陣（P0–P4）在本機 3090 排隊；交錯召回 `26806` 判為未定（+0.25 點 Recall@50） |
| [多保真度歸因](https://github.com/cactus8603/icml27-multifidelity-attribution) | ✅ 同一軌跡下 EK-FAC 0.150 LDS，精確參考 0.953 LDS | ✅ 剩餘方向餘弦 0.206，通過門檻與置換控制；混合法比 Spell 高 0.041 LDS | ◐ 同軌跡差 0.80 LDS；換初始化只剩 0.019 LDS（區間跨 0），可轉移性未定 | 3/5、4/5 | `27191` 判未定（+0.042，區間跨 0）；預算曲線完成；下一步為多種子期望真值停損（約 8–10 H200 小時，PREREG 待寫） |
| [Λ 漏出預測](https://github.com/cactus8603/icml27-inoculation-lambda) | ◐ 語言漏出成立（無防禦 17/20 含西語，其他 0/20）；GCD 各臂差異未建立 | ◐ 一步近似通過數值檢查；51/51 組事前評分完成；方向待行為結果 | — 還缺重訓信度上限與 PHM 對照 | 2/5、4/5 | `27112` 行為試驗 51 格（38 格完成、1 格執行中）→ `27113` 驗收 → `27162` 比較 |
| [文生圖罕見失敗](https://github.com/cactus8603/icml27-t2i-rare-failure) | ◐ CLIP 前置檢查通過；正式雙偵測器事件率待測 | ◐ 代理設定 213/10,000 命中；正式訊號待人工盲評 | ◐ 高斯模擬省 88 倍（10⁻⁵）、13 倍（10⁻⁴）；實際生圖待量 | 2/5、4/5 | `27190` 第二段 11 格（執行中）→ `27512` 第二段 b 選格 |
| [強驗證器](https://github.com/cactus8603/icml27-fallible-verifier) | ◐ 強模型誤收 23.32%（GSM8K）、57.29%（ASDiv-A）；共同誤收未確立 | ◐ GSM8K Brier −41.95% 通過；ASDiv-A −6.35% 未過 15% 門檻 | ✅ 省 11.70%，事後最佳 48.29%，差 36.59 點 | 3/5、3/5 | 閒置 |
| [RL 稀疏蒸餾](topics/rl-sparse-distill/)（**暫停**） | ◐ 外部 32K 有交叉（pass@1 +7.60、pass@64 −1.30）；原 3K 配對截斷，不可比 | ◐ 單 token 分叉 +34.38 點，但截斷使區間為 −8.59 至 +39.06 | ◐ 分叉上限差 2.34 點；pass@64 上限差 3.68 點 | 2/5、3/5 | 暫停（2026-11-17） |

## 資源總表

**硬體與空間限制**

| 項目 | 狀況 |
|---|---|
| H200 叢集 | 共享節點 8 張 H200（每張 141 GB）；帳號上限 4 GPU、64 CPU、680 GiB RAM；單一作業最長 48 小時 |
| 本機 | 7 張 RTX 3090（24 GB）；`/ssd4` 無配額指令 |
| 叢集磁碟 | `/work` 群組配額 1536G。**2026-11-17 已用 1508G，只剩約 28G**；報告 63 時（11-16）還剩約 111G。下載任何新權重前要先清空間 |
| 截止 | ICML 2027 截稿 2027 年 1 月底 |

**各題資源**

- **VRAM**：叢集實測峰值，取自 [`codex/HARDWARE.md`](codex/HARDWARE.md)。vLLM 和 JAX 會預先保留顯存，所以數字偏大，實際需求較小。
- **GPU 時**：以 H200 計。完整論文欄取自[報告 63](reports/2026-11-16-63-full-paper-resources-and-allocation.md)，是推估。
- **新增磁碟**：完整論文必備項目要新下載的量，不含已在本機的檔案。

| 題目 | 實測峰值 VRAM | 能否用 3090（24 GB） | 下一步 GPU 時 | 完整論文 GPU 時 | 完整論文新增磁碟 | 需申請的資源（已降為低優先） |
|---|---|---|---|---|---|---|
| 多條件檢索 | 2.1–6.4 GiB | ✅ 能，已在 3090 開發 | 約 0.1／次 | 主線 5–15；加蒸餾版 +30–70 | 22–25G | Winoground（加分） |
| 多保真度歸因 | 126.4 GiB（JAX 精確重訓）；便宜方法 76 GiB | ✗ 只能用 H200 | 0.5–0.7 | 60–90 | 2–9G | 無 |
| Λ 漏出預測 | LoRA 39.6 GiB；fp32 梯度 71.7–81.6 GiB；生成 17.7 GiB | ✗ 7B 不行；更大模型要更多 | 5.3–7.0 | 150–350 | 55–85G | Llama-3.1-8B（改用 OLMo） |
| 文生圖罕見失敗 | 生成 36 GiB；基準測試 52.9 GiB | ✗ 原批量不行 | 第二段約 38 | 300–400 | 20–30G（題目上限 35G） | Ring-A-Bell、MMA（改用 I2P、UnlearnDiffAtk） |
| 強驗證器 | 125–127.5 GiB（vLLM 預留；32B-AWQ 權重約 19G） | ✗ 原配置不行 | 0.25–0.75 | 30–60 | 30–50G | 無 |
| RL 稀疏蒸餾 | 72–127 GiB（vLLM 預留；1.5B 模型） | ◐ 縮小 KV 預留後或許可以，未驗證 | 每組 5–7 | 500–800 | 約 45G | GPQA、Llama-3.2-3B（加分） |

- **主機 RAM**：大部分 H200 作業申請 170 GiB，實際峰值 2–29 GiB。
- **六題合計**：完整論文約需 1,100–1,800 H200 GPU 時、190–250G 新磁碟，不可能全做。分配建議見[報告 63](reports/2026-11-16-63-full-paper-resources-and-allocation.md#分配建議供使用者決定)。

## 主攻（2026-11-17）

- [多條件影像檢索](topics/compositional-retrieval/)　[repo `icml27-compositional-retrieval`](https://github.com/cactus8603/icml27-compositional-retrieval)　**第一主攻**，在本機 3090 上進行。
- [用少量精確計算改善訓練資料歸因](topics/multifidelity-attribution/)　[repo `icml27-multifidelity-attribution`](https://github.com/cactus8603/icml27-multifidelity-attribution)　**第二，H200 第一順位跑停損**。主張改為：在種子雜訊主導的設定下，以多種子期望重訓效果為目標，用便宜集成當低雜訊先驗，加上少數種子的精確 replay 做收縮校正。停損實驗約 8–10 H200 小時；現象與空間都 PASS 後升為 H200 主線，排在 Λ 與文生圖之前。[文獻與 SOTA（報告 65）](https://github.com/cactus8603/icml27-multifidelity-attribution/blob/main/reports/2026-11-17-65-sota-and-literature.md) · [判定（報告 66）](https://github.com/cactus8603/icml27-multifidelity-attribution/blob/main/reports/2026-11-17-66-go-no-go-expected-lds.md)

## 優先完成現象與訊號

- [用訓練前的 Λ 預測防禦漏出](topics/inoculation-lambda/)　[repo `icml27-inoculation-lambda`](https://github.com/cactus8603/icml27-inoculation-lambda)（已搬）　補完四類防禦的漏出差異，檢查 Λ 能否預測哪種設定比較容易漏出。
- [文生圖罕見失敗的機率估計](topics/t2i-rare-failure/)　[repo `icml27-t2i-rare-failure`](https://github.com/cactus8603/icml27-t2i-rare-failure)（已搬）　補完正式事件率和人工盲評，確認生成失敗有多罕見、偵測器報出的陽性有多少是真的。

## 其他候選

- [強驗證器與少量金標](topics/fallible-verifier/)　[repo `icml27-fallible-verifier`](https://github.com/cactus8603/icml27-fallible-verifier)（已搬）

## 暫停與淘汰

- [大型平行解碼提交集合](topics/large-commit-sets/)　論文潛力降為 **2/5**，暫停。
- [只移植少數 RL 推理決策的蒸餾](topics/rl-sparse-distill/)　論文潛力 **3/5**，暫停（2026-11-17）：完整論文約需 500–800 H200 GPU 小時，時間不夠。
- [注意力寫入與最佳化器](topics/gate-optimizer.md)　論文潛力 **2/5**，本輪淘汰。
- [條件擴散的兩個時鐘](topics/biased-gen-cfg.md)　論文潛力 **2/5**，本輪淘汰。

[選題標準](criteria.md) · [研究排程](shortlist.md) · [實驗硬體](codex/HARDWARE.md) · [作業紀錄](killtests/SUBMITTED.log)
