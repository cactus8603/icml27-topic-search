## R3 文獻新穎性快篩（2026-11-14，唯讀，沒有改任何檔案）

**結論先講**：我沒找到任何一篇工作做了「新權重讀舊權重算的 KV 時，偏移多大、何時超過數值底線，並用 dKV 預測、勝過 lag」這件事。但題目的前提已有不少人碰過：PipelineRL 圖 7 的定性結論是沿用舊 KV「只略差」，近一年的系統做法也都把它當成已解決的問題。所以剩下的新穎性只有很窄的一塊。

### A. 本題複述
- 非同步 RL（例如 PipelineRL）在生成途中換上新權重時，進行中的請求不重算前綴，新權重讀到的是舊權重算的 KV。
- 本題量三種條件：nn（新權重＋新 KV）、no（新權重＋舊 KV）、oo（全舊）。以「同一權重只改分塊方式」的數值 KL 為底線，KL(nn‖no) 達到底線 2 倍算越界。
- 主張是：越界與否、偏移大小，用「同一前綴在新舊權重下的 K/V 相對變化（dKV）」配合前綴長度做 log-log 迴歸，預測得比只看落後步數（lag）準。這樣就能逐次決定哪一次更新需要重算 KV，取代 PipelineRL 只有一個「最大 lag」的旋鈕。
- 目前證據：
  - H200 W1r 只有 Qwen3-1.7B、SFT 快照、單一 seed。
  - 留出誤差 19.9%，門檻 20%，邊際很薄；優勢主要來自換領域的 B2 stream。
  - 真實 GRPO 快照的 W2 還沒跑。

### B. 最接近的工作（依重疊程度排序）
| # | 論文 | 方法一句話 | 它報告的關鍵數字 | 與本題重疊 | 本題仍不同之處 |
|---|---|---|---|---|---|
| 1 | **PipelineRL**，Piché, Kamalloo, Pardinas, Chen, Bahdanau；arXiv 2509.19128（2025-09）。OpenReview 正式稿的場地未確認（頁面被驗證擋住）。https://arxiv.org/abs/2509.19128 | 提出 in-flight 權重更新。§5.1／圖 7 比較「重算 KV」與「沿用舊 KV」兩種混合策略序列，對完全 on-policy 的 KL | Qwen2.5-7B、最大 lag 32、序列長 2048、從第 0／100／190 步的 checkpoint 開始量。原文：沿用舊 KV「introduces only slightly higher divergence」，並據此選擇保留 KV。圖中的 KL 數值我讀不到（HTML 版文字沒有） | 直接比較過 stale 與 recompute，題目前提在這裡已有定性答案 | 沒有拿數值底線做比較，沒有分析偏移和 ΔW、lr、dKV 的關係，也沒有提出何時該重算的準則 |
| 2 | **PR vs CR 部落格**，Tong, Li, Sheng（2026-01，WIP，非論文）。https://tongyx361.github.io/blogs/posts/pr-vs-cr/index.html | 把「部分 rollout＋沿用舊 KV」（PR-SKV）列為一種分布偏移，並寫成 open question | 沒有實驗（原文：「Experiments are still in progress」） | 問題陳述與本題幾乎一樣，也預言「小更新可忽略、大更新或長上下文才顯著」 | 沒有量測，也沒有預測器 |
| 3 | **Motif 3 Technical Report**，Lim 等（Motif Technologies）；arXiv 2608.09119（2026-08）。https://arxiv.org/abs/2608.09119 | §5.2.2：in-flight 更新時保留 KV、不重算；用行為策略的 logprob 做 token 級 IS 修正，權重截在 [0.2, 5.0]，軌跡年齡上限 1 步 | 沒有量 stale KV 造成的偏移 | 工業界的實際做法：沿用舊 KV，偏移交給 IS 吸收 | 沒有量化，也沒有重算準則 |
| 4 | **NeMo-RL PR #4018**（2026-09／10，GitHub PR，非論文）。https://github.com/NVIDIA-NeMo/RL/pull/4018 | prefix cache 跨權重更新沒清掉，新請求會用到年齡無上限的舊 KV；修正為 refit 時清 cache | Nemotron-H 非同步 GRPO、約 49k token 的 prompt（約 85% 共用前綴）：被 logprob 誤差門檻遮掉的序列從 1–2/2048 升到 17 步後的 242/2048 | 實測證據：舊 KV 累積久了，logprob 偏差會顯著變大 | 只是工程修正；in-flight 請求仍沿用舊 KV，只加了警告 |
| 5 | **torchtitan PR #4761**（2026-10 合併，GitHub PR，非論文）。https://github.com/pytorch/torchtitan/pull/4761 | 預設跨權重同步保留 KV，用 policy 版本當 cache_salt，再以 max_offpolicy_steps 控制落後 | Qwen3-4B DAPO：preserve 模式被搶占的請求 0 個、reset 模式 687 個；TTFT 34.8 ms 對 93.6 ms；端到端差異在雜訊內。沒有 KL 數字 | 主流框架已把「沿用舊 KV、只用 lag 上限控制」做成預設 | 和本題要挑戰的正是同一個「只看 lag」的做法，但它沒有任何偏移量測 |
| 6 | **Can KV Cache Be Reused Across Model Updates? A DeepSeek-V4-Flash Case Study**，Tao, Jin, Zhao；arXiv 2611.14051（2026-11-10）。https://arxiv.org/abs/2611.14051 | 跨 4 次發布版本直接重用 KV，提出 Functional Cache Divergence（比較重用 cache 與原生 cache 下的下一個 token 分布） | TTFT 中位數加速 2.10–5.08×；檢索準確率從接近原生到大幅下降都有，視更新而定 | 「舊權重的 KV 給新權重讀」的分布偏移量測幾乎相同，而且是 4 天前才出的 | 它處理的是發布版本之間的大更新，不是 RL 小步更新；沒有底線，沒有 dKV／lag 預測 |
| 7 | **KVShareArena**，Shi, Lou（UCF）；arXiv 2609.10266（2026-09）。https://arxiv.org/html/2609.10266v1 | 跨 checkpoint 重用 KV 的評測基準 | 損害較受權重空間距離左右，較不受譜系影響（k-projection 距離 0.136 對 0.090）；免訓練方法的平均分數變化在 ±0.02 內 | 已經觀察到「權重距離可解釋重用損害」，接近本題「用 ΔW／dKV 預測」 | 測的是任務分數；沒有 RL 相鄰步、沒有底線、沒有預測模型 |
| 8 | **DroidSpeak**，Liu 等；NSDI '26（arXiv 2411.02820）。https://arxiv.org/abs/2411.02820 | 同一 base 的不同 fine-tune 之間共用 KV；離線 profiling 找出關鍵層並重算那些層 | 全部重用時 HotpotQA 常掉超過 50% 準確率點；約 11% 的層是關鍵層；prefill 加速 1.7–3.1×；吞吐最多 4× | 跨權重重用 KV，並選擇性重算 | 權重差很大（不同 fine-tune）；用任務品質決定，不是用 KL 或 dKV 預測 |

其他確認過但重疊較低的工作：
- **KV4-RL**（arXiv 2610.14448）：部分 rollout 換版本後選擇重新 prefill，理由是舊 KV「cannot be reused」；沒有量化。
- **Activated LoRA**（NeurIPS 2025，arXiv 2504.12397）：讓 adapter 在設計上就能直接用 base 模型的 KV。
- **KV-Cache Patching**（arXiv 2610.41480）：在 LoRA 與 base 之間互換 KV，可部分或完全移除 fine-tune 行為；摘要只有定性結論。
- **Stale but Stable**（arXiv 2607.18722）：lag 造成的 logprob 偏差；不涉及 KV。
- **GrowMTP**（arXiv 2609.16648）：給出單次更新偏移 ≤ ½·Fisher 範數、與 lr 成線性的領頭階界。

未確認：Stanford CS224R 的 PipelineRL lag 專題（PDF 解析失敗）；CacheBlend 選 token 用的 KV 偏差準則（arXiv 頁面沒寫，我是憑記憶）；AReaL 中斷式 rollout 是否丟棄 KV（只看了摘要）。

### C. 有沒有工作做了實質相同的事？
沒有。剩下的新穎性是：在 RL 小步更新（lag 1–32、RL 常用 lr、含 LoRA）下，

1. 用同引擎的數值底線為「沿用舊 KV」的偏移定出顯著性門檻；
2. 用一個便宜、可在線算的 dKV 特徵預測偏移，並證明它勝過 lag，最好也勝過 lr·lag 和 ‖ΔW‖。

這塊很窄：
- 「量偏移」的部分已被 PipelineRL 圖 7 與 2611.14051 部分涵蓋。
- 「權重距離可解釋重用損害」已被 KVShareArena 定性觀察過。
- 真正獨有的是「dKV 預測器加上重算準則」，而且必須在真實 GRPO 上贏過 lr·lag 和 dW，才站得住。

### D. 衝突或已隱含的結論；審稿人最可能的質疑
- **衝突**：PipelineRL 的結論是沿用舊 KV 只略差、不值得重算；torchtitan 的實測也顯示端到端差異在雜訊內。W1／W1r 在小 lr 下停在底線附近，這和它們一致，等於支持「這問題不重要」。要有價值，必須證明在真實 RL 設定下存在明顯越界、而且會影響學習。NeMo #4018 的長前綴累積案例是目前唯一的反例。
- **已隱含**：一階或二階理論（Fisher／Taylor；GrowMTP 的界）本來就推得出偏移 ∝ 更新量。dKV 是更新量的直接代理，它比 lag 準在審稿人眼中幾乎是預期之內。
- **最可能的質疑**：
  - 「這不就是 PipelineRL 圖 7 的延伸？」
  - 「直接量 KL 或 IS 比率（Motif 3、TIS／MIS 做法）就好，何必預測？」
  - 「dKV 要用新權重重算前綴才拿得到，成本和直接重算差不多」：這是致命的成本論點，必須說明 dKV 能用抽樣、少數層或少數 token 便宜估出。
  - 「IS 修正已經吸收這個偏移」。
  - 「只有 1.7B、SFT、單一 seed，誤差 19.9% 對門檻 20%」。

### E. 審稿人預期的規模、資料、系統、指標
- **模型**：至少 7B–8B，並有一個 MoE 或 30B 級別。PipelineRL 用 7B，Stale but Stable 用 Qwen3-30B-A3B。
- **資料**：真實 GRPO／DAPO 數學推理（OpenReasoner Zero、DAPO-Math），長生成 8k–32k。
- **系統**：在 PipelineRL、verl 或 NeMo-RL 的真實 in-flight 更新中實測。
- **指標**：
  - KL 或 logprob 誤差、IS 比率尾部、ESS；
  - 下游學習曲線與準確率（AIME 等），比較重算與沿用；
  - 重算成本與吞吐、TTFT；
  - 預測器相對 lag、lr·lag、‖ΔW‖ 的留出誤差；
  - 多個 seed。

### F. 發表場地與 ICML 適配度
- **同類工作實際發表的地方**：
  - RL 系統：arXiv 加 NeurIPS／ICML；PipelineRL 在 OpenReview 上有正式稿，場地未確認。
  - 非同步 RL 的 off-policy 修正：ICML 2026、NeurIPS 2026，例如 Missing Old Logits 同時列在兩處的頁面上。
  - KV 重用：NSDI、EuroSys、MLSys。
  - 大量工程結論只出現在框架 PR 和技術報告。
- **ICML 適配度：中偏低。** 作為「預測律＋重算準則」的經驗研究，只有在真實 RL 規模下證明「越界常見且影響學習」，加上「預測器便宜且勝過 lr·lag／ΔW」才有機會。否則比較像 workshop 或 MLSys 的短文，或是 PipelineRL 類論文中的一段分析。

### 主要來源
- [PipelineRL](https://arxiv.org/abs/2509.19128)
- [PR vs CR 部落格](https://tongyx361.github.io/blogs/posts/pr-vs-cr/index.html)
- [Motif 3](https://arxiv.org/abs/2608.09119)
- [NeMo-RL PR #4018](https://github.com/NVIDIA-NeMo/RL/pull/4018)
- [torchtitan PR #4761](https://github.com/pytorch/torchtitan/pull/4761)
- [2611.14051](https://arxiv.org/abs/2611.14051)
- [KVShareArena](https://arxiv.org/html/2609.10266v1)
- [DroidSpeak](https://arxiv.org/abs/2411.02820)
- [KV4-RL](https://arxiv.org/html/2610.14448)
- [KV-Cache Patching](https://arxiv.org/abs/2610.41480)
- [Stale but Stable](https://www.alphaxiv.org/overview/2607.18722)
- [GrowMTP](https://arxiv.org/pdf/2609.16648)
- [Activated LoRA](https://arxiv.org/abs/2504.12397)
