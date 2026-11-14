# 六篇 ICLR 2027 Rebuttal 原始討論

分數箭頭取自 11/06 及 11/14 晚間的兩期 Radar，斜線分隔排序後的投稿評分。下文以公開 OpenReview 的 review、作者回覆、reviewer 跟進交叉核對。作者回覆的數字以原評估切分和比較條件為準。 [Radar 資料](https://github.com/dipta007/iclr-2027-rebuttal-radar/blob/a67aefa2c4c2d231e471abab4c36cb771c672ec9/docs/data/papers.json)

<a id="cocom-20666"></a>

## CoCom #20666

*Contextual Commitment for Parallel Masked Diffusion Decoding*。投稿分數 `3/3/2 → 4/3/2`，均分 2.67 → 3.00。

**原問題。** GwZc 發現 CoCom 用 block 64，Native block size 卻沒寫。表裡 LLaDA-1.5-8B 的 GSM8K Native 只有 57.24%，遠低於官方 block-wise 結果。Reviewer 問這個落差是否來自 baseline 的解碼設定。 [原評語](https://openreview.net/forum?id=BJw3PRAyUN&noteId=TOQtQdQVu0)

**實際補的實驗。** 原 Native 用整段 256-token canvas，CoCom 用 block 64。作者重跑 Native block 64，沿用相同 prompt、256-token 輸出長度、Native 預測規則與 256 次 forward 預算，四個基準都報。 [作者對照表](https://openreview.net/forum?id=BJw3PRAyUN&noteId=48ycds756K)

| LLaDA-1.5-8B | 原 Native 準確率 | Native block 64 | CoCom block 64 | Native／CoCom NFE |
|---|---:|---:|---:|---:|
| HumanEval | 40.24% | 39.63% | 43.90% | 256／64.76 |
| MBPP | 36.80% | 39.00% | 40.40% | 256／51.58 |
| GSM8K | 57.24% | 82.03% | 80.97% | 256／61.39 |
| MATH500 | 24.00% | 38.40% | 38.60% | 256／96.37 |

**回覆怎麼寫。** 作者先粗體交代原設定不一致，再說 block size 能獨立影響準確率。用一張表同時保留原結果與修正後結果，正面寫出 GSM8K 比修正後 Native 低 **1.06 個百分點**、但少 **76.0% NFE**。新版在 protocol table 補 block length，附錄放完整對照，主文寫品質與成本的取捨。 [作者回覆](https://openreview.net/forum?id=BJw3PRAyUN&noteId=48ycds756K)

**為什麼加分。** GwZc 明說新對照顯示和 Native 的品質相近、效率改善顯著，升到 **4（clear accept）**。這個回覆裡的精確成本數字是 NFE。 [Reviewer 跟進](https://openreview.net/forum?id=BJw3PRAyUN&noteId=vsCviUfBFZ)

<a id="tacg-32163"></a>

## TACG #32163

*Trajectory-Aware Commit Gating for Diffusion Language Model Decoding*。投稿分數 `2/2/2 → 3/2/2`，均分 2.00 → 2.33。

**原問題。** RDjT 認為 per-task tuning 可能撐起 headline gain，單一 seed 的結果看不出穩健性。Dream 的 NFE 甚至上升。他要求所有任務用一套參數、多 seed、釐清 KLASS eligibility 是否也在實驗中生效，並寫清 HOLD probe。 [原評語](https://openreview.net/forum?id=ofh5olmwsc&noteId=K7F06ufE2c)

**實際補的實驗。** LLaDA-8B 四個任務共 2,240 題，每個 backbone 各用跨任務共用設定。原稿已列的一套 LLaDA 參數，在回覆的共用表裡把平均準確率 **48.00% → 49.62%**，配對 bootstrap 的 95% CI 為 **+0.36 至 +3.00 個百分點**，平均 NFE **76.59 → 67.87**。逐任務調參是 +3.02 點，共用設定只保留其中 53.6%。Dream 的 balanced 設定 **55.79% → 56.24%**，NFE **75.72 → 76.61**，差的 CI 跨零。LLaDA 是 greedy、結果對 seed 不變，Dream 表沿用固定 seed 的 stochastic fallback 並報題目層級 bootstrap，因此不能把這張表解讀為 Dream 跑了三個獨立種子。作者也指出 Confidence + TACG 原本就已停用 KLASS eligibility，並改方法與偽碼讓設定可查。 [共用設定回覆](https://openreview.net/forum?id=ofh5olmwsc&noteId=oflLXlTI6H) · [KLASS 設定](https://openreview.net/forum?id=ofh5olmwsc&noteId=rX9IlGk36B)

**另一位 reviewer 要的控制。** 固定每輪額外提交上限同為 2，比 `Confidence + HG + Extra` 和完整 TACG。NFE 接近（**68.15 vs. 67.87**），加入 TILG 後平均準確率 **48.60% → 49.62%**。另外在同一張 H200、batch 1、GSM8K、長度 256、block 64，對 Native Confidence 的實測從 **1,832.6 ms → 1,497.8 ms**，約 **1.22×**，峰值顯存增加 0.38 GiB。 [組件對照](https://openreview.net/forum?id=ofh5olmwsc&noteId=TkZGziphPI) · [同 block 計時表](https://openreview.net/forum?id=ofh5olmwsc&noteId=VST0j10XYe)

**回覆怎麼寫、為什麼加分。** 作者按 `W2 & Q1` 等原編號回答，把原稿已列的共用設定、rebuttal 新試的設定和 per-task 調參分列。HOLD probe 說明只把目標位置多遮一輪，其他選中的位置照常寫入，再用同 confidence bin 比 top-1 翻轉率。RDjT 說多數疑慮已解，特別肯定「何時提交」與「寫哪個 token」的區分、HOLD probe、commit-count replay 和實測效率，表示加分，現為 **3**。他仍覺得 heuristic 偏多、共用設定增益較小，追問一句話的核心新意。作者最後把主張落在「保留當下的原生 argmax，歷史訊號只決定何時寫」。 [HOLD probe](https://openreview.net/forum?id=ofh5olmwsc&noteId=drNRWtFCgx) · [Reviewer 跟進](https://openreview.net/forum?id=ofh5olmwsc&noteId=YtgYliP2jg) · [作者回答核心貢獻](https://openreview.net/forum?id=ofh5olmwsc&noteId=QwmqqcaAHV)

<a id="sample-the-order-50735"></a>

## Sample the Order #50735

*Sample the Order, Not the Token: Test-Time Scaling for Masked Diffusion Language Models*。投稿分數 `3/2/2 → 3/3/2`，均分 2.33 → 2.67。

**原問題。** odeR 問 high-temperature baseline 的崩潰，究竟來自 token 抽樣，還是同一溫度也改變 unmasking 的位置排序。他還問在不同 backbone、stochastic sampler 和 matched compute 下，OSE 是否仍有優勢。 [原評語](https://openreview.net/forum?id=mISIJ9ktNq&noteId=7ItArDGGPl)

**實際補的實驗。** 在 Dream 固定排序溫度 `T_rank=0.1`，只變 token 溫度，再比 coupled 與 order-space exploration（OSE）。在 `T_tok=0.7` 的 vote@8，GSM8K 從 coupled **24.8%** 到 decoupled **91.5%**，高於 OSE 的 **88.0%**。MATH500 是 **7.8% → 50.5%**，OSE 是 **58.2%**；該設定下 single-draw accuracy 是 decoupled **23.5%** 對 OSE **38.7%**。作者另加兩個 backbone、Dream block size 32/64 與溫度掃描，報告 OSE 與 default 同樣本數時使用相同 forward 次數。 [分離溫度對照](https://openreview.net/forum?id=mISIJ9ktNq&noteId=zqZRbkGDNB) · [成本與加測](https://openreview.net/forum?id=mISIJ9ktNq&noteId=ZAoykswD0a)

**回覆怎麼寫、為什麼加分。** `Q1` 先寫改了哪個控制變數，接三欄對照，再解釋為何結論要縮窄。odeR 說新實驗處理了大部分疑慮，表示升分，現為 **3**。他特別指出保護 ranking 後，token 溫度在短答已具競爭力。作者下一則回覆交代已改引言、摘要與結論，把崩潰限於 ranking 和 token 共用溫度的 default sampler，並交代 OSE 在長生成和需逐任務調溫時的取捨。 [Reviewer 跟進](https://openreview.net/forum?id=mISIJ9ktNq&noteId=H9uOlQ9Clh) · [作者跟進與修訂](https://openreview.net/forum?id=mISIJ9ktNq&noteId=gLiv1Hnnrj)

<a id="restorekv-28487"></a>

## RestoreKV #28487

*RestoreKV: Recovering Full-Cache Behavior Under Aggressive Query-Agnostic KV Cache Eviction*。投稿分數 `3/3/1 → 3/3/3`，均分 2.33 → 3.00。

**原問題。** QRC6 認為論文只對照 per-context 最佳化還不夠，因為 KV-Distill 和 Still 也會訓練可重用的壓縮器。他要求固定八個補償 states，比較「針對保留的 cache 聯合學」和「單獨學八個 states，到推論時再接上同一份保留 cache」。 [原評語](https://openreview.net/forum?id=6x2I3ITASd&noteId=8rIRskd9so)

**實際補的實驗。** Qwen3-4B、RULER-4K，同一個 evictor、同樣的 retained KV、八個補償 states 和最終推論預算。兩個主要對照都可在 eviction 前看完整 cache。保留 **5%** 時，原 KVzip **38.2**、獨立訓練再拼接 **54.3**、聯合訓練但只看保留 cache **64.4**、完整聯合訓練 **73.2**。保留 **20%** 時，獨立訓練 **93.3**、聯合訓練 **93.5**，只看保留 cache 的版本反而 **93.7**。作者也補與近作的 RULER-32K 對照，並明確區分發表數字、架構 profiling 估計與實測延遲。 [隔離訓練目標的對照](https://openreview.net/forum?id=6x2I3ITASd&noteId=On7vlCkBNk) · [既有方法與成本比較](https://openreview.net/forum?id=6x2I3ITASd&noteId=KehAo2Jlsi)

**回覆怎麼寫、為什麼加分。** `Q1` 開頭直接回答有沒有差，指出 **+18.9 點發生在 5% 預算**，再交代 20% 預算下差距縮小。QRC6 明說他要求的實驗實質強化了論文相對於既有方法的獨特貢獻，表示會提高 recommendation，現為 **3**。這張表把「訓練目標帶來收益」和「多了記憶容量」分開。 [Reviewer 跟進](https://openreview.net/forum?id=6x2I3ITASd&noteId=4wCxFxcOQg)

<a id="dynin-omni-11132"></a>

## Dynin-Omni #11132

*Dynin-Omni: Omnimodal Unified Large Diffusion Language Model*。投稿分數 `3/2/2 → 3/3/3`，均分 2.33 → 3.00。

**原問題。** sb2y 問，在兩個階段各自開關 EOS supervision，是否真能證明完整 `Off→On` 優於 `AlwaysOn`。NJyB 另問合併規則是否在相同後續訓練下比較，以及同去噪步數是否真代表同成本。 [sb2y 原評語](https://openreview.net/forum?id=AcFMWNvY8v&noteId=EcX96slqun) · [NJyB 原評語](https://openreview.net/forum?id=AcFMWNvY8v&noteId=E7CQ4ubpXC)

**實際補的實驗。** 固定 8B MMaDA 初始化、相同資料和 **6,144** 次更新跑完整路徑。`Off→On` 的 ASR／TTS WER 為 **64.27%／15.57%**，staged `AlwaysOn` 為 **71.29%／16.41%**，最近方法的 `attenuation→On` 為 **65.16%／17.08%**。混合訓練的 `AlwaysOn` 在 TTS 單項更好（**14.78%**）。合併規則則從同 checkpoint 出發，共用 backbone 插值、資料與合併後 **8,192** 次更新，ASR WER 為自家 **22.11%**、兩種替代 **28.69%／29.25%**。 [完整路徑和合併控制](https://openreview.net/forum?id=AcFMWNvY8v&noteId=oDirebeUB2)

**成本怎麼答。** 作者固定各模態 token 長度和 refinement 設定，在一張 B200、兩筆 inputs 量時間和 FLOPs。128 NFE 的 joint／anchored 延遲 **10.543／10.612 秒**，幾乎一樣。另有一組 32 題、各重跑兩次的配對資料，joint 相對 anchored 的 Overall 改善在 16 NFE 的 95% CI 為 **[0.0550, 0.2198]**，在 128 NFE 是 **[−0.1050, 0.2198]**。作者把效果限在低預算時的新證據，並標明減量控制和全規模結果的不同條件。 [成本與不確定性回覆](https://openreview.net/forum?id=AcFMWNvY8v&noteId=v2n6ygnc5P)

**為什麼加分。** sb2y 點名完整 EOS 路徑和等量合併後訓練，認為主要實驗疑慮解了，仍認為架構新意有限，升到 **3**。NJyB 點名 baseline、兩項控制、成本與不確定性分析，也升到 **3**，保留減量實驗外推的疑慮。 [sb2y 改分說明](https://openreview.net/forum?id=AcFMWNvY8v&noteId=FXJbkhDwwe) · [NJyB 改分說明](https://openreview.net/forum?id=AcFMWNvY8v&noteId=f1Uw2V1UZm)

<a id="rethinking-soft-tokens-23776"></a>

## Rethinking Soft Tokens #23776

*Rethinking Soft Tokens for Parallel Decoding in Diffusion Language Models*。投稿分數 `3/2/1 → 3/2/2`，均分 2.00 → 2.33。

**原問題。** C2iV 指出，SLERP 和正規化的線性插值實際走同一條球面弧線，使用同數值的插值係數會落在不同角度，原消融無法支持兩者有不同的幾何軌跡。Reviewer 也指出「用來算 reference 的輸入」可能已給了被評估位置的候選 token，另有 **48 個主要設定中 26 個**和附錄描述不符的重現性疑問。 [原評語](https://openreview.net/forum?id=9yFvS60i0H&noteId=29VZr7q9tM)

**實際補的實驗與回覆。** 把被評估位置維持 masked，只讓其他位置提供訊號。在 LLaDA 上，和 multiplicative／additive reference 的平均 JS divergence 分別為 **0.0737／0.1989**。作者接受兩種插值走同一路徑，將消融主張限定在同數值參數下的**角度參數化**，也說 reference 可以用來診斷，但不識別模型內部機制。Reviewer 回覆之後，作者另報已更新補充程式和 README 的預設設定。 [遮罩控制與理論範圍](https://openreview.net/forum?id=9yFvS60i0H&noteId=qky6sG6nHl) · [作者更新重現說明](https://openreview.net/forum?id=9yFvS60i0H&noteId=uc3zv7qz7e)

**為什麼只升一分。** C2iV 明說 additive reference 和 SLERP 的澄清答到了部分理論疑問，**1 → 2**。當時仍質疑論文設定與補充檔案不一致，以及 baseline 能否重現。後來作者更新檔案的說法出現在這則改分之後，公開跟進裡沒有看到 reviewer 再確認。 [Reviewer 改分說明](https://openreview.net/forum?id=9yFvS60i0H&noteId=MFBZrdenL8)
