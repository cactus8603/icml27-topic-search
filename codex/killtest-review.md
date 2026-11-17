# ICML 2027 停損判準與平行排程審查

2026-11-14（台灣時間）

**建議並行保留四題，論文優先順序為 CFG 偏差泛化 → 多保真度資料歸因 → Inoculation 漏出預測 Λ → Neural Chameleons 防禦。** 前兩題的正面貢獻較清楚。Λ 先補一個小規模的跨防禦預測實驗。Chameleons 已有單層模型權重和吞吐數字，適合當第四題做快篩。多條件檢索已交出正式 P1，兩條路徑均未過，建議停目前的 LLR 版本。 [原 shortlist](../shortlist.md) · [檢索正式判讀](../killtests/compositional-retrieval/results/aggregate.md#L118)

## 第一輪怎麼並行

| 論文順位 | 題目 | 第一輪工作與下一個決策 | GPU 小時估算 |
|---:|---|---|---:|
| 1 | CFG 偏差泛化 | CelebA 與階層模型訓練已提交。先從紀錄看吞吐，等最低點落在邊界的規則定好再判讀重現與三模型探索曲線。 [提交紀錄](../killtests/SUBMITTED.log#L10) · [實驗 README](../killtests/biased-gen-cfg/README.md) | 第一週預估 8–20，上限 30 |
| 2 | 多保真度資料歸因 | H200 吞吐已有記錄。優先拿到重訓和便宜基線的結果，用預先固定的精確查詢批次確認 MAGIC 重現與混合增益。 [吞吐紀錄](../killtests/multifidelity-attribution/logs/mf-bench-26272_4294967294.out) | 階段 0 預估約 10–25，其中 50 個精確查詢依吞吐外推約 50 分鐘 |
| 3 | Inoculation 漏出預測 Λ | 有限差分已送出。等結果期間先整理 Phase B 的組別與判讀。接著用一個任務、五種設定、三個種子試跑，確認排序能否量得穩。 [提交紀錄](../killtests/SUBMITTED.log#L1) · [PREREG](../killtests/inoculation-lambda/PREREG.md#L37) | Phase A 預估約 4–7，小試跑另估 |
| 4 | Neural Chameleons 防禦 | 抽取與分析已提交。先看閘門 2b，確認八層基線確實有改善空間，再讀監測器主指標。 [提交紀錄](../killtests/SUBMITTED.log#L11) · [PREREG](../killtests/chameleon-defense/PREREG.md#L52) · [吞吐紀錄](../killtests/chameleon-defense/logs/cham-bench-26273.out#L41) | 活化抽取外推約 0.17，含分析的原估約 1.2，上限 4 |

有四張可用卡時，首批可按每題一張安排。有三張時，先給 CFG、資料歸因與 Λ。Λ 進入整理 Phase B 的空檔，就讓 Chameleons 補上。排程先扣掉已送出的工作所占卡數。CFG 的兩個訓練陣列和資料歸因的多個陣列，各自以同一題合計設定並行數。四題若都有可用的結果，可以一起保留。額外 GPU 時間優先給已出現主假設訊號、且下一關能判斷論文貢獻的題目。卡數是排程假設，工作狀態依提交紀錄及結果檔判讀，即時隊列與共用磁碟餘額未確認。磁碟依共用環境、模型快取和各工作的暫存尖峰合計。 [已送工作](../killtests/SUBMITTED.log) · [CFG 訓練陣列](../killtests/biased-gen-cfg/scripts/w1_celeba_train.sbatch#L6) · [資料歸因重訓陣列](../killtests/multifidelity-attribution/scripts/s0_retrain.sbatch#L6) · [Neon 配額中斷紀錄](../reports/2026-11-14-58-neon-lm-killtest-interrupted.md)

第 56 號回報把程式正確性、重現、量測雜訊和主假設依序檢查，這個結構合理。平行排程時，建議每題記錄已用 GPU 小時、下一關預估、主效應及區間、下一關通過後能支持的主張。**「閘門通過」只給下一階段的預算**。資料或實作不齊記「未能判讀」。測完且區間排除值得追的效果，才以「效應不足」停目前切入點。區間仍跨決策線的灰區，用事前約好的那一次補樣與預算判斷。明顯低於實務門檻但區間很寬的 pilot，可以基於資源順位暫停。 [第 56 號回報通用規則](../reports/2026-11-14-56-killtest-criteria-review.md#通用規則)

## 優先四題

### 1. 條件擴散模型偏差泛化（CFG）

**現行設計。** 第一週重現 CelebA 的偏差曲線，並用三個階層模型驗證 oracle 曲線。第二週正式看同一模型對的 `t*(w)`，也就是平滑後分數距離的最低點。階層資料要求 `w=3` 的點估計至少早 20%、95% 區間排除零效應，CIFAR 要同向且區間排除零。20% 做早期資源門檻合理，現有 PASS 仍容許區間內有小於 20% 的提早。工作區已有 BP 單元測試和預處理的 CelebA 資料。 [PREREG](../killtests/biased-gen-cfg/PREREG.md) · [BP 結果](../killtests/biased-gen-cfg/results/bp_unit_test.json) · [資料備妥紀錄](../killtests/biased-gen-cfg/PREREG.md#L75)

**先處理最低點與不確定性。** 合成曲線中，`w=0` 的真最低點在評估範圍之外，程式以最後一格 epoch 195 當最低點，與 `w=3` 的 epoch 120 比較後仍輸出 `c_pass_criteria_met=true`。目前已加上 `main_metric_censored` 標籤，但它仍可與通過旗標同時出現。任一分支最低點貼邊、過平或曲線被截斷，就先延長量測或依事前規則處理。程式重抽模型後也只看端點有沒有被抽到，忽略抽中次數。六個模型形成的十五對共享端點，建議先用逐模型留一檢查穩定性，再用能處理端點依賴的小樣本區間。 [主指標與邊界旗標](../killtests/biased-gen-cfg/scripts/aggregate_week1.py#L89) · [成對資料的端點依賴](https://arxiv.org/abs/1312.3398)

**補完正式判準。** 評估前先固定分數空間與噪聲步的加權方式。CelebA 用噪聲預測，階層模型目前用 `x0` 預測，兩者的 cosine 最低點可能不同。正控制應用同一類別的約 `n/K` 筆作對照，才對應條件分支實際看到的資料。原負控制要求每模型 70,000 筆，現有階層資料腳本只產 40,000 筆。每類約 11,667 筆仍可能進入偏差期，建議以同資料量的常數條件檢查由分條件造成的效應，並保留 condition dropout 當作解讀因素。加碼目前只處理階層資料過關、CIFAR 未過的組合，還要寫明反向、區間過寬或 CIFAR 單獨過關時的決策。論文的正面貢獻應再驗證早停或 guidance 規則在留出的設定能否改善品質與記憶的取捨。 [既有 CFG 審查](../killtests/biased-gen-cfg/REVIEW.md#L19) · [PREREG §3–7](../killtests/biased-gen-cfg/PREREG.md#L27) · [階層資料產生腳本](../killtests/biased-gen-cfg/scripts/gen_hier_data.sh)

### 2. 多保真度資料歸因

**現行設計。** CIFAR-10 階段 0 用 200 次重訓、50 個精確查詢驗證改進空間。便宜方法 LDS ≥0.8、`κ_res/κ < 0.5`、混合增益 <0.04 任一成立即停。階段 1 擴到 200 個查詢，混合方法須同時超過簡化 Spell 與最佳便宜方法至少 0.08、兩組配對區間下界皆 >0.03，並且跨種子至少持平。中間效果可走逐查詢可靠度路線。等量測預算和跨種子對照值得保留。現有 H200 記錄已驗證前向、反向、兩次重訓及 TRAK 投影的吞吐，階段 0 主指標仍待交付。 [PREREG](../killtests/multifidelity-attribution/PREREG.md) · [H200 吞吐紀錄](../killtests/multifidelity-attribution/logs/mf-bench-26272_4294967294.out)

**階段 0 的兩個硬停風險偏高。** 便宜方法若有 0.80、精確方法約 0.91，仍有約 0.11 的差距可驗。`κ_res/κ` 衡量所選子空間捕捉的矩陣能量，也可能與逐查詢等權的 LDS 排序不同。建議把兩個數字當作診斷，主要以「同成本比較的混合增益區間已排除 0.04」決定是否因效應小而停。還要核對重現幅度。原作 MAGIC 0.9087 用 500 查詢、300 次重訓、十個訓練種子匯總，現行程式用 50／200／單一種子做階段 0，0.88 適合作為初步警訊，跨設定的重現結論需要對齊條件或先建立種子變異。 [PREREG §4–6](../killtests/multifidelity-attribution/PREREG.md#4-閘門依序前三關失敗未能測試不判題目限-3-個工作天修修不好記為可行性停損) · [Spell 原文 §4.1 和表 1](https://arxiv.org/html/2609.15044v2#S4.SS1)

**仍需驗收重訓筆數。** 新版已檢查七個便宜基線與精確查詢區塊，但四次無移除重訓只到兩次也會估雜訊，預定 100 次的跨種子重訓有部分資料就會算分。建議先驗收指定的重訓編號，再裁定。跨種子差目前只要為負就直接停，可用配對區間確認小差距是否超過雜訊。階段 0 的 50 個查詢也包含在階段 1 的 200 個查詢中，建議另報剩下 150 個查詢上的增益。完整 `Y` 可先離線估潛力，論文要報省算力則需實際算 `B` 個方向並計入便宜歸因、選方向及重播成本。灰區把 `k` 與 `B` 一起擴大並重估方向，這個結果也要說明預算與查詢集合的變動。 [雜訊與資料驗收](../killtests/multifidelity-attribution/scripts/aggregate.py#L71) · [跨種子程式](../killtests/multifidelity-attribution/scripts/aggregate.py#L248) · [加碼定義](../killtests/multifidelity-attribution/PREREG.md#L42)

### 3. Inoculation 漏出預測 Λ

**現行設計。** Phase A 是零訓練的有限差分和外部數字對照，外部相關只供參考。真正的 Phase B 在三個任務各用 13 個有防禦設定，要求合併 Spearman ≥0.6、95% 下界 ≥0.3、三個任務同向，且拿掉任一防禦家族後仍 ≥0.4。拿掉無防禦對照以免灌高相關、把參考文獻的小樣本降為描述，這兩個設計合理。Phase B 的訓練與評估流程還未實作。 [PREREG](../killtests/inoculation-lambda/PREREG.md) · [目前實作範圍](../killtests/inoculation-lambda/README.md)

**先修缺件與校準。** 合成測試中，有限差分只給一個設定、三列資料且實測等於預測的兩倍，仍以 Pearson=1 通過。Phase B 的程式也讓每個任務僅八個設定、一個種子的表通過，而只有表頭的 CSV 得到 FAIL。建議有限差分先檢查完整設定與數值誤差，再看相關。Phase B 先驗證完整的任務、組別、種子與前三關結果，無法形成有效相關時回報未能判讀。 [有限差分判讀](../killtests/inoculation-lambda/scripts/aggregate.py#L49) · [Phase B 判讀](../killtests/inoculation-lambda/scripts/aggregate.py#L218)

**把分數與實際選防禦的效益接起來。** 目前 Λ 在 LoRA rank 64 量，最終訓練用 rank 32。可用同 rank 檢驗對應，跨 rank 的預測另報。主要相關會納入沒有達到 ID 學習目標的設定，可能讓 Λ 讀到學習程度差異。建議小試跑先看固定 ID 水準下的排序，並與已指定的梯度餘弦和一個各組都能算的簡單基線比較。若論文主張跨防禦泛化，至少要在留出家族或任務上有可用的預測價值。灰區新增六組要在看結果前指定，否則加碼時會改變評估對象。 [PREREG §1、§3](../killtests/inoculation-lambda/PREREG.md#1-凍結的-λ-版本主指標只用這一版)

### 4. Neural Chameleons 防禦

新版的閘門 2b 先要求第 12 層與八層基線從乾淨到測試情境的 TPR 各跌至少 30pp。這能先確定有東西值得改善。正式判讀看監測器 TPR 增加至少 20pp、下界 ≥10pp，偵測器 AUROC 已移出裁定。灰區改補同一概念的未用樣本，評估對象一致。現有權重與吞吐足以做快篩，所得結論限於這個單層 checkpoint 和實測概念。 [PREREG §4–7](../killtests/chameleon-defense/PREREG.md#L35) · [H200 吞吐紀錄](../killtests/chameleon-defense/logs/cham-bench-26273.out#L41)

目前由乾淨測試負例取 99 分位，這給的是乾淨條件的 ROC 表現點。主指標即使通過，也要看同一閾值下測試情境的負例 FPR 是否跟著升高。新版已增列該項，建議和 TPR 並列用於是否加碼的判斷。若要主張可部署於約 1% 的 FPR，用獨立負例校準並回報測試 FPR。若要主張提升來自使用的子空間，保留與隨機子空間的同成本對照。 [主指標、基線及 FPR 對照](../killtests/chameleon-defense/PREREG.md#L61)

## 其餘候選與停案

### 多條件檢索

目前的 P1 有兩條路徑，CLIP B/32 + LLR 的 COCO(Qwen) COR 下界 ≥45%，或 CS-CLIP 加 LLR 增益 ≥5pp 且配對下界 >0，各用單尾 α=0.0125。COR 要求加上正確細節後分數上升、加上錯誤細節後分數下降。**正式結果三個前置閘門全過，P1 兩路皆失敗。** 第一條 COR 23.18%，上界 27.50%。第二條相對實測 CS-CLIP 的 58.15% 下降 29.47pp，差值上界 −23.58pp。兩條上界離門檻夠遠，依事先規則可直接停現有方法。 [新版 PREREG §4–5](../killtests/compositional-retrieval/PREREG.md#L30) · [正式結果](../killtests/compositional-retrieval/results/aggregate.md#L118) · [Half-Truths 原文表 8](https://arxiv.org/html/2602.23906v2#A3.SS3)

獨立判讀已重算兩條路線，發現錯誤細節的 PMI 多數仍為正，使 COR 的上限落在 40.5%（CLIP）和 35.8%（CS-CLIP）。如果以後換新聚合方法，P2 現有設計可比較 `k=4` 與 `k=1` 的錯誤率。若要把差異歸因於條件數，建議用同一批正例與巢狀條件，並對齊負例難度，因為各 `k` 的候選影像族群可能不同。 [獨立判讀](../killtests/compositional-retrieval/VERDICT_stage1.md#L35) · [P2 判準](../killtests/compositional-retrieval/PREREG.md#L46) · [查詢取樣](../killtests/compositional-retrieval/scripts/e2_build_queries.py#L61)

### 文生圖罕見失敗

先用可精確計算的高斯尾機率驗證估計器，再小範圍檢查代理事件，最後做大樣本與人工盲標，順序合理。最新二十次重複的高斯輸出顯示 subset simulation 通過，CE-IS 未過。A 要至少三組模型與提示達標、每組十萬張至少五個命中，還有類別覆蓋要求。真機率正好在下限 `5×10⁻⁵` 時，單組十萬張達五次命中的機率約 56%。五次命中的精確 95% 區間約 `[1.6×10⁻⁵, 1.17×10⁻⁴]`。前兩輪依命中選組，最終再把篩選資料算進比率，會挑中受抽樣波動影響的組別。這個流程適合作探索，確認時用新的一段 seed 算區間。 [高斯閘門結果](../killtests/t2i-rare-failure/results/gate_a/gate_a.md) · [PREREG §4–5](../killtests/t2i-rare-failure/PREREG.md#L39) · [精確二項區間](https://itl.nist.gov/div898/handbook/prc/section2/prc241.htm)

新版已排除 seed 有缺口的格，A 仍可能在最後階段的格數不足時印 FAIL。B 的 60%／40% 用點估計決策，程式沒有強制兩類盲標對照到齊。建議先驗收預定分片、seed 與對照數，再用區間決策。人工查核目前能衡量偵測器陽性中的真陽性比例，若論文要估所有真實失敗的發生率，還需要掌握漏判率。建議保留這題當補位，先看小規模代理事件及實測吞吐，再決定是否投入預估約 28 GPU 小時的大篩選。 [A 判讀](../killtests/t2i-rare-failure/scripts/aggregate.py#L79) · [B 判讀](../killtests/t2i-rare-failure/scripts/review_stats.py#L20) · [成本與第二週判準](../killtests/t2i-rare-failure/PREREG.md#L88)

### RL 稀疏決策蒸餾

先確認配對確實有 pass@1 與 pass@64 的取捨，再要求混合解碼的上限比 RL 高 ≥2pp、且在相近 pass@1 贏 WiSE-FT，過了才花錢蒸餾。這個停損順序合理。新版已用同一 HF 引擎比較混合解碼和 RL，兩配對擇一的下界也已校正。A 的 GPU 正確性測試已過，閘門 0 已送。現行閘門 0 以每題 128 次的正式 test 評測挑配對，後續若改方案，可先在固定 dev 確認配對與原作重現協定，再以獨立 test 檢定。灰區加碼的資料也要保留給評估。 [PREREG §2–5](../killtests/rl-sparse-distill/PREREG.md#L11) · [提交紀錄](../killtests/SUBMITTED.log#L9) · [A 正確性結果](../killtests/rl-sparse-distill/slurm/logs/rsd-correct-26270.out)

**等 A 的資格結果，再處理 WiSE 比較。** 程式只按 pass@1 把所有點連線，會保留已被支配的點。合成驗證中，pass@1 40%、pass@64 60% 被同時有 50%、80% 的設定支配，但在 40% 時仍拿 60% 當前緣。超出 WiSE 的 pass@1 範圍時，目前也會以端點值算差並容許通過。建議事先定義排除被支配點的前緣與共同範圍外的比較。PREREG 一處說未過即停，下一節允許灰區加碼，應在解讀判準 1 前統一。 [WiSE 前緣與裁定](../killtests/rl-sparse-distill/scripts/aggregate.py#L46) · [灰區規則](../killtests/rl-sparse-distill/PREREG.md#L33)

### Neon 語言版

回報 35 要求三個設定中至少兩個超過調整取樣的基線，語言續寫看 MAUVE +2，數學看相近 pass@1 下的 pass@16 +2pp。把權重方法和調溫度／截斷的基線比較合理。現有程式的主要設定是在 dev 選，test 最佳值已標為樂觀值。建議補一張正式判準表，強制三個種子到齊，為品質匹配事先定容差，以配對差值區間裁定，並說清楚同語料兩種模型規模涵蓋的範圍。現有三組正式結果都未完成，先導數據對取樣參數很敏感。等資產可重建、基線與判讀定好，再以單一設定重驗。 [原提案](../reports/2026-11-14-35-iclr2026-topic-probe.md) · [判讀程式](../killtests/neon-lm/scripts/aggregate.py#L10) · [中斷紀錄](../reports/2026-11-14-58-neon-lm-killtest-interrupted.md)

推理模型 2 位元量化的原切入點已有深查停案建議，也沒有可同批評估的 PREREG，因此目前排在實驗候選之外。 [量化深查](../reports/2026-11-14-31-reasoning-lowbit-quant-deep-dive.md)
