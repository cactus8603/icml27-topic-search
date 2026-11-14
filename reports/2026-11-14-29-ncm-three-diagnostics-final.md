# NCM 免訓練診斷：最終回報（E1、E2、E3 全部完成）

完整表格、原文與所有 CI 都在 REPORT.md。下面只列主要數字。

## 輸出位置
- 根目錄：/work/ChihChungHsu/cactus8603/icml/ncm-diag/
- 完整報告：REPORT.md（由 scripts/make_report.py 從原始 JSON 產生）
- 原始結果：
  - E1：results/e1/full/{small,medium}/cond_{a,b,c,d,e,d1,d2}.json（另有 .pkl 與 config 複本）
  - E2：results/e2/full/e2_{small,medium}.json 與 e2_{small,medium}_perchunk.npz
  - E3：results/e3/full/e3_{small,medium}.json
- 腳本：scripts/（ncmlib.py、e1_ablate.py、e2_predictability.py、e3_decompose.py、job.slurm、make_report.py）
- log：logs/ncmd-full-26142.out、logs/ncmd-smoke-26140.out；正式指令在 logs/full_command.txt

## Slurm 與 GPU 時數
- smoke job 26140：48 秒，n=32，驗證腳本。
- 正式 job 26142：Elapsed 00:52:44，約 0.88 GPU 小時，E1、E2、E3 的 small 與 medium 依序跑完。
- 合計約 0.89 GPU 小時，預算 8 小時。partition h200q，1 GPU。
- 正式指令：`python e1_ablate.py --size small --tag full && ... --size medium ... && python e2_predictability.py --size small/medium --tag full && python e3_decompose.py --size small/medium --tag full`，由 scripts/job.slurm 包裝，環境 envs/c5。

## 驗證
- E1(a) 重現與先前一致：small 93.17、medium 88.24。樣本是 5 seed（42,0,1,2,3）x 256 = 1280，先前重現已 ≥1024，所以沿用。
- E2 的 teacher forcing 與 generate() 路徑對照：層 1–11 的 logits 最大絕對差 ≤ 1.8e-4（前 8 個 chunk）。這是用 spy 攔截 `_sample_level`、回傳真實碼做的，可排除條件洩漏。

## E1 解碼懲罰消融（Gen PPL 為 GPT-2-large；entropy 為每樣本 128 個 BERT token 的 unigram 熵，nats）
| 條件 | small PPL（sd） | small ent | medium PPL（sd） | medium ent |
|---|---|---|---|---|
| (a) 官方預設 | 93.17（2.43） | 4.3140 | 88.24（1.35） | 4.3390 |
| (b) 關 frequency_penalty | 78.27（1.81） | 4.1592 | 74.37（1.24） | 4.2309 |
| (c) 關 adjacency_penalty | 93.27（2.59） | 4.3039 | 87.72（1.32） | 4.3324 |
| (d) 關 allowance+deficit | 328.55（8.70） | 4.7193 | 309.41（3.39） | 4.7264 |
| (e) 全關 | 77.35（1.91） | 4.1143 | 73.38（1.39） | 4.2117 |
| (d1) 只關 deficit（補充） | 100.26（3.39） | 4.3445 | 91.83（1.25） | 4.3598 |
| (d2) 只關 allowance（補充） | 328.55（8.70） | 4.7193 | 309.41（3.39） | 4.7264 |

- 「關掉」的依據（eval_shard.py 行號）：
  - freq：第 38 行讀取，第 100 行與第 113 行，`--freq 0` 即關。
  - mu／deficit：第 39 行與第 116 行。
  - adj：第 40 行與第 80 行，傳空字串得 ADJ=None。
  - allowance：第 69 行，config 設空字串得 ALLOW=None。
- 第 116 行的 deficit 項在 `freq > 0` 區塊內，所以關 frequency_penalty 時 deficit 與 allowance 也同時失效。
- (d) 與 (d2) 數字完全相同，因為 allowance 一關，deficit 就不會作用（第 116 行要求 allowance 非空）。
- 前 5 筆原文（seed 42）每條件都在 REPORT.md 與 JSON 的 `first5_texts_metric_view`，是 raw_ids 以 BERT 解碼（保留 [CLS]）的結果，即指標實際評的文字。

## E2 各層 teacher-forced CE（nats／code，5000 個 chunk 評測，66,825 個 chunk 對照，bootstrap 1000 次）
下表為 generator CE，以及 generator 減層 unigram 的差值（括號為 95% CI）。層 unigram 是 add-0.5、V=8192。
| 層（長度） | small gen CE | small 差值 vs 層 unigram | medium gen CE | medium 差值 |
|---|---|---|---|---|
| 0（1） | 7.868 | +0.334 [0.311, 0.357] | 7.487 | -0.047 [-0.053, -0.041] |
| 1（8） | 8.448 | +0.361 [0.353, 0.370] | 8.079 | -0.007 [-0.008, -0.005] |
| 2（16） | 8.651 | +0.098 [0.090, 0.106] | 7.802 | -0.752 [-0.762, -0.741] |
| 3（24） | 8.259 | -0.258 [-0.267, -0.249] | 7.329 | -1.188 [-1.200, -1.177] |
| 4（32） | 7.397 | -0.882 [-0.893, -0.871] | 6.224 | -2.054 [-2.068, -2.041] |
| 5（40） | 6.637 | -1.297 [-1.308, -1.284] | 5.439 | -2.495 [-2.508, -2.481] |
| 6（52） | 5.259 | -2.431 [-2.444, -2.418] | 3.868 | -3.822 [-3.835, -3.808] |
| 7（64） | 2.725 | -4.790 [-4.801, -4.780] | 1.659 | -5.857 [-5.867, -5.848] |
| 8（76） | 3.680 | -3.714 [-3.729, -3.702] | 2.067 | -5.328 [-5.339, -5.317] |
| 9（92） | 3.942 | -3.418 [-3.431, -3.404] | 2.121 | -5.239 [-5.249, -5.228] |
| 10（108） | 4.476 | -2.965 [-2.977, -2.952] | 2.314 | -5.127 [-5.136, -5.117] |
| 11（128） | 1.023 | -6.012 [-6.020, -6.004] | 0.468 | -6.566 [-6.572, -6.561] |

- 層 unigram CE 在各層為 7.534、8.086、8.553、8.517、8.279、7.933、7.690、7.515、7.394、7.360、7.441、7.035，兩個模型共用，因為對照只依賴 tokenizer 與資料。
- 按位置 unigram 的 CE 與差值、以及各層 unigram 熵都在 REPORT.md 的 E2 表。
- 位置 unigram 的差值與層 unigram 的差值只差 0.01–0.15。

## E3 品質損失分解（n=1280 chunk，生成 seed 42 與 0 的 Gen PPL 平均；k=12 為確定性，只跑 seed 42）
原始 validation 文字本身的 GPT-2-large PPL = 53.92，entropy 4.3145。

| k | small 預設 PPL | small 預設一致率 | small 全關 PPL | small 全關一致率 | medium 預設 PPL | medium 預設一致率 | medium 全關 PPL | medium 全關一致率 |
|---|---|---|---|---|---|---|---|---|
| 0 | 94.28 | 0.024 | 78.43 | 0.026 | 87.85 | 0.024 | 73.97 | 0.025 |
| 4 | 104.30 | 0.074 | 87.64 | 0.081 | 97.38 | 0.114 | 80.73 | 0.123 |
| 6 | 120.83 | 0.290 | 106.16 | 0.314 | 95.50 | 0.461 | 81.33 | 0.490 |
| 8 | 104.58 | 0.701 | 91.46 | 0.743 | 80.21 | 0.817 | 67.52 | 0.859 |
| 10 | 81.91 | 0.869 | 70.07 | 0.913 | 71.70 | 0.914 | 59.89 | 0.957 |
| 11 | 74.39 | 0.912 | 63.27 | 0.954 | 68.86 | 0.934 | 57.49 | 0.976 |
| 12 | 65.36 | 0.957 | 54.21 | 0.999 | 65.36 | 0.957 | 54.21 | 0.999 |

- 這張表是精簡版：k=1,2,3,5,7,9 的兩個 seed 個別數字、entropy 與「非 [CLS]」一致率在 REPORT.md。
- small 在 k=6 的預設 PPL 最高（約 120.8），比 k=0 的 94.3 高。
- k=12 的預設懲罰條件 PPL 為 65.36，高於原文的 53.92；懲罰全關的 k=12 為 54.21。
- k=12 在 small 與 medium 相同，因為共用 tokenizer。

## 與 brief 不一致或沒能確認的事（REPORT.md 有完整 10 條，這裡列要點）
1. 「641 個 code」是各層長度總和，也就是 code 位置數；每層詞表是 8192。
2. 資料不是單句，是 71,825 個 packed 128-token chunk，內含多個以 [CLS] 串接的句子。E2 與 E3 的「句子」都指 chunk。E2 是隨機 5000 個 chunk（seed 0）當評測集，其餘 66,825 個當對照估計集（不重疊）。E3 用 1280 個 chunk（seed 123）。
3. 官方 `hyps`（在 EOS 截斷）幾乎都是空字串，因為樣本都以 [CLS] 開頭。官方指標實際評的是 `raw_ids`。我的「前 5 筆原文」用 raw_ids 解碼，截斷版另存在 `first5_hyps_official_dec_cut_at_eos`。
4. 釋出程式碼沒有任何 conditioning corruption 實作（搜尋 `corrupt` 在所有 .py 零命中，只在 config 的 `code_corruption` 設定裡出現）。推論與 teacher forcing 都是乾淨條件，所以 E2 只有一組數字，沒有另跑「官方推論設定」那組。
5. 官方生成的第 0 層用 generator.pt 內存的 level0_unigram 抽樣，不用 generator 的第 0 層頭。E2 的第 0 層 CE 是該頭的 teacher-forced 值，官方生成不使用。E3 的 k=0 用 unigram 抽樣，k≥1 用真實第 0 層碼。
6. E3 的預設條件在所有 k（含 k=12）都對 decoder 輸出套懲罰，因為官方流程如此；全關條件是 argmax。
7. E1(d) 我定義為 allowance 與 deficit 同時關，另補了 d1 與 d2。
8. E2 的位置 unigram 內插權重 lambda 在各層都選到格點上界 1024（格點為 1、4、16、64、256、1024）。最佳值可能在格點之外，所以另附純 add-0.5 版本。
9. E3 只用 2 個生成 seed，沒做更多。
10. 沒有 git commit，也沒有修改任何資產目錄。
