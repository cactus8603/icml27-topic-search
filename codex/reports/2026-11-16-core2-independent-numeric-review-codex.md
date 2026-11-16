# [codex] CORE2 獨立數值審查

[codex] 2026-11-16 23:45（台北）  
Reviewer：`/root/independent_numeric_review`

**兩個已授權 job 的獨立原始陣列重算均 PASS，與原 mechanical 指標的差異符合固定 `1e-12` tolerance。** 這個 PASS 只表示數值重算相符；使用者最終決策仍為 `PENDING`。沒有執行 GPU、重新 inference、修改 producer／analysis 來源或計畫，亦未讀其他 job 的結果。

## 凍結、順序與讀取記錄

- Analysis full ref：`7b4855f5d1b7f6e6c1f3850b8a2a9e41b5ac82ed`。
- Approved plan：`codex/retrieval_matrix_analysis/plan.analysis.approved.v1.json`，SHA256 `c201108db0f6cc5d923f07aa16f2d26b77588656ce38507821aebd3fb42576fe`。
- 先確認 root 的 original mechanical COMPLETE，再分別執行凍結 `run.sh independent --accept-independent-role`。兩個正式獨立階段均 exit 0，state PASS；沒有失敗重跑。
- 我未實作這些 producer、評估或分析程式，角色與 mechanical actor／實作者排除列表均重新驗證。
- 正式陣列讀取與下方 receipt 摘要讀取各有獨立 READ_STARTED／READ_COMPLETED，記錄於 analysis PREREG 與外部 durable journal。未改動原 producer PREREG。

| Job | Producer full ref | Independent receipt SHA256 |
|---|---|---|
| `fixed-embedding-core2` | `99431594893bba94337b3befd10a7a483ba7374f` | `c5b312f3ea2332a7bcae146df8f6b84bfc7ade6f1ce59afbccb72077bca33693` |
| `mcq-core2` | `25f7bd515875a0c698202d0047d7b7952fd6846b` | `1ddf6e2d34c6ac8eea58c037180cc06a890967cc28645fa5862457590d590e71` |

## Embedding：原官方三資料集

保留 COCO／Flickr 的兩方向各 7 個官方 IR 欄位，以及 NegBench 官方 R@5；完整欄位皆獨立重算相符。以下列出凍結 panel qualification 使用的三個官方指標，數值為 fraction，差值單位為 percentage points（pp）。

| Dataset | 官方欄位 | 獨立重算 | 凍結 target | 差值 pp | 絕對差 ≤ 1 pp |
|---|---|---:|---:|---:|---|
| COCO | `coco_t2i_recall_5` | 0.792556169 | 0.793000000 | -0.044383 | 是 |
| Flickr | `flickr_t2i_recall_5` | 0.961200000 | 0.962000000 | -0.080000 | 是 |
| NegBench COCO | `negbench_coco_recall_5` | 0.765811146 | 0.766000000 | -0.018885 | 是 |

三個固定對照中，3/3 落在原先 ±1 pp 範圍。此處只記錄固定數值條件，不代替使用者定案。

| Dataset | 原完整 queries | 原完整 gallery | 數值 agreement |
|---|---:|---:|---|
| `coco` | 25,014 | 5,000 | PASS |
| `flickr` | 5,000 | 1,000 | PASS |
| `negbench` | 25,014 | 5,000 | PASS |

- 使用原 BF16 GPU score caches、原 native topk admission；完整 items／model／settings fingerprint、原 batch 邊界與 bytes SHA、array/state receipts 均驗證。沒有用新的 CPU dot 或 stable FP32 full-gallery ranking 取代原官方分數。
- COCO／Flickr 的 I2T recall 為 relevant-caption fraction；AP@10 以全部 positives 為分母；同分採原 evaluator 的 document-ID ordering；所有原 query 與已登記空 caption 保留。

完整數值與逐資料集證據：[embedding independent receipt](/ssd4/cactus8603/icml/dl/retrieval-matrix-analysis/results/fixed-embedding-core2/independent-review.json)。

## MCQ：全部三個官方 benchmarks

全部 15,912 題、58,891 pairs、8 subsets 保留；原 pair batch 為 4。COLA 使用 text-choice 兩個 strict comparisons；SCPP 為五 subsets 等權 macro；NegBench 為 COCO／VOC 等權 macro。Ties fail，沒有 row-weighted pooled macro。

| Benchmark | 完整題數 | 獨立 macro accuracy | 凍結 target | 差值 pp | 絕對差 ≤ 1 pp |
|---|---:|---:|---:|---:|---|
| `cola_text_choice` | 210 | 0.723809524 | 0.724000000 | -0.019048 | 是 |
| `negbench_mcq` | 10,945 | 0.645620067 | 0.646000000 | -0.037993 | 是 |
| `scpp` | 4,757 | 0.838187208 | 0.839000000 | -0.081279 | 是 |

三個固定對照中，3/3 落在原先 ±1 pp 範圍。三個 benchmark 都保留，沒有依觀察結果換掉未達範圍的項目。

| 原 subset | 完整題數 | 獨立 accuracy |
|---|---:|---:|
| `coco_mcq` | 5,914 | 0.507101792 |
| `cola` | 210 | 0.723809524 |
| `replace_att` | 788 | 0.880710660 |
| `replace_obj` | 1,652 | 0.970338983 |
| `replace_rel` | 1,406 | 0.760312945 |
| `swap_att` | 666 | 0.869369369 |
| `swap_obj` | 245 | 0.710204082 |
| `voc_mcq` | 5,031 | 0.784138342 |

- 逐一核對所有原 examples／pairs 的唯一性與順序、9425 張原 images 的映射與 SHA／bytes，以及原 score/state 及逐 batch fingerprint。
- 原 checkpoint、source、model spec、actual runtime、計畫與完整 array receipts 均保持原凍結來源。MCQ agreement 與各 target 差值是不同判準；沒有把單純重算 agreement 當作 empirical qualification。

完整數值與證據：[MCQ independent receipt](/ssd4/cactus8603/icml/dl/retrieval-matrix-analysis/results/mcq-core2/independent-review.json)。

## 本次讀取事件定位

| Job | 正式獨立讀取 START hash | Receipt 摘要讀取 START hash |
|---|---|---|
| `fixed-embedding-core2` | `cf6bc98e1d08db41287568fe8b905e0e87b95e491d2095465c6a692b3372cbe2` | `1452c20c1783cf54db12739397f83776be1e351f3e5b24b1c8d5d40b33c25437` |
| `mcq-core2` | `149653d7f1b312f17ca45ac03750aae757af4743a003e746151a4e747c1a3ca1` | `5e36903d498411f83eb978a47d2e80b566cbda3c2fcfce8587fc30e0064faba2` |

只處理以上兩個明確授權 job；MCMR 與其他模型的效果未讀。科學判讀與後續採用由使用者定案。
