# [codex] 文獻公開開發資料：full cheap 結果獨立複核

[codex] 2026-11-16 15:12（台北時間）。審查者未實作 `literature_dev.py`／`literature_data.py`。只讀本輪已完成的公開資料、canonical vectors及cheap報表；無GPU、無改碼／選法／追加實驗、無讀cross效果、private或封存資料。核算不取得main的writer lock，沒有阻擋後續stage。

**結論：指定範圍內全部核算通過，沒有發現排名、指標、bootstrap或FACTOR分母的阻擋錯誤。** 這仍是公開開發資料上的固定checkpoint比較；沒有新增正式確認，也不是新admission方法或LCSE的效果驗證。

## 首次效果讀取與凍結來源

- `main/report.json`：2026-11-16 **15:09:18.482733 +08:00**。
- `main/per_query.json`：2026-11-16 **15:09:18.483176 +08:00**。
- 讀取前已確認 `index-meta.json` 為COMPLETE且兩報表都存在。之前只讀technical progress、plan／source／prepared panels；未讀未完成vectors或任何cross檔案。
- Producer執行來源為`cbd545ac96e4978454c41a78f78b75419d413b77`。四個plan source-file SHA及plan bytes均與該commit一致；index記錄的實際Git HEAD為`031b21f9f26612d169682e7ab03a663e7173013f`，其source-file SHA仍完整符合plan，不把後續docs/log commit誤當方法改動。

| 固定產物 | SHA256 |
|---|---|
| `literature_plan.json` | `f0685e544409d0082f642d3ad36ce5b85323465f7159c1880549f396093fb2e2` |
| `panels/preparation.json` | `86814f40da7a74c7f606cc99ccc4d6bae0a7229ab1eb1008b884cb885638525a` |
| `main/index-meta.json` | `52325472152758089383733d599b39a2feaff7d9dd547653ca40c708f17408c6` |
| `main/report.json` | `7942b6d14fe28bc1c52d6a6f8eb6d6b56f3294e66d92ed9d55659cfe1d5ee6eb` |
| `main/per_query.json` | `d6cc5d3d231318d5cdfcfdc494eeeb4b518c3801485f99b860b439d3903ce747` |

全部prepared panel SHA與receipt一致；4份NPY及state的SHA、COMPLETE rows／shape／float32、finite與unit norm檢查通過。CORE維度2,048、CS維度512；image rows5,000，sorted unique text rows52,775；text-table與從所有有效query及FACTOR captions自行重建的排序表完全相同。index identity的plan／source-files符合凍結記錄，`formal_evidence=false`。

## 獨立核算範圍與做法

未呼叫producer的排名、metrics、bootstrap或FACTOR計分helper。先以純合成矩陣驗證同分順序與已知常數bootstrap，再直接讀canonical float32 NPY，以同一全圖庫dot products重建dev排名：

`rank = 1 + count(score > source_score) + count(score == source_score 且 gallery位置早於source)`。

這與producer的全排序演算法不同，保留「同分按原numeric-image-ID圖庫順序」規則。兩checkpoint×plain/negated×1,000，**4,000個source ranks逐條零差異**；五項metrics均吻合。四組paired contrast另按排序後JPEG群、seed2026111602、10,000次有放回抽樣重算，percentile95%CI和sample SE（ddof=1）逐項吻合，容差1e-10百分點。

Full部分獨立核對query資格、IDs／順序／分母，並從完整per-query ranks重新計算平均metrics；**沒有重新對全部full queries計算所有dot products**。FACTOR全1,695 items的兩caption cosine、判斷、各test-type／operator彙總均直接由canonical vectors重算吻合；gap容差1e-7。

## NegBench結果與有效分母

同一5,000圖庫；只有原始空caption `negbench:negated:301061:0` 事前排除，保留其原caption位置，未做gold／模型分數篩選。Plain full25,014/25,014有效；negated full25,013/25,014有效。Dev兩版均1,000有效、1,000來源圖／互異JPEG群，source image、caption index和group逐對相同。Full兩個平均不是同分母的paired估計。

全部指標以百分比表示：

| 範圍 | Checkpoint／文字 | MRR | R@1 | R@5 | R@10 | R@50 |
|---|---|---:|---:|---:|---:|---:|
| dev1,000 | CORE plain | 66.21684 | 55.60 | 79.30 | 87.20 | 97.20 |
| dev1,000 | CORE negated | 65.16088 | 54.40 | 79.70 | 85.30 | 96.90 |
| dev1,000 | CS plain | 57.16629 | 44.50 | 72.50 | 81.70 | 97.00 |
| dev1,000 | CS negated | 53.36044 | 40.80 | 67.10 | 77.40 | 95.30 |
| full | CORE plain | 66.84426 | 56.07260 | 79.99520 | 87.11122 | 97.43344 |
| full | CORE negated | 65.29763 | 54.30376 | 78.43521 | 85.91532 | 97.11750 |
| full | CS plain | 57.94464 | 45.69841 | 72.69129 | 81.89814 | 96.35004 |
| full | CS negated | 53.67821 | 41.17459 | 68.33647 | 78.56315 | 95.09055 |

Dev paired bootstrap代表數字（差值與區間均為百分點，right−left；五項指標的完整CI／SE均已核算）：

| Contrast | ΔMRR［95% CI］ | ΔR@1［95% CI］ | ΔR@50［95% CI］ |
|---|---:|---:|---:|
| CS−CORE，plain | −9.05055［−11.36661, −6.72305］ | −11.10［−14.30, −7.80］ | −0.20［−1.30, 0.90］ |
| CS−CORE，negated | −11.80044［−14.24351, −9.37316］ | −13.60［−16.90, −10.30］ | −1.60［−2.90, −0.30］ |
| negated−plain，CORE | −1.05596［−2.11242, −0.00907］ | −1.20［−2.90, 0.50］ | −0.30［−1.00, 0.30］ |
| negated−plain，CS | −3.80585［−5.36727, −2.30728］ | −3.70［−6.00, −1.40］ | −1.70［−2.80, −0.60］ |

CORE negated−plain MRR的CI上界極近0，且本輪有多項探索比較、未多重比較校正，宜描述小幅下降。Dev CORE negated已有969/1,000來源圖入top50；同一圖庫中，單純擴大候選預算可再增加的來源圖入池Recall最多3.1百分點。這不代表top50內排序已良好，也沒有替代尚未讀取的cross結果。

CS實際tokenization audit：dev plain／negated均沒有超77-token query；full plain0條、full negated2條超限，未依長度篩除。故dev差距不能歸因於這個77-token截斷事件。這是audit記錄的核對，沒有另執行tokenizer／模型。

## FACTOR：配對診斷邊界

| Test type | n | CORE正確率 | CS正確率 | 核算邊界 |
|---|---:|---:|---:|---|
| operator | 1,100 | 63.00000%（693） | 63.90909%（703） | a/b、row-weighted micro average；不是五算子百分比的等權平均 |
| compound | 145 | 95.17241%（138） | 82.06897%（119） | a/b，與operator分開；CORE僅7個錯例 |
| equivalence | 450 | 無binary accuracy | 無binary accuracy | 全部`both→None`；平均absolute gap分別0.02471659／0.01927922，exact ties均0 |

五個operator的核算亦完全吻合：CORE的NOT／AND／OR／BUT_NOT／NEITHER為61.66667／82.0／20.0／74.8／67.2%；CS為48.0／96.66667／70.66667／60.0／63.2%。這是完整句子的holistic cosine配對診斷，**不是使用oracle parse的LCSE重現**。Equivalence gap不能改稱binary accuracy；不同checkpoint分數尺度下的平均gap也不單獨證明誰更會邏輯。

CS在AND配對較強，卻在本輪來源圖檢索MRR較弱，說明兩項測量並不等價。不能將這個跨架構／規模checkpoint比較，解讀為CS fine-tuning本身的因果效果；本輪沒有同backbone base-CLIP對照。各FACTOR差異也未估計群組CI，同圖多題不能直接當互相獨立樣本。

## 可支持與不能支持

可支持：本輪固定公開開發資料上，CORE的來源圖MRR高於CS；兩者在原始caption加入否定／改寫後的MRR下降，CS下降幅度較大；不同operator／compound診斷呈現不同強弱。數字、分母與預定條件bootstrap均可由保存產物獨立核算。

不能支持：完整Boolean relevance／full-match AP的優劣、純粹否定詞的因果效應、LCSE或新入池政策的收益、任意資料域的一般化、正式獨立確認，或cross reranker的效果。官方GT只標來源圖，並未窮盡其他語義正例；plain→negated同時包含LLM改寫差異；JPEG群bootstrap條件於此開發集，未完整建模共享gallery／語義近重複。所有公開資料仍可繼續開發，但後續選擇不能回寫為本輪已預先確認的主張。

**驗收狀態：PASS，限本報告所述cheap範圍與核算深度。** 未讀或驗收cross效果、Half-Truths、MCMR新效果或封存確認資料。
