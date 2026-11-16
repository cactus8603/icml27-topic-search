# [codex] 固定方法 × 資料集：公開開發基準 v1

[codex] 2026-11-16（台北）。依使用者要求，後續新方法使用同一組方法家族、資料及評估設定。機械清冊是 [baseline_suite.json](baseline_suite.json)，官方權重／程式來源是 [benchmark_sources.json](benchmark_sources.json)。本版固定比較範圍，並不表示每格都已有可用實作或已完成推論；實際產物與獨立驗收另外記在 `reports/`。所有資料均為已曝光的公開開發資料，`formal_evidence=false`。

既有產物可用性及未執行格的快照見 [baseline inventory](results/2026-11-16-baseline-suite-inventory.json)，之後執行的新結果另加帶日期收據，不改寫此快照或把缺格補成0。

## 固定兩個主要方法家族

| 家族 | 2B | 8B | 選用依據 |
|---|---|---|---|
| CORE | core-emb-2b；core-reranker-2b | core-emb-8b；core-reranker-8b | 直接針對組合語意／否定的官方方法；延續本機基準 |
| Qwen3-VL | Qwen3-VL-Embedding-2B；Qwen3-VL-Reranker-2B | Qwen3-VL-Embedding-8B；Qwen3-VL-Reranker-8B | 官方完整尺寸對照，也是CORE原論文直接比較的強基線 |

每個家族、每個尺寸都分成兩列：**embedding-only** 與 **embedding→同家族同尺寸reranker、每題50個候選**。因此是兩個方法家族、八個系統設定，不能把同一模型的不同尺寸或階段算成八個獨立方法家族。跨尺寸2B→8B／8B→2B仍只屬既有CORE容量拆解，不是本版每輪必跑的額外主格。

CORE沿Qwen3-VL embedding／reranker繼續進行組合訓練，兩家族共享backbone。這是同架構／尺寸下不同訓練方法的對照，不把它們當成兩種獨立架構的泛化證據。

「最強」限定為與本研究相關、具有公開權重／程式且能完整對齊2B／8B的強比較對象，不宣稱它們是所有榜單的全球前二。較新的WeMM是2B／4B／9B，GME／UniME是2B／7B，Jina-reranker-m0只有本次可核的2B；它們不因尺寸缺格而被判品質較差，也不把7B或9B改名為8B。未來若擴大尺寸範圍，另立suite版本，保留原版結果。

CS-CLIP、既有M1、LCSE AND與M2保留為**輔助及歷史對照**，不取代上述兩家族主表。尚未做的LCSE尺寸適配不能只換名稱就算已重現。固定方法不預設8B一定勝2B，也不因某份開發集的贏家而改換比較名單。

## 固定資料與用途

| 資料面板 | 固定規模 | 主要用途與指標 | 每輪範圍 |
|---|---|---|---|
| NegBench negated-dev1000 | 1,000個不同JPEG來源群，各一題；完整5,000圖庫 | **主要開發比較**：來源圖MRR，另列R@1/5/10/50 | 八個設定的目標主表；新方法必須對照同尺寸、同階段的已驗收基線 |
| NegBench plain-dev1000 | 同1,000來源圖、同caption原始位置；同圖庫 | 普通語句保留能力；來源圖MRR／Recall | 四個embedding-only設定；本版未安排plain精排，不能聲稱已控制精排後的retention |
| NegBench full plain／negated | 普通25,014有效、否定25,013有效；同5,000圖庫 | 全量embedding覆蓋；描述性MRR／Recall | 四個embedding-only設定，無全量昂貴精排 |
| FACTOR-Bench | operator1,100、compound145、equivalence450；共1,695比較 | operator／compound各自accuracy與算子細分；equivalence另列分數差 | 四個embedding-only設定的完整句子cosine診斷；不是LCSE重現 |
| 舊COCO AND面板 | 8,191圖、136題、110個已定義正例連通群 | 原類別／面積標註規則下的完整排名mAP、Recall@50 | 四個同尺寸pipeline作補充；保留既有M1／LCSE／M2，樣本較小且標註有已知限制 |
| MCMR原生圖文pilot | 8,320商品、128題 | 商品ID真值的MRR／Recall；單正例AP等於RR | 四個pipeline作跨域開發延伸；不是完整官方MCMR，也不是未見確認 |

固定資料的原始來源、query文字／ID／順序、gallery成員／順序、正解、來源群與檔案SHA全部沿用現有清冊。不重抽dev1000，不為某模型排除失敗題。NegBench唯一空caption按既有規則保留原位置、排除模型輸入並報分母；不能另按長度、正解入池、解析成功或效果挑題。

NegBench／FACTOR與舊AND有COCO來源重疊。各資料集分開列結果，不合成跨集平均mAP，也不把多份衍生面板當成獨立確認。NegBench只標描述的來源圖，不能把其MRR改稱所有AND／NOT條件的完整相關性mAP。舊AND與MCMR小面板不能替代較大主開發集；新的完整多條件真值仍是獨立資料工作。

## 可用資訊、候選與成本

- NegBench／FACTOR／舊AND只允許完整query文字與候選原圖進模型。來源ID、gold／positive lists、objects audit、官方oracle parse只供驗收和計分，不能作條件拆解或候選選擇輸入。新方法若使用query-only parser，必須先固定其程式、prompt、版本與失敗處理。
- MCMR兩臂都保留商品原圖及完整title、description、features、price與Date First Available；不能丟掉metadata後仍當成同資訊的原生比較。產品／domain ID不當語意文字輸入。
- Pipeline先用該家族、該尺寸的retriever排序完整圖庫，前50交給對應reranker；所有query均保留。選入集合依cross分數放最前，尾序保留原retriever順序；同分沿原入池順序。換retriever須重建其候選，不沿用另一模型的top50冒充自身管線。
- 初始開發主對照固定CORE 2B，完整表仍呈現兩族的2B／8B與兩階段，不以表現較弱的精排列作唯一比較。新方法必須呈現對原純向量列及對應pipeline列的差異，不能隱藏精排退步。
- embedding-only是零次cross配對；pipeline固定每query50個配對。2B與8B即使同50次，也不是相同實際成本。分報額外條件編碼、模型／索引載入、影像索引建置、CPU選池、cross推論、端到端實測與峰值顯存；不能把離線cache建置當線上延遲。不同family／尺寸的成本另列。
- 新候選選擇方法若從前200再選50，必須在該輪先固定support及規則，保留相同reranker／batch／精度作配對比較；不把50→200的額外calls藏在方法名裡。跨尺寸混合管線與25／100／200配對為額外成本實驗，另行登記。

## 固定版本與結果流程

1. `baseline_suite.json`釘資料及協定檔SHA，`benchmark_sources.json`釘官方model revision與程式commit。新增方法不能改既有suite；資料或基準變更另升版本、兩版本並列。
2. 新方法先提交config／來源、輸入allowlist、query集合、候選預算、失敗處理與評估方式，再執行。只換新方法，不同時換資料、標註、模型版本或主指標。
3. 所有方法對同一query逐題配對；NegBench dev1000按JPEG來源群bootstrap10,000次、seed2026111602。舊AND沿已凍結連通群；MCMR按來源商品／已核來源群。FACTOR按來源圖聚合，若新增CI必須先另固定方法，不把同圖多題當獨立樣本。
4. 全量NegBench caption平均是描述值，與dev1000的群配對CI分開。重複開發及多重比較明示；CI跨0或排名反轉照實保留，不重抽資料直到得到預期排名。
5. 逐題rank／分數、有效／失敗數、成本與來源收據保存在ignored執行目錄；精簡表及獨立審查只寫`results/`或`reports/`。`NOT_RUN`、技術失敗、已執行、獨立驗收四種狀態分開，缺格不填0，也不能算贏。
6. 本版registry驗證器只驗固定清冊、引用與檔案身份；它不載模型、不重算效果、不證明方法已實作或完成獨立驗收。真正新GPU入口還須驗model／processor／環境／來源／batch並生成綁suite SHA的收據。Qwen與新增8B面板在具備已提交的執行適配及技術smoke前保持待執行，不能借用CORE結果或舊136題冒充新1,000題。

```bash
python3 -B codex/retrieval3090/benchmark_suite.py verify \
  --suite codex/retrieval3090/baseline_suite.json

# 原公開資料全驗收；會讀準備收據列出的來源／清冊／原圖。
dl/retrieval3090/envs/retrieval3090-torch214/bin/python -B \
  codex/retrieval3090/literature_data.py verify
```

固定suite本身不新增GPU工作。已有baseline快取驗收後重用；3090只在合適空閒／低使用率與模型實測顯存餘裕下執行，無卡即排隊。8B不因此降低精度或縮圖假裝同配置；若需要大顯存或訓練則留給叢集。每輪完成驗收後，只刪已核擁有者、已由canonical產物保留的冗餘檔，保留可重用向量／權重、逐題結果與追溯收據。
