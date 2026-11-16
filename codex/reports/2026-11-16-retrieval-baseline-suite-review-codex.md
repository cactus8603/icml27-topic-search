# [codex] 固定baseline suite v1：獨立registry審查

[codex] 2026-11-16 16:40（台北時間）。審查者未實作本次registry／validator。唯讀審查指定JSON、BENCHMARKS、程式／測試與允許的source caches；沒有網路、GPU、模型權重／8B效果／新MCMR效果或封存資料讀取。除CPU測試自行清理的合成暫存外，只新增本報告，未改producer、registry、其他chat的LOCKS／SUBMITTED或scaling檔案。

**PASS，沒有blocking問題。** 此PASS只驗registry身份、比較矩陣與文件邊界，不宣稱全部模型已適配／執行或全部目標格已有研究效果。

## 核對結果

- 兩個方法家族CORE／Qwen3-VL，各2B／8B，各embedding-only／同尺寸pipeline50，恰為**8個設定**。八模型的family／role／scale／revision與各method引用一致；pipeline沒有跨家族／跨尺寸借用模型，embedding維度符合各模型釘版。兩家族共享Qwen backbone關係，不代表兩種獨立架構，也不宣稱全球前二。
- 獨立重建目標格集合：NegBench negated-dev1000有8格；plain-dev1000、full negated、full plain、FACTOR各4個embedding格；舊136題AND及MCMR各4個pipeline格，合計**32格**。Primary固定1,000個JPEG群／完整5,000圖庫；舊小面板與跨域pilot沒有替代主表。
- **17個allowlisted artifact**的SHA／JSON expectations與引用通過。開啟前先核路徑清單，只有公開panels、準備／來源清冊、靜態plans及舊receipt；validator沒有追隨JSON內嵌的其他路徑或讀取8B結果。
- Qwen只完成metadata／官方程式釘版：子審核4個API cache、10個小檔、5個官方code檔的SHA／size／Git blob與固定URL；10個weight entries只核API LFS metadata。四個checkpoint均未聲稱已下載權重，四個Qwen methods明示未實作／載入／評估。未離線重建完整官方commit tree，也沒有重新驗模型payload；這不阻擋metadata registry入版。
- Plain僅有embedding retention控制，未冒稱已控制pipeline後retention；NegBench來源圖MRR／Recall、FACTOR配對accuracy／equivalence gap、舊AND標註下mAP、MCMR商品ID指標分開，不合成跨集平均或混稱Boolean mAP。MCMR須保留image＋metadata；gold／objects／oracle parse不得進模型或admission，query-only新parser另先固定。
- 0次cross與50次cross分開；同50次不等於2B／8B相同實際成本。載入、建索引、額外條件編碼、CPU選池、cross及實測端到端範圍分列。兩個CORE2B階段均保留作anchor，明示不得只對變差的CORE50聲稱改善。
- 14項合成tests涵蓋身份變動、路徑／symlink隔離、嚴格JSON與pointer／型別、同dataset及cost class的baseline引用、CLI唯讀PASS／FAIL，全部通過。另以獨立assert核對實際八設定、32格分布、0／50 budget與同family／size模型配對，全部通過。

## Inventory最後核對

首次讀 `codex/retrieval3090/results/2026-11-16-baseline-suite-inventory.json`：**2026-11-16 16:39:19.195456 +08:00**。其內容只含已授權／既有審查數字，未含新8B效果。

Inventory cell IDs與suite完全相同：**32格＝8 INDEPENDENTLY_REVIEWED＋24 NOT_RUN**；primary八格只有**2 ready／6 pending**，`primary_matrix_complete=false`。Qwen或CORE8B未借舊136題作業補新1,000題，未執行格沒有補零或稱勝出。舊AND／MCMR的CORE2B格只引用既有reviewed study，沒有改稱新執行或獨立確認。全部evidence SHA與現有檔案相符；cheap／CORE50的數字轉錄及既有審查引用一致。

## 實際驗證命令

```bash
python3 -B codex/retrieval3090/benchmark_suite.py verify --suite codex/retrieval3090/baseline_suite.json
PYTHONDONTWRITEBYTECODE=1 python3 -B -m unittest discover -s codex/retrieval3090 -p test_benchmark_suite.py -v
```

第一命令回 `status=PASS, scope=registry-identity-only, formal_evidence=false, artifact_count=17, cell_count=32`；第二命令14 tests全部OK。沒有執行會讀原圖的 `literature_data.py verify` 或任何GPU入口。新family／資料適配仍須提交runner設定、smoke、model／processor／環境／batch收據及獨立結果驗收，不能把本次registry PASS當成上述工作已完成。

## 審查版本SHA256

| 檔案 | SHA256 |
|---|---|
| `baseline_suite.json` | `dc990e521bd387faabb8bf78df2e5cd50e6bd72ef1ca64d541a63e414eb42d9f` |
| `benchmark_sources.json` | `f1ff97f362aca106eabb0fd6778b60d3780ee7be4e71a64bd398e73018222c54` |
| `BENCHMARKS.md` | `e6c20330ecc478662b058aff3bc0ae35d72d8fba471693cb9b4dec9e0801b801` |
| `benchmark_suite.py` | `ccf546d1cdf2330af7a942de293242f60dd2ea0a04f76f93120118c9df893bab` |
| `test_benchmark_suite.py` | `dc714684d8bda7593a1cb868ec1284062fac74669696f0388c409180a609a53f` |
| baseline inventory | `d092bf634cf7c13680333e6c04cc1a978692f489705096639b3b907b1fe51e33` |
