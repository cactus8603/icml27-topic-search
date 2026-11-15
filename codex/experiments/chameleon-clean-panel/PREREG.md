# Neural Chameleons 乾淨資料面板對照

同一個固定的公開第 12 層 probe，在本地凍結面板與作者公開 pipeline 可重建的 medium 候選面板，是否有不同的乾淨 TPR？沿用正式重現閘門的狀態，這次只量可公開重建的面板選樣與標籤共同變化。原作者歷史評估的選樣 ID 與 merged probe 仍待取得。 [資料來源稽核](../chameleon-source-audit/PROTOCOL.md)

## 固定量測

- 兩面板原各 1,000 正／1,000 負，合併同一公開來源中的精確配對後，對 2,826 個去重列只評一次。共有 1,174 個重疊配對，57 個在兩個面板標籤不同。以原面板標籤回填同一分數。使用既有 Gemma-2-9B snapshot、同一 chat template 與第 12 層 demo probe。左側 padding 與顯式 position ID，BF16 eager backbone，FP32 計算各回應 token sigmoid 後平均。兩面板共用模型、排序與 batch 16，使用原模板核對 2,826 列皆無截斷或外層 prompt 標記。只做乾淨 teacher forcing。
- 各面板的主描述量是用自己乾淨負例取排序第 `floor(0.99*n)`（0-based），以嚴格 `score > threshold` 計算 TPR、實際 FPR，回報平手筆數。每面板 1,000 負例時沿用作者第 991 順位，嚴格大於者至多 9 個。另交換兩面板門檻看其移轉性。共享 1,174 列另依兩套原標籤計各自門檻，描述在固定題目上的標籤敏感度。
- 用共同原列分層配對重抽 1,000 次，種子 `2026111537`。 strata 為兩臂成員資格及原標籤的聯合組合。每次對同一共享列沿用同一抽中次數，維持兩面板各類別大小，重估每個面板的門檻。報個別原法 TPR、門檻及兩面板原法 TPR 差的中央 95% 區間，解讀限此固定來源和此公開 probe。

## 執行

只用叢集已備好的本地模型與 TEST JSON。CPU 預檢驗權重索引與四個 shards 的 safetensors headers、探針形狀／有限值、模板、runtime 及面板綁定。獨立 smoke 僅檢查 16 列含最長序列的前向可行性，收據只含計時、顯存和身分欄位。主執行需要同一凍結版本的成功 smoke 收據。只寫聚合 JSON。

一張 H200、8 CPU、48 GiB RAM，360 tokens 內的 batch 16 估 25–40 GiB GPU、2–8 分鐘，作業上限 20 分鐘。耗時與 VRAM 以首次 smoke／主執行實測。用輸入內容與標籤獨立的 digest 凍結資料。

```bash
stage=codex/experiments/chameleon-clean-panel
../icml/envs/inoc/bin/python -IB "$stage/test_measure.py"
../icml/envs/inoc/bin/python -IB "$stage/measure.py" preflight
bash "$stage/run.sbatch" FULL_FREEZE_COMMIT preflight
```

## 讀取紀錄

- [codex] 2026-11-15 11:05（台北），Codex 已讀 `26572.json`。凍結版本 `5bff580`。
- [codex] 2026-11-15 11:04（台北），獨立子代理已獨立讀取並複核同一份彙總。
- [codex] 2026-11-15 12:13（台北），Codex 重讀 `26572.json` 的共有列彙總，核對兩套標籤分母與各自門檻。
