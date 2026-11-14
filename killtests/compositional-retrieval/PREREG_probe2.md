# PREREG_probe2：停損後第二探針（整句條件 PMI s~）

2026-11-15 [codex] 依第 60 號報告 §7，原判準與輸出保留供追溯。第 5 節及舊輸出 `rejection_text` 的方向淘汰解讀已撤回；新的補量與題目判讀見 [PREREG_headroom.md](PREREG_headroom.md)。

**狀態：停損後第二探針；同一批 1437 列（Half-Truths 三子集）已看過 stage1 結果；凍結後只跑一次。**
凍結 = 協調者 commit 本檔、`configs/params_probe2.json` 與 `scripts/{probe2_condpmi,probe2_embed,aggregate_probe2,unit_tests_probe2}.py`、`slurm/probe2.sbatch`。凍結前不得對真實資料執行主判定（`probe2_condpmi.py` 的 CLI 與 `aggregate_probe2.py`）。凍結後的任何修改須另記日期並標為探索性。
設計來源：`reports/2026-11-14-57-retrieval-fail-deep-judgment.md` 第 3 節。本檔不改設計，只把它寫成可機械執行的規則。stage1 的 PREREG.md、VERDICT_stage1.md 與 results/ 既有檔案不變。

## 1. 資料、模型、評分
- 資料、模型、bootstrap 設定與 stage1 相同：HT 三子集共 1437 列；CLIP ViT-B/32（`clip_b32`）與 CS-CLIP B/32（`cs_clip`）；n_boot = 10000，bootstrap seed 20261114（取自 `params_probe2.json: primary.boot_seed`，與 stage1 `gate_g3.boot_seed` 同值）。沿用 stage1 已存的 `results/ht/*.npz`（整句 embedding 與影像 embedding），不重算。
- 評分：對整句 T（full、anchor、half 各一，**不拆單元**）
  `s~(T) = tau * s(I,T) - log mean_{I' in 池} exp(tau * s(I',T))`，tau = 模型 `logit_scale.exp()`，s 為 cosine。
- 池：與 stage1 PREREG §3 相同，HT 三子集所有唯一影像（依 sha256 去重）共 1379 張，不使用任何標籤。
- COR(s~) = 逐列 `s~(full) > s~(anchor) > s~(half)`（嚴格大於，平手算錯）。COR(holistic) 為 stage1 同定義（整句 cosine），同一批列。

## 2. 唯一主指標與三態
指標：**CS-CLIP 在 COCO(Qwen)（n = 509）上，COR(s~) − COR(holistic) 的配對差（pp）**；配對 bootstrap（列層級），單尾 alpha = 0.025（下界 = 2.5 百分位、上界 = 97.5 百分位，欄位 `lo_1s`／`hi_1s`）。
- **PASS**：點估計 >= +5pp，且單尾下界 > 0。
- **FAIL**：單尾上界 < +5pp。
- **灰區**：其餘；不加碼、不補資料，交使用者決定。
只有一個主指標，不做多重性調整（alpha 即 0.025，沿用設計文件）。

## 3. 構造閘門（任一失敗 = 記為「未能測試」，不輸出 PASS／FAIL，也不得據以淘汰）
閘門都在主指標同一模型與子集（CS-CLIP、COCO(Qwen)、n = 509）上評估。
1. **正控制 >= 70%**：把 half 句中的 d− 換成「別列」隨機取的細節，其餘不變，記為 ctrl；`COR_pos = P(s~(full) > s~(anchor) > s~(ctrl))` 要 **>= 70%**。
   - 替換細節 = 同子集內另一列的 d+ 文字（`extract_detail` 的輸出）；候選須為不同列、非空，且字串**不得與本列 d+ 相同**。
   - 隨機：seed **20261115**；每個子集用 `numpy.random.default_rng([20261115, 子集索引])`（索引依 coco_qwen、coco_mistral、cc3m_qwen 為 0、1、2），依列序逐列，從合格候選（依列序排列的清單）以 `rng.integers(0, 候選數)` 抽一個。與模型無關（兩模型用同一組文字）。
   - 句子構造用 `probe2_condpmi.replace_detail`：對 anchor 與 half 做 token 層級 difflib；第一個含字母數字的新增／替換段換成該細節，其餘含字母數字的新增段刪除，其他段保留；若無新增段則接句尾。
2. **負控制 <= 20%**：把影像打亂（同子集內的排列，無不動點）後，三個句子都用打亂後的影像算 s~，`COR_neg` 要 **<= 20%**（隨機排序的期望 1/6≈16.7%，n=509 的標準差約 1.65pp，20% 約為期望 +2 SE；凍結前由協調者改，原草稿 15% 在純隨機下約 84% 機率失敗）。池不變。
   - 隨機：seed **20261116**；`numpy.random.default_rng([20261116, 子集索引]).permutation(n)`，遇不動點則與下一位交換（見 `neg_control_perm`）。單次，不平均。
3. **成對正確率上限 > 約 63%**：因 COR 要求 full > anchor > half，必有 `COR(s~) <= P(s~(full) > s~(half))`（成對正確率，嚴格大於）。要主指標 PASS，COR(s~) 須 >= COR(holistic) + 5pp；故成對正確率須**嚴格大於 `COR(holistic) + 5pp`**，此處 COR(holistic) 為本探針同批資料實測點估計（stage1 實測 CS-CLIP COCO(Qwen) 為 58.15，換算門檻 = 63.15）。不是固定 63，而是機械換算；通過條件為 `pairwise > COR_holistic + params_probe2.gates.pairwise_margin_over_holistic_cor_pp`。未通過 = 「構造不可行」，狀態記為未能測試。
4. 控制組 embedding 缺失（正控制資料未產生）視為閘門失敗。

## 4. 只描述、不參與判定
- CLIP B/32 上的同一指標（含三個子集與兩個模型的 COR、配對差、成對正確率、控制組 COR）。
- 池換成 COCO val2017 全部 5000 張影像（`pool_coco5000_<model>.npz`）時，主指標的敏感度（COR、配對差、成對正確率）。
- B 做法（VLM 條件 LLR）：**未做**。若日後做且 COR >= 90%，觸發 46 號報告的旗標；本探針不判定。
- 跑前預檢（已完成，CPU，見 `results/probe2/precheck.json`）：holistic 成對正確率（兩模型三子集）與邊際 PMI（stage1 單元 PMI）在最佳全域門檻下的 COR 上限；只作描述。
- 全部子集與模型的 `pos_control_cor`、`neg_control_cor`。

## 5. 若 FAIL 的淘汰理由寫法
僅在三閘門都通過且主指標 FAIL 時才可淘汰，理由逐字為：「免訓練、固定索引的雙塔證據聚合，邊際和條件兩種形式都贏不了 CS-CLIP；剩下的路線已被 CS-CLIP 和 CORE 佔住，沒有可區隔的切入點」。若是正控制（或任一閘門）失敗，記為「未能測試」，不淘汰。灰區交使用者，不加碼。PASS 不等於題目成立，只代表可以進一步考慮（需另訂規則）。

## 6. 代理落差（規則 4）
本探針與真正要問的問題（單向量、固定索引檢索能否以證據聚合超越 CS-CLIP）之間的落差：
1. 評分只是「每個文字用同一批影像池做正規化」的整句 PMI；它不是真正的 p(I|T)，也不是 B 做法的 VLM 條件 LLR（B 未做）。
2. 任務是 Half-Truths 的三元序（full > anchor > half），不是真實的多條件檢索；池是這 1379 張評測影像本身（transductive、無標籤），不是獨立大檢索庫；COCO 5000 張池只是敏感度。
3. 只在 CS-CLIP B/32 與 OpenAI CLIP B/32 兩個雙塔模型上測；CORE reranker 未評估。
4. 同一批 1437 列已被 stage1 看過結果；本探針的 alpha 沒有扣除這次重複使用，因此即使 PASS 也只是探索性證據。
5. 控制組為合成：正控制的替換細節是別列的 d+（對本影像顯然錯誤），比 Half-Truths 真實 d−（情境中合理的替代）容易；通過正控制只代表評分能分辨「明顯錯誤」，不代表能分辨真實 d−。
6. 單次打亂的負控制有抽樣誤差（隨機排序下 COR 的期望約 1/6 = 16.7%）；門檻定為 20%（約 +2 SE），凍結前定案。
7. 閘門 3 的 63.15% 以 stage1 之 holistic COR 為基準換算；不同資料或模型不可套用。
