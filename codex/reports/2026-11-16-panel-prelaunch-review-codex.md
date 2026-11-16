# [codex] P2 panel 啟動前獨立審查

[codex] 2026-11-16（台北）。審查者 official_p0 並非 panel 模組實作者；只讀審查 `codex/retrieval_matrix_panel/`，未修改其程式、未讀 P0 模型結果、未開舊 test queries，亦未執行 GPU 或產生模型分數。結論：本次發現的兩項會阻止／削弱正式執行的問題已由 root 修正，修正版 CPU 驗收通過，沒有剩餘啟動阻擋項。真正的 P2 推論仍須等待各模型的完整 P0 結果、非產生者獨立重算及資格收據；本審查不授予任何模型 P0 PASS。

第一項原缺口是 builder/runtime 只接受 development audit 的 PASS 字串，沒有綁當前資料版本。新增 `data_audit.py` 後，兩端共同驗 audit 的獨立重建旗標、manifest/source receipt/freeze、四排除集合、完整20k影像收據、圖片解碼與題目重建；runtime 再核本次兩個 dev dataset entries 的 gallery/queries/image receipt SHA 與 audited v2 一致，不能拿 v2 PASS 接 v1 queries。錯誤 manifest/source/image receipt、缺獨立重建、錯接 v1 題目均有拒絕測試。

第二項是 COLA PACO 的共享圖片 symlink 被過窄的 resolved-root 檢查拒絕。實際有2902張 val2017 指向既有 COCO val root、1044張 train2017 指向舊 v1 dev/images。修正版只明示這兩個共享根，logical path 仍拒絕 absolute/`..`，resolved path 仍須位於固定允許根，且每張核 receipt 的 SHA/bytes；未放寬成任意路徑，也未開 v1 test。

准入規則已核對：COCO、Flickr、NegBench 三個不同資料集的報告必須全列，≥2 個主目標誤差≤1pp才 PASS；第三個不達標結果仍保留。COCO/Flickr 取 `official.t2i.recall_5`、NegBench 取 `official.R@5`，不從 secondary CPU ranking 指標替代。報告的完整分母、模型、資料集、plan SHA、settings 與完整 model spec 均需相符；獨立重算收據須標示不同 producer/reviewer、audited arrays，並綁三份報告實際 byte hashes。`admit()` 重算原收據並將當前完整 model spec 的 canonical SHA 重綁；獨立合成測試確認 batch、processor、revision、報告 bytes 或 review report hashes 變更均會被拒絕。

Qwen2 factory 已在乾淨 CPU 程序核對：`Qwen2OfficialModel` 繼承預先捕捉的原 `OfficialModel` class；GME 的 `super().__init__` 仍呼叫原 class，不受全域 factory 替換影響。另以無模型配置的 constructor mock 檢查 CORE/GME/VLM2Vec 分流，沒有載權重或使用 GPU。兩種實際 base phase `P0`、`P0-Qwen2` 的 builder 資料接線皆通過；該項只 mock admission 以檢查 builder，未儲存任何可執行 production plan／假資格收據。

COLA 評分與固定作者 `scripts/eval.py` 核對：先按每類 hard_list 選圖，再按完整 score-threshold groups 算 AP，最後在公開詞彙類別上等權平均。200組大量平手的合成向量與獨立逐threshold公式、任意圖片排列全部一致。PACO caption 只取 encoded attributes+noun，count不進文字。caption、GQA公開圖數與 float32 CPU similarity 的適配差異有明示，沒有宣稱 COLA paper exact-match。模型輸入 queries 不含 positives/ignore/label/hard_list；全庫候選與取K不使用標籤。COLA只接已具原始完整模型規格的固定基線，不供方法選擇。

完整資料 adapter CPU load 與圖片 SHA/bytes 核驗通過：

| 資料 | 全庫影像 | 查詢／labels | 正例總數 | ignore總數 |
|---|---:|---:|---:|---:|
| conjunction-dev v2 | 20000 | 1089 | 1845 | 9391 |
| negation-dev v2 | 20000 | 237 | 1164 | 5926 |
| COLA GQA | 2589 | 320 | 3347 | 793067 |
| COLA PACO | 7921 | 400 | 11118 | 2980012 |

最新15項模組 CPU 合成測試、Python syntax、shell syntax 與來源 snapshot 收據一致性全部 PASS。完整 adapter／factory／實際 data-audit 綁定證據為 `dl/retrieval-matrix/data/coco-development-v2/panel-adapter-independent-review.json`；額外 qualification/AP 合成證據為同目錄 `panel-synthetic-independent-review.json`。先前 v2 原始 annotation 獨立重建證據 `independent-audit.json` 仍有效。保留限制：否定237題僅診斷；正例互斥不等於統計獨立；舊v1曝光與公開來源近重複未完全排除的限制仍須揭露。
