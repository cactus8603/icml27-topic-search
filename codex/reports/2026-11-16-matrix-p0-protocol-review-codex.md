# P0 官方協定獨立審查

[codex] 2026-11-16 20:18（台北）。審查者 `official_p0` 沒有實作 runner/models/metrics/builder/launcher，沒有執行 GPU 推論，也沒有讀取新主結果。審查者負責官方公開來源核對與 acquisition；本報告的獨立範圍是執行程式如何落實官方協定。

**結論：首批 CORE2／8、Qwen3-VL Emb2／8 × 完整 COCO、NegBench 否定、CORE 指定 Flickr，共 12 格，可進入 Git 凍結與排隊。** 這是執行前協定審查通過；GPU 技術 smoke、完整推論、±1 pp 重現與獨立結果重算都尚未由本報告證明。

## 主要核對

- 官方來源為 `Alibaba-NLP/core-emb@400cf919838ceeecf4c94e3ac178a4270abcc4c5`。27 個 release 檔案通過 Git blob 核對。runner 載入未改官方 Qwen3 wrapper 與 `embedding.py`，不另寫 prompt／pooling；兩種模態實際 `is_query=False`，生效 system 是 `Represent the user's input.`。last-valid-token pooling、bf16 下 L2 normalize 與原碼一致。
- 向量以 float32 儲存是 bf16 值的無損展開；官方分數計算時轉回 bf16，在 GPU 做 matmul。普通 COCO／Flickr 依原碼使用 1000 query block；NegBench 依原碼完整 25014 × 5000 一次矩陣。逐一 row 呼叫 `torch.topk`，不換成 stable CPU sort。
- COCO／Flickr 的 top100 分數與原始 docid 交給同一 `pytrec_eval`，保留其分數平手時的 docid 處理。I→T 是 reference-caption fraction recall，沒有偷換成 any-positive success。NegBench 直接用 `torch.topk(k=5)` 命中原圖的比例，不經 pytrec tie sorting。CPU stable order 只作明示的補充診斷，P0 不拿它對論文主數字。
- 對照官方 CSV 重排圖庫與 caption 次序；正式普通／否定分母皆 25,014，不用舊 pilot。獨立直接掃描兩份 prepared panel：各 5000 張、25014 caption；每題 `positive_ids=[source_image_id]`，沒有 `positives` 覆寫或 ignore，圖片檔名與 image ID 相符。plain 原始空句 0、negated 原始空句 1。
- 未改官方 processor 的 CPU 空句測試通過，空字串仍有合法 system/user/assistant token。checkpoint tokenizer 與 wrapper 共追加兩個 endoftext，保持 release 行為。原環境 Transformers 5.20.2 的 min_pixels API 不相容，隔離固定 4.57.3 + qwen-vl-utils 0.0.14 後成功；舊環境未修改。
- builder 綁定完整模型 payload SHA、原 CSV／prepared panel／Flickr manifest 及收據 SHA、官方源碼 SHA、執行源碼 SHA、獨立依賴目錄 SHA、版本與 install report。launcher 使用離線模式、指定 torch214 Python 與獨立 PYTHONPATH；正式 runner 驗 Git commit 內容與工作樹乾淨，先 synthetic smoke，再資料索引。
- 本地 image4／text32 與作者 COCO/Flickr64、Neg32 不同，已明示為事前記憶體設定。smoke 比較 batch1 與固定 batch，強制 cosine ≥0.9995 且 max-absolute ≤0.01；失敗寫 FAILED，不能覆寫成 PASS；正式 index 再查數值門檻及模型／source／runtime identity。這不是4對64的直接驗證，也不是論文效果驗收。

## 審查中修正且已複核

1. 舊 draft 的 score 用 float32 CPU dot／stable ties，會改變 bf16 cutoff 與 tie；正式 published path 已另存 native score/topk。
2. 原 prepared panel 排序及排空句會改變官方次序／分母；正式 P0 已按原 CSV 恢復並保留空句。
3. NegBench 原碼是完整矩陣與逐 row topk，已固定 25014 block、逐 row topk。
4. draft smoke 只有記錄 cosine/maxabs，沒有強制門檻；已改機械失敗，並新增 NaN／缺失／超界檢查。
5. prepared labels 需要綁定來源 image ID；`load_labels` 已驗唯一 source positive／無 ignore，並獨立掃全量資料核對。

## 已執行的技術驗證

- 在固定隔離依賴下執行 `python -B -m unittest discover -s codex/retrieval_matrix -p 'test_*.py'`：**26 tests PASS**。包含斷點續算、來源／SHA／未釘 commit 拒絕、空句／CSV 次序、label separation、數值門檻、候選前綴與多正例指標。
- 額外獨立合成資料：21 query × 8 gallery、bf16 大量同分、數字 docid 刻意非排序。runner 的 COCO T→I、COCO I→T 與 NegBench 計分，逐一對照直接的官方 pytrec／torch-topk 語義：**PASS**。資料完全合成，不是本題效果。收據：`dl/retrieval-matrix/sources/review-synthetic/review.json`。
- 官方 CPU processor 空 caption／精確 token 與版本收據：`dl/retrieval-matrix/sources/core-empty-caption-cpu.json`。
- P0 缺模型新增檔案共 67 檔、33,906,836,845 bytes（31.578 GiB）；下載程序 exit0，逐檔 payload 在 finalize 前做 trusted LFS SHA-256／Git blob SHA-1 核驗，獨立收據核對無缺漏。`dl/retrieval-matrix/acquisition/verification-summary.json` 為 COMPLETE，receipt SHA256 `073525aa1a830b1f29c2c18b642c72d54397e623f73ff9e68972fc70e7d923a2`。

## 未涵蓋、不可宣稱完成的事項

此首批不含完整 P0 的 GME、VLM2Vec、MCMR、reranker MCQ，也不含 P2／P3／P4 矩陣結果。GME/Qwen2 需獨立舊 API 環境；VLM2Vec-V2 是 DoRA adapter，CORE generic loader 沒有套用；CORE 公開 MCMR 轉換腳本缺失，exact 商品文字尚未可驗。MCMR 的論文目標是 MRR@10，不能以無截斷 MRR 代替。CORE MCQ 0.646／0.698 是 VOC/COCO 宏平均，不能對 COCO-only。

論文 Appendix image tokens1000/text200 與公開 release code 不符，首批依事前固定的 release code 執行；不得看分數後選擇較接近目標的模板／解析度。首批實際計分後，每模型至少兩個不同資料集的主指標在 ±1 pp 內，才能進已驗收 P2。GPU smoke 與研究輸出尚未於本報告核定。

## 審查版本

以下是審查完成時檔案雜湊；若後續更動相關行為，需重新核對影響範圍。

| 檔案 | SHA-256 |
|---|---|
| `codex/retrieval_matrix/runner.py` | `83515be3f102b95be4d3c1049be9c944f57802cb224d72f4d73489f82994d149` |
| `codex/retrieval_matrix/models.py` | `2befb40f767a01e6b886e69918ab371648b447f19465b28562b524f393179afe` |
| `codex/retrieval_matrix/metrics.py` | `dc7400e5e5392de5a737238e29465dd60a1a1f7827e20d7fe8b97e10eba6a91b` |
| `codex/retrieval_matrix/build_plan.py` | `3ca6cdf265cc8cb3e318d89f386e21700e397960927e741a1112fe7301d5411c` |
| `codex/retrieval_matrix/run.sh` | `2246d3a2f667948ec08b9a2a94248f62c9b8a42e7634d5b1422bc34253b92e50` |
| `codex/retrieval_matrix/check_dependencies.py` | `831bc71cb90a7a6cadcabddbc0ba3408f016586243d0e675d10b2ffc642b9a73` |
| `codex/retrieval_matrix/official_protocols.json` | `1320d67f9979f6b244cbb124fbe8e4c40f8325055e19b394ab8aaf87e1f87b11` |
| `codex/retrieval_matrix/PREREG.md` | `c36f40b4ee6ebf1d4aa8a3f3bacb2fd2b5e1c0b5e428ddd9025ba32453635524` |
