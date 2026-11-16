# [codex] 報告 64 公開資料取得與 MCMR 全量資料獨立稽核

時間：2026-11-16 20:26（台北）。本報告僅涉及資料來源、切分、檔案身份與可用性；沒有模型推論、研究結果讀取或自建 panel test 內容存取。

## 已完成的公開評估資料

| 資料 | 完成範圍 | 驗收 |
|---|---|---|
| SugarCrepe++ | 五個官方 JSON，4,757 題、1,542 個互異 COCO val2017 圖 | 五 JSON 來源 Git blob 雜湊相符；全部影像 JPEG 可驗、SHA256 有收據、缺圖 0 |
| Flickr30K，CORE 指定 HF test 1k | 1,000 JPEG、5,000 captions，原 parquet 141,411,915 bytes | parquet 來源 LFS SHA256 相符；全部嵌入影像逐 byte 匯出一致；缺圖 0；CORE CSV 與中立 manifest 均保留 HF 原字串 |
| COLA GQA | 320 queries、2,589 圖 | HF parquet 的 image identity、label、hard_list、query vocabulary 與原作者 GitHub JSON 逐列一致 |
| COLA PACO | 400 queries、7,921 圖 | 同上，逐列一致；原 HF revision 缺 COCO 圖，已依官方原始 filename 從既有檔案及 COCO 官方逐圖服務補齊 |
| COLA multiobjects | 210 配對 | 兩張圖身份與兩段 caption 均與原 GitHub JSON 逐列一致 |
| COLA 圖片總聯集 | 10,623 圖、1,629,290,474 bytes | 缺圖 0；2,702 張 VG 圖比對官方 HF LFS SHA256；7,921 張 COCO 圖記官方 URL、本機 SHA256，再以首次收據全量 offline 核對；全部圖片格式驗收通過 |

資料位於 [dl/retrieval-matrix/data](/ssd4/cactus8603/icml/dl/retrieval-matrix/data)。每份的 `audit.json`、`file_receipt.jsonl`、圖片收據及下載／offline 驗收 log 都已寫入。2,902 張 COLA 圖重用既有資料；其餘實際下載。沒有重複下載既有 SugarCrepe++ 的 COCO 圖。

來源固定版本：

- SugarCrepe++：[作者 repo `c083bbf…`](https://github.com/Sri-Harsha/scpp/tree/c083bbf039a5ca215acaf25e7ee736d2ea6f3864) → [作者 HF `dea2a1b…`](https://huggingface.co/datasets/Aman-J/SugarCrepe_pp/tree/dea2a1b6f9e1069c609f676aa55ec61e9b65fb61)。五子集題數依序為 replace_att 788、replace_obj 1,652、replace_rel 1,406、swap_att 666、swap_obj 245。
- COLA：[作者 GitHub `997457b…`](https://github.com/arijitray1993/COLA/tree/997457b78a1cadf81954747e2ebae9d532f0c2d3) 與 [作者 HF `daee86b…`](https://huggingface.co/datasets/array/cola/tree/daee86b15a60308cfcc277ea11ab7f6be60fdfbc)。原 JSON、官方評分程式、Git tree、branch／tag 查詢與 project 頁均保存在 `cola/upstream/`。
- Flickr30K：遵循 [CORE 官方固定版本下載說明](https://github.com/Alibaba-NLP/core-emb/blob/400cf919838ceeecf4c94e3ac178a4270abcc4c5/datasets.md)，取得 [royokong/flickr30k_test `a07382c…`](https://huggingface.co/datasets/royokong/flickr30k_test/tree/a07382c123b95eec85827a9d8ce3eb135093411a)。parquet SHA256 為 `246b2f511b245fa9f74a29a8a21ef48bd229778e9a90553f168b1b22ecd020c9`。

最終 offline COLA receipt SHA256：`5ac0a7c72b81ea6c2ebd475f625d13fc1f93070bd37dd14260e5c84038535daa`。本輪新增的取得程式及八個完整性測試位於 `codex/retrieval_matrix/`；測試涵蓋 LFS／Git 內容竄改、檔案大小、路徑穿越、固定 revision、offline 禁止補網路、Flickr split 截斷與 COCO 先前收據不符，八項皆通過。

## 必須保留的資料版本差異與缺口

**COLA 官方 train 未發布；使用者已決定「沿用 COCO dev，COLA 僅測試」。** 目前作者 GitHub 唯一 branch 是 main、tags 為空，tree 只發布四個 val benchmark；HF configs 也只有 val；[官方 project 頁](https://cs-people.bu.edu/array/research/cola/) 只連到這兩處。沒有取得作者訓練產物或 seen／unseen query 清單。GQA／PACO 原始 train 並不等於作者排除 unseen 組合後的 COLA train。本輪沒有將公開 val 擅切 dev；新的使用者決定已撤銷「取得 COLA train 作 dev」需求，依新政策由 COCO 建開發集。COLA all mAP 可測；seen／unseen 分項仍不能憑空推定。

**COLA GQA 發布版本與論文圖數不同。** 目前原 GitHub JSON 與 HF 都是 2,589 張，且每張都有正類別標註；無法以全負例列解釋論文約 1,952 張的差異。未任意刪至目標筆數。可評估完整發布集，但不能將分母差異隱去後宣稱精確重現論文。

**Flickr30K 切分名稱與 caption 版本須明示。** CORE 指定 HF 的 1,000 圖 ID 與 [Karpathy 原作者發布檔](https://cs.stanford.edu/people/karpathy/deepimagesent/caption_datasets.zip) 的 test 完全一致；與 [Flickr30K Entities 作者 test.txt](https://github.com/BryanPlummer/flickr30k_entities/blob/68b3d6f12d1d710f96233f6bd2b6de799d6f4e5b/test.txt) 只有 25 圖重疊。HF 與 Karpathy 原始字串有 574／5,000 captions 不同，涉及 425 圖，主要為額外字面雙引號；去掉外引號後仍有 27 圖不完全相同。因此同時交付兩個明確版本：

- `flickr30k_test/test_manifest.jsonl` 與 `test_1k_flickr.csv`：CORE 指定 HF 原文，供重現同一輸入。
- `flickr30k_test/karpathy_test_manifest.jsonl`：原作者原始 captions，保留原 test 列序，使用相同完整影像檔。

原 Karpathy archive SHA256：`4cfd70132527b80933105e5829dc9034eaab9573482e2e680abbab6130244817`；原 Flickr30K JSON SHA256：`db779ee2c5df40c25ff0eebeca464f6124b620cc4a9398650aea71fd1b53d4af`。來源身份核對收據在 `flickr30k_test/audit.json` 的 `source_lineage`。沒有自行修 caption 後仍沿用同一版本名稱。

## MCMR 全量解壓的獨立審查：PASS

審查對象是 root 實作的 `full_data.py`、`preparation.json` 及 `images-receipt.jsonl`，並非沿用製作者的完成宣告。獨立稽核沒有 import 製作程式，也沒有重新解壓覆寫資料，而是讀原始 archive、metadata、落地影像與舊 pilot 收據逐項交叉核對。

權威證據：

- [獨立稽核 JSON](/ssd4/cactus8603/icml/dl/retrieval-matrix/data/mcmr-full/independent-audit.json)，完成時間 `2026-11-16T20:25:49+08:00`，`status=PASS`。
- [可重跑的獨立稽核程式](/ssd4/cactus8603/icml/dl/retrieval-matrix/data/mcmr-full/independent_audit.py) 與同目錄 `independent-audit.log`。
- 製作者程式 SHA256：`4a3a132ec257e0785052f6c99a3c25d53c489318951d6137e4eff147d30ec117`；原 archive SHA256：`7cbda9a5fbdba1240590885c576b4b31dccd047f14d2cc5b8a8e87c94bda2b86`；影像收據 SHA256：`26ca885a59d6c9a47d1883018d83cc7afa3ccfb7d6acdc67640906096d967376`。

完成的獨立檢查：

1. 原 `candidates.jsonl` 與 normalized metadata 的 104,981 個 candidate ID 完全一致、無重複；逐筆原 URL basename 與 `image_filename` 一致，URL 本身也一致。需 103,176 張互異圖，全部在新完整影像目錄，零缺圖。
2. **全部 123,561 張**落地圖的 bytes 與 SHA256 均符合影像收據；再串流讀原 `images.tar.gz`，**全部 123,561 個**非空 member 的原始 bytes／SHA256 逐一符合相同收據，且 archive／receipt 成員集合完全相同。
3. 原 archive 唯一零 byte 檔 `31RbkrUK1DL._AC_.jpg` 已明示列在 `unusable_source_entries`，不被任何 candidate 引用，不算有效影像；沒有靜默刪除 candidate 或 query。
4. 舊 pilot 為 **8,310 張圖**（8,037 JPG、273 PNG）及一個 `extraction.json`。全部 8,310 圖同時符合舊 extraction 收據與原 archive SHA256，pilot manifest 的舊 SHA 也未變。pilot 目錄最新 mtime 為 `2026-11-16 04:16:18+08:00`，早於新解壓；新舊圖不共 inode；新 `images` 與舊 pilot 都是正常目錄。製作程式的寫入範圍及本次實際檔案身份均支持「沒有覆寫舊 pilot」。

程式審查的兩個後續防護建議不影響上述目前產物的 PASS：未來可直接鎖定 normalized metadata 的固定 SHA，或從 raw 重算映射；另可對 `out/images` 子目錄本身增加 symlink 拒絕。目前 normalized 映射已由本次獨立逐筆核對，且實際 `images` 非 symlink。

## 執行邊界

本輪只使用 CPU 與公開、無須登入的下載來源；沒有新增 GPU 工作、登入帳號、代替使用者申請資料或同意額外條款。按 root 的凍結要求，四個 `codex/retrieval_matrix` 取得程式／文件在指定時間後保持不變；最終實際狀態以本報告及 ignored `dl/` 的完成收據為準，root 可在首批 P0 的 clean-tree 閘門後刷新機械來源表。
