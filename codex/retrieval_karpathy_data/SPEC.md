# [codex] 原版 Karpathy COCO test 資料取得

只供事前固定的官方 baseline 資格核對；不跑模型、不依效果選圖片或修改 development。原版 Stanford `caption_datasets.zip` SHA `4cfd70132527b80933105e5829dc9034eaab9573482e2e680abbab6130244817`；其中 `dataset_coco.json` SHA `2fd999220673258012acfb411a4e7e66af7d488050b2519b0badcc49b7600b8d`，取原順序 `split=test` 的 5,000 張。4,990 張有 5 句，10 張有 6 句，**共 25,010 句**，原 `raw` 包含空白均保留。若某官方 baseline 用前五句，必須另在其 protocol 固定，不覆寫此來源。

輸出在 `dl/retrieval-matrix/data/karpathy-original/coco-test/`：

- `manifest.jsonl`：每圖 `{id,cocoid,imgid,filename,filepath,split,path,captions,sentids}`。`id` 是原 COCO ID 的字串，`imgid` 是 Karpathy 列 ID；`path` 是絕對本機路徑。來源 `val2014` 文件名保留，caption 全取原 `raw`，不 trim、不大小寫轉換。
- `image-receipt.jsonl`：每圖 ID、filename、path、SHA256、bytes、CRC32、width、height、官方單圖 URL 與重用 cache／下載來源。
- `acquisition.json`：原 JSON、官方 ZIP 中央目錄、COCO 尺寸 annotation、固定 cache inventory、所有輸出 SHA 與覆蓋筆數。
- `offline-audit.json`：全圖／全 caption 原樣與 SHA／官方 ZIP CRC 的離線重核。

先盤點固定 shared caches，逐圖用官方 val2014 ZIP 中央目錄的 uncompressed bytes／CRC32 核對後才可按原名 symlink。中央目錄只下載 4.5MB，加 ZIP tail 64KB，未下載 6.6GB archive。缺圖只從 `http://images.cocodataset.org/val2014/COCO_val2014_<ID>.jpg` 取得，同樣驗官方 CRC／bytes、COCO annotation 尺寸與 JPEG，再記 SHA。COCO 官方 host 的 HTTPS 憑證不匹配，使用其公開 HTTP URL，不關閉 TLS 驗證。

原 Karpathy 與 CORE val2017 交集只有 593/5,000，不能沿用 CORE val2017 當同一官方 split。Flickr 原 photo ID 與 COCO annotation 的 `flickr_url` 僅作精確 ID 交集盤點，不代表視覺近重複全排。COLA GQA 的 VG ID 與 COCO ID 不同 namespace；未找到官方 VisualGenome image_data mapping 時交集為 unknown，不能記零。
