# [codex] CORE MCQ 首批來源清冊

官方 CORE：`Alibaba-NLP/core-emb` commit `400cf919838ceeecf4c94e3ac178a4270abcc4c5`；27 個固定檔案保存在 `dl/retrieval-matrix/sources/core/`，逐檔 Git blob SHA-1 與 SHA-256 見 `protocols.json`。其中 `tasks/cola.py`、`tasks/scpp.py`、`tasks/negbench.py` 的評分分支、`models/load_models.py` 與 `main.py` 固定了 engine、request、原始分數與圖片 token 數。表格驗收固定 CORE paper2609.04083 Table8，包含兩個 NegBench 子集的等權宏平均。

CORE2B revision `64e1a36cd6ecace459a4c7cd8f7d0ef5d6fee30b`；CORE8B revision `d95d36a34300e34b87cc1793827ac798642c4488`。完整權重、processor、tokenizer、config 與作者 `args.json` 的實際 payload SHA-256 全列在 `backend_specs.core.draft.json` 及執行 plan。`args.json` 補件收據在 `dl/retrieval-matrix-rerank/acquisition/core-args-receipt.json`；不修改原 embedding snapshot。

官方 ms-swift4.0.0 wheel SHA-256 `891e798a9b4d258bdf3f145ae84a5c15325a57f78bb82ff8242dcc2b569db66b`；已安裝所有 Swift `.py` 另列實際 SHA-256。作者 checkpoint 記錄 swift4.0.0.dev0，未釋出精確 development commit，因此環境有此已揭露差異。實際 auto model type／template、CPU 合成圖 processor、yes/no binary projection 核對收據為 `dl/retrieval-matrix-rerank/cpu-probe.json`。完整安裝清單與下載 wheel hashes 見同目錄 `install-report.json`、`requirements.freeze.txt`。

COLA 作者 `multiobjects.parquet`、SugarCrepe++ 原五個 JSON、NegBench 作者 COCO／VOC MCQ CSV 的 SHA-256 全列在 `plan.core-mcq.v1.json` 的 `source_data`。影像每張的 bytes／SHA-256／尺寸在 `dl/retrieval-matrix-rerank/data/mcq-prepared/images-receipt.jsonl`。完整 15,912 題與 58,891 pairs 均按原來源順序保存；沒有因空字串、重複圖或重複題刪列。

VOC CSV 來自 NegBench 作者公開 Drive 檔案 `1BuB48dS4asJEUdWfTExSe-Wr_5IrtcqH`。VOC2007 原 test archive 使用 torchvision 官方指定 `https://thor.robots.ox.ac.uk/pascal/VOC/voc2007/VOCtest_06-Nov-2007.tar`，MD5 `b6e924de25625d8de591ea690078ad9f`；另記 archive SHA-256 `6836888e2e01dca84577a849d339fa4f73e1e4f135d312430c4856b5609b4892`。5031 列 MCQ 引用的4951張圖全部存在；重複題保留。唯一資料路徑適配是 CSV image_path basename 對已核驗原圖目錄。

固定資料配對以 CPU 直接比較官方分支：COLA 的 image0 正 caption0、image1 正 caption1，兩個 caption-choice 比較皆須嚴格勝出；SCPP 兩正 caption 都須勝同一負 caption；Neg caption0 須勝其餘三 caption。Jina 官方 COLA 分支改為 image-choice，留在另一協定，不混入 CORE 首批。
