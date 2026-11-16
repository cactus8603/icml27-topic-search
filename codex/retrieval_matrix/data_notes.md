# [codex] 公開評估資料取得紀錄

範圍：報告 64 的 COLA（GQA／PACO／multiobjects）、SugarCrepe++、CORE 論文指定 Flickr30K 1k。僅下載資料與機械檢查，不執行模型、不讀研究結果、不使用或列舉自建 panel 的 test 內容。不建立非官方替代 dev 切分。

`data_acquisition.py` 使用現有 `dl/retrieval3090/envs/retrieval3090/bin/python`，將資料放在 `dl/retrieval-matrix/data/`。下載可續傳已驗收的完整檔案，`--offline` 逐檔驗收；每個資料集有 `audit.json`、`file_receipt.jsonl` 及 SHA256。Git/LFS 檔比對來源 SHA256，Git 小檔比對 blob SHA1，同時另記本機 SHA256。COCO 官方逐圖服務不發布各檔雜湊，初次取得記來源 URL 與 SHA256，後續 offline 核對先前收據，並驗 JPEG。

操作：

```bash
TMPDIR="$PWD/hf_home/tmp" HF_HOME="$PWD/hf_home" \
  dl/retrieval3090/envs/retrieval3090/bin/python \
  codex/retrieval_matrix/data_acquisition.py --workers 16

TMPDIR="$PWD/hf_home/tmp" \
  dl/retrieval3090/envs/retrieval3090/bin/python \
  codex/retrieval_matrix/data_acquisition.py --offline

TMPDIR="$PWD/hf_home/tmp" \
  dl/retrieval3090/envs/retrieval3090/bin/python -m unittest \
  codex.retrieval_matrix.test_data_acquisition -v
```

## 來源與切分

- SugarCrepe++：[作者 repo](https://github.com/Sri-Harsha/scpp/tree/c083bbf039a5ca215acaf25e7ee736d2ea6f3864) 指向 [作者 HF](https://huggingface.co/datasets/Aman-J/SugarCrepe_pp/tree/dea2a1b6f9e1069c609f676aa55ec61e9b65fb61)。五 JSON 共 4,757 筆、1,542 個互異 COCO val2017 影像；完整覆蓋本機已有影像，不重複下載。各子集 replace_att 788、replace_obj 1,652、replace_rel 1,406、swap_att 666、swap_obj 245。原始 caption、額外正 caption、負 caption 均保留。這份公開資料只有 evaluation 用途；不得拿來調參後仍稱 unseen。標註 CC BY 4.0，影像保留 COCO／原 Flickr 的條款。
- Flickr30K：[CORE 官方 datasets.md](https://github.com/Alibaba-NLP/core-emb/blob/400cf919838ceeecf4c94e3ac178a4270abcc4c5/datasets.md) 明指 [royokong/flickr30k_test](https://huggingface.co/datasets/royokong/flickr30k_test/tree/a07382c123b95eec85827a9d8ce3eb135093411a)。固定 parquet 141,411,915 bytes，SHA256 `246b2f511b245fa9f74a29a8a21ef48bd229778e9a90553f168b1b22ecd020c9`，含完整 1,000 JPEG／5,000 captions。匯出 `test_manifest.jsonl` 以及 CORE task 要求的 `test_1k_flickr.csv`（filename、raw、imgid），不改資料列序或 caption 順序。這 1,000 圖與 [Flickr30K Entities 作者 test.txt](https://github.com/BryanPlummer/flickr30k_entities/blob/68b3d6f12d1d710f96233f6bd2b6de799d6f4e5b/test.txt) 僅重疊 25 圖；應稱「CORE 指定 HF test 1k」，不可混稱 Entities 的官方切分。HF card 未聲明新 license；沿用 Flickr30K 上游影像與學術使用條件，下載這份 CORE 指定公開版本不代表重新授權或作者原始申請流程已完成。
- COLA：[作者 repo](https://github.com/arijitray1993/COLA/tree/997457b78a1cadf81954747e2ebae9d532f0c2d3) 與 [作者 HF](https://huggingface.co/datasets/array/cola/tree/daee86b15a60308cfcc277ea11ab7f6be60fdfbc)。GQA JSON／parquet 實際是 2,589 圖、320 類，PACO 是 7,921 圖、400 類，multiobjects 210 配對；HF 和原始 JSON 的 image identity、label、hard_list 逐列核對。影像 union 是 10,623 張（VG 2,702、COCO 7,921）。HF card 宣稱包含 PACO 的 COCO 圖，但此 revision 實際完全沒有 `images/coco/`；從既有 COCO val2017／panel dev 圖重用可得者，其餘依 PACO 原始路徑從 `http://images.cocodataset.org/{train2017,val2017}/{id}.jpg` 取得。HTTP 是官方逐圖服務的公開方式；HTTPS 憑證主機名不符，程式未停用 TLS 驗證。標註 MIT，影像遵守原 Visual Genome／COCO 條款。

## 尚缺作者資料的項目

1. COLA **官方 train**：固定 GitHub tree 只有四份 val 標註，唯一 branch 是 main，tags 為空；作者 HF configs 也只有 val；[作者 project 頁](https://cs-people.bu.edu/array/research/cola/) 只指向這兩處，沒有額外 train 連結。原始 GQA／PACO train 不等於經作者處理、排除 unseen query 的 COLA train。不能將公開 val 擅自切成 dev 來宣告符合「COLA 官方 train 切 dev」。需作者提供確切 train artifact，或使用者另定並凍結替代資料政策。
2. COLA **seen／unseen**：發布檔沒有 query seen/unseen 清單，train 也缺，因此這兩個分項不可憑類別排序推斷；可計算完整官方發布集上的 all mAP，但必須標明 GQA 發布 2,589 圖與論文約 1,952 圖不同，不能視為相同重現分母。
3. COLA `scripts/eval.py` 的 `__main__` 範例仍以 list 拆解目前已為 dict 的 singleobjects JSON；應讀 `labels`／`data` 鍵。其計算規則仍可依函式核對，禁止將範例直接執行的失敗誤判為資料缺失。

## Flickr30K 原始 split 交叉核對

另從 [Karpathy 原作者 caption archive](https://cs.stanford.edu/people/karpathy/deepimagesent/caption_datasets.zip) 保存 `dataset_flickr30k.json`、原始 archive 與 SHA256 收據至 `flickr30k_test/upstream/`。CORE 指定 HF 的 1,000 個 image ID 與 Karpathy test 完全一致，但只有 575 圖的五個 caption 原字串完全一致；HF 部分 caption 包了額外的字面雙引號。匯出時保留 HF 內容，未自行去引號。這個差異必須在重現表說明，不能把 CORE-HF 文字與 Karpathy 原始文字當作逐 byte 相同。機械比對詳見 `upstream/karpathy_source.json`。

這些差異在主評估前已明示，不以結果選擇資料版本。各下載／驗收完成狀態以 `data_sources.json` 與本機 `audit.json` 為準。
