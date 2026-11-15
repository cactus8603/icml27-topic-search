# dev-3090 需要的檔案與資料集

[claude] 2026-11-16（台北）。清單來自程式實際開啟的檔案：在 CPU 預檢中掛上 Python 的開檔追蹤，再加上 GPU 步驟會載入的模型檔。以下路徑都以 `ICML=/work/ChihChungHsu/cactus8603/icml` 為根目錄。凍結程式寫死這個根目錄，3090 主機要放在同一個絕對路徑，或用 symlink 指過去。 [開發規範](SPEC.md)

全部約 **24 GB**。用 `bash check_host.sh --hash` 可檢查一台主機是否齊全。

## 資料集

| 項目 | 路徑 | 大小 | 用途 | 固定版本 |
|---|---|---:|---|---|
| **開發圖庫** | `data/coco2017-anchor-rerank/`（`images/*.jpg` 與 `manifest.json`） | 1.3 GiB | 8,192 張 COCO train2017 JPEG，其中 8,191 張入池（`578492` 因位元重複排除），136 條雙物件查詢、118 群。預檢會逐張核對 JPEG 的 SHA-256 | manifest `8f43eedb…cf78d` |
| 前一輪面板清冊 | `data/coco2017-rerank/manifest.json` | 304 KiB | 預檢用來確認開發圖庫和前一輪的影像、查詢不重疊，不讀圖片 | `df51d103…3f` |
| **COCO 標註** | `data/coco2017/annotations_trainval2017.zip` | 242 MiB | 讀 `instances_train2017.json` 的類別、影像尺寸與標註面積，建立每張圖的類別（非 crowd、面積至少 2%），再核對 136 題的相關清單。D1／D2 也用它區分「只符合一個條件」的圖 | `113a836d…0268` |
| 26601 公開收據 | `runs/retrieval-anchor-rerank/26601/summary.json` | 8 KiB | recall-budget 凍結程式要求核對的來源收據 | `817c6ab7…dfa` |

開發圖庫的 JPEG 都來自 COCO 官方 train2017。如果主機上缺少某些圖，可以從 `http://images.cocodataset.org/train2017/<12 位數 id>.jpg` 重新下載，但每張都必須通過 manifest 裡的 SHA-256 核對。

## 模型

都是 `params_headroom.json` 固定的 revision，放在 `hf_home/hub/models--<org>--<name>/snapshots/<revision>/`。

| 模型 | revision | 大小 | 角色 |
|---|---|---:|---|
| `kbora/CS-CLIP`（ViT-B/32，open_clip） | `afd6e90f39fe49137fe15d6a462d994eae4811f5` | 578 MiB | 第二個固定索引，入池策略的整句／條件分數 |
| `Alibaba-NLP/core-emb-2b` | `cc9870ea92d64ebd5f8f9fc368d1120453564158` | 8.0 GiB | 第一階段索引（CORE-EMBED），決定 CORE 排名與前 200 名 |
| `Alibaba-NLP/core-reranker-2b` | `64e1a36cd6ecace459a4c7cd8f7d0ef5d6fee30b` | 4.0 GiB | 交叉編碼重排器（CORE-RERANKER），每題前 200 名逐張打分 |

**模型複製要解開 symlink。** 叢集上 snapshot 裡的檔案是 symlink，指向 `hf_home/hub/blobs/` 的共用存放區。只複製 `models--*` 資料夾會得到無效連結，要用 `rsync -L` 或 `cp -L`。

## 執行環境

| 項目 | 路徑 | 大小 | 說明 |
|---|---|---:|---|
| Python 環境 | `envs/vllm033/` | 9.8 GiB | 自帶 Python 3.12 執行檔。Torch 2.14.0+cu130、Transformers 5.20.2、numpy 2.3.5、ijson、Pillow |
| open_clip 補充套件 | `envs/t2irare_extra_cscomp/` | 182 MiB | open_clip 2.32.0、ftfy 等。凍結程式會把它**放在 sys.path 最後**，讓 vllm033 自己的 torch／torchvision 優先 |

- **GPU 驅動**：Torch 是 CUDA 13.0 版，NVIDIA 驅動要支援 CUDA 13.0（R580 以上）。如果 3090 的驅動較舊，需要另建一個相同 Transformers 版本的環境，並在 `meta.json` 記錄版本差異。這樣的話，基線重現不一定能逐位相同。
- **主機 RAM** 至少 32 GiB。**磁碟**另留約 0.5 GiB 給輸出。
- 3090 支援 BF16，模型精度設定與 H200 相同。

## 不需要複製（也不應複製）

- `runs/retrieval-independent-panel` 與它的資料：這是**封存的正式驗證面板**，開發主機不應持有。
- Half-Truths、SigLIP2、CLIC、Qwen3-VL fallback、COCO val2017 圖片：這版開發程式沒有用到。

## 從叢集複製到 3090 主機

在 3090 主機上執行，`cluster` 換成叢集登入節點的位址。`/./` 讓 `rsync -R` 保留它後面的相對路徑。

```bash
ICML=/work/ChihChungHsu/cactus8603/icml
src=cluster:$ICML/.
mkdir -p "$ICML"
rsync -aR --info=progress2 \
    "$src/envs/vllm033" "$src/envs/t2irare_extra_cscomp" \
    "$src/data/coco2017-anchor-rerank" "$src/data/coco2017-rerank/manifest.json" \
    "$src/data/coco2017/annotations_trainval2017.zip" \
    "$src/runs/retrieval-anchor-rerank/26601/summary.json" "$ICML/"
rsync -aRL --info=progress2 \
    "$src/hf_home/hub/models--kbora--CS-CLIP/snapshots/afd6e90f39fe49137fe15d6a462d994eae4811f5" \
    "$src/hf_home/hub/models--Alibaba-NLP--core-emb-2b/snapshots/cc9870ea92d64ebd5f8f9fc368d1120453564158" \
    "$src/hf_home/hub/models--Alibaba-NLP--core-reranker-2b/snapshots/64e1a36cd6ecace459a4c7cd8f7d0ef5d6fee30b" \
    "$ICML/"
mkdir -p "$ICML/tmp" "$ICML/runs/retrieval-dev"
bash check_host.sh --hash
```

如果 3090 主機本來就能掛載叢集的 `/work`，就不用複製，直接執行 `check_host.sh` 確認即可。

## 在叢集上的檢查結果

2026-11-16 02:30 在叢集登入節點執行 `check_host.sh --hash`：四個固定檔案的 SHA-256、8,192 張 JPEG、三個模型的必要檔都齊全，Python 環境的匯入也通過（登入節點沒有 GPU）。
