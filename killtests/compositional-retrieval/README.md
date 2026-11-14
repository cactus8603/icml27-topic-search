# 多條件檢索的單調證據聚合：E0–E2 停損實驗

## 目的與對應回報
對應 `reports/2026-11-14-46-deepcheck-compositional-retrieval.md` 第 6 節 E0（基線重現）、E1（training-free 對數概似比聚合）、E2（條件數稀釋），以及 `2026-11-14-56-killtest-criteria-review.md` 的通用規則與第二梯建議。純推論、不訓練。判準全文見 `PREREG.md`（單一主指標＋三態；兩個主指標，各 α=0.025）。

## 資料與授權
| 項目 | 來源 | 授權 | 狀態 |
|---|---|---|---|
| Half-Truths（1437 列，含影像） | HF `kbora/Half-Truths` | CC-BY-4.0（來源影像保留原權利） | 已下載（約 200 MB） |
| CS-CLIP B/32 權重 | HF `kbora/CS-CLIP` | MIT | 已下載（約 605 MB） |
| CS-CLIP 程式 | GitHub `kargibora/CS-CLIP`，clone 於 `/work/ChihChungHsu/cactus8603/icml/ext/CS-CLIP`（預設分支 main-final，commit dcf79b9）| MIT | 已 clone；資料卡指向的 `release/paper-reproduction` 分支未使用，改以本實驗自寫評分並與官方腳本逐列交叉核對 |
| OpenAI CLIP ViT-B/32 | open_clip `pretrained=openai`（HF `timm/vit_base_patch32_clip_224.openai`） | MIT | 已快取：官方 clip 套件用的 `ViT-B-32.pt`（354 MB）在 `icml/tmp/crosscheck/home/.cache/clip/`；timm 版在 HF 快取 |
| COCO val2017 影像與 instances 標註 | images.cocodataset.org | 標註 CC-BY 4.0；影像 Flickr 條款 | zip 已下載（約 1.07 GB），已解壓（5000 張） |
| CORE reranker | HF `Alibaba-NLP/core-reranker-2b`（CC-BY-4.0，4.3 GB） | CC-BY-4.0 | **不取得**：模型卡的 GitHub 連結為空、wrapper 類別不公開、需 transformers>=4.57；列為未評估 |

## 環境
重用 `envs/t2irare`（torch 2.5.1、transformers 4.46.3）。缺的 open_clip、torchvision 0.20.1、pyarrow 17、timm、ftfy、openai-clip 以 `pip install --target` 裝到 `icml/envs/t2irare_extra_cscomp`（不動共用環境；`scripts/common.py` 會把它加入 sys.path）。pyarrow 26 與 numpy 1.26 不相容，已改用 17.0.0。

## 檔案
- `scripts/`：`common.py`、`extract_ht.py`（GPU）、`analyze_ht.py`、`e2_build_queries.py`、`e2_extract.py`（GPU）、`e2_analyze.py`、`unit_tests.py`、`crosscheck_repo.py`、`rerun_compare.py`、`throughput.py`、`download_assets.py`、`aggregate.py`
- `configs/params.json`（所有門檻與 seed）、`slurm/`（env.sh、bench、stage1、stage2）、`results/`、`logs/`

## 預估資源
- GPU：ViT-B/32 推論，Half-Truths 1437 列＋COCO 5000 張，兩模型含重跑，估計 **約 0.3–0.6 GPU 小時**（上限 3 小時；回報 46 估 1–10 小時，因不含大模型而遠低）。吞吐實測 job 約 10 分鐘。
- 磁碟：已用約 2.3 GB；尚需 COCO 解壓約 0.8 GB、OpenAI CLIP 約 0.6 GB、結果 npz 約 0.1 GB；合計 <6 GB（上限 40 GB）。

## 送出順序（寫出，未執行）
先在 login 節點（需 /work 有空間；寫入失敗就停）：
```bash
cd /work/ChihChungHsu/cactus8603/icml27-topic-search/killtests/compositional-retrieval
export TMPDIR=/work/ChihChungHsu/cactus8603/icml/tmp PIP_CACHE_DIR=/work/ChihChungHsu/cactus8603/icml/tmp/pip-cache HF_HOME=/work/ChihChungHsu/cactus8603/icml/hf_home
/work/ChihChungHsu/cactus8603/icml/envs/t2irare/bin/python -I scripts/download_assets.py     # 補下載 OpenAI CLIP 權重並解壓 COCO val2017
/work/ChihChungHsu/cactus8603/icml/envs/t2irare/bin/python -I scripts/unit_tests.py          # CPU，數秒
git add -A . && git commit   # 凍結 PREREG 與判讀腳本（commit 由人執行）
```
再送出：
```bash
sbatch slurm/bench.sbatch       # 10 分鐘吞吐實測；若實測推估 >2 倍預估則重排
sbatch slurm/stage1_ht.sbatch   # G1–G3 與 E0／E1
# 閘門通過後
sbatch slurm/stage2_e2.sbatch   # E2 與最終 aggregate
```
最終看 `results/aggregate.md`。

## 風險與未確認事項
1. 以上多數步驟尚未在 GPU 上跑過；只做過 CPU 極小 smoke test（見回報）。OpenAI CLIP 經 open_clip 載入與官方 `clip` 套件是否逐位等價，要靠 stage1 的交叉核對確認（容差 1e-4）。
2. CS-CLIP 載入照官方腳本（open_clip `ViT-B-32` 非 quickgelu＋strict load）；官方 README 的 HT 準確率（76.4 等）與論文 v2 的 COR（58.2 等）是不同指標，G2 只以 COR 阻擋。
3. v1／v2 數字不同（v1 CLIP 40.6、CS-CLIP 69.3 為另一協定），重現只對 v2。
4. 細節抽取規則對 swap 類 binding 失效（PREREG §7）；P1 可能因此偏低，這是假設的一部分，不調整。
5. P1 的 FAIL 條件比回報 46 原案嚴（明確失敗不給 CS-CLIP 救援），請審查者確認。
6. E2 的條件與文字模板（COCO 類別、"a photo of ..."）是我定的，回報 46 只寫「COCO 或 VG」；J 缺一個條件但可能含其他未列類別，confound 已在 PREREG 聲明。
7. CORE reranker 未評估，46 的「reranker >= 90% 限定情境」旗標無法判定。
8. 群組額度吃緊：下載與解壓前先確認剩餘空間；寫入失敗即停（本次曾遇一次 EDQUOT fsync 暫時錯誤）。

## 第二探針（停損後；PREREG_probe2.md）
stage1 P1 FAIL 後的整句條件 PMI（s~）探針；同一批 1437 列已看過結果，凍結後只跑一次。檔案：`PREREG_probe2.md`、`configs/params_probe2.json`、`scripts/probe2_{precheck,condpmi,embed}.py`、`scripts/aggregate_probe2.py`、`scripts/unit_tests_probe2.py`、`slurm/probe2.sbatch`；輸出在 `results/probe2/`。
- 主指標與閘門只需 stage1 已存的 npz（CPU）；GPU 只用於正控制句與 COCO 5000 影像池的 embedding。
- 跑前預檢 `results/probe2/precheck.json` 已完成（CPU，描述用）。
- B 做法（VLM 條件 LLR）：**未做**。
- 凍結（commit）後執行：`sbatch slurm/probe2.sbatch`（內含凍結檢查、單元測試、embedding、分析、`aggregate_probe2.py`）。
