# 第 60 號報告 §7：固定索引與 reranker 補量

[codex] 這是沿用同一批資料的事後重判。先凍結本檔、`configs/params_headroom.json`、`scripts/{headroom,headroom_models,test_headroom}.py` 與 `slurm/headroom.sbatch`，再依協作規則送一次正式 GPU 作業（上限 3 GPU 小時）。現象與訊號沿用第 60 號報告。原 Stage 1 與 Probe 2 結果作為兩個免訓練聚合器的實作描述，撤回其方向淘汰解讀及 probe2 `rejection_text`。原結果表保留供追溯。新的模型分數未讀取、未產出。

## 評估

- 主判讀沿用 `kbora/Half-Truths` 凍結快照的 COCO(Qwen) 509 列，以官方列序讀取。其他兩個子集保留同一流程的描述性輸出。正式 COR 為 `P(score(full) > score(anchor) > score(half))`，平手算錯。分數用未四捨五入的比例乘 100，固定依點估計判讀。個別模型另報既有 seed 的列層級 95% bootstrap 與全對半正確率，不增加判準。
- 固定索引類 CS-CLIP、CORE-EMBED-2B、SigLIP2 So400m patch14 384。CS-CLIP 沿用 Stage 1 同批 cosine，使用 row ID 和 image SHA-256 檢查列序。新模型每張圖獨立產生一個 L2 正規化向量，文字分別獨立編碼，cosine 評分。CORE 依作者 wrapper 的實際系統／使用者訊息格式、使用固定 query instruction、checkpoint chat template、末位置 hidden state。SigLIP2 使用凍結 processor、英文小寫、固定 64 token padding、輸出 pooling embedding。各 checkpoint完整 revision 見設定。
- CLIC 選作者發布的原 CLIP ViT-L/14 PixPr-RedCaps 權重。其模型設定可由原生 OpenCLIP 處理，沿用原 CLIP tokenizer。已在半個工作天內確認公開的固定 revision、設定與 safetensors，納入固定索引基線。離線權重仍須在正式執行前完成準備。不得臨時更換 checkpoint。[CLIC 權重與設定](https://huggingface.co/nmndeep/CLIC-ViT-L-14-224-PixPr-RedCaps/tree/4b10418c8e4330aeed37f18f9d00f23c475de767)
- 上限參考 `Alibaba-NLP/core-reranker-2b`。使用作者 reranker 的系統、query、document 訊息格式，caption 為 query、圖像為 document。分數為第一個 assistant token 的 `logit(yes)−logit(no)`，其排序與作者二元 sigmoid 等價。使用指定 revision，以合成影像檢查是否能載入及算分。若在正式運行前一個工作天內仍無法跑通，記錄故障並選 `Qwen/Qwen3-VL-2B-Instruct` 固定 revision。備援把圖像和 `Does the following caption correctly describe this image? Answer with exactly yes or no.\nCaption: {caption}` 送進同 checkpoint chat template，`\n` 是換行。取第一個 assistant token、全詞表 log-softmax 下小寫單 token `yes` 的對數機率，不生成回答。CORE reranker 和備援擇一，模型身份隨結果保存，在看正式 COR 前鎖定。
- E2 舊設定 `main` seed、`k=1…4`，以正式 COCO 標註重生同一批查詢並在同一個 job 計算。固定索引模型輸出原本的 pairwise 錯誤與 recall@50，reranker 只對固定的正例及混淆影像打分。兩者皆用既有的 `err(k=4)−err(k=1)`、I* 影像群集 bootstrap，數值只作描述。

## 判讀

`B_fix` 為必算固定索引模型在主子集 COR 的最大值。`R` 是選定 reranker 在同一主子集的 COR，是跨注意力參考。

- 有效固定索引 COR `≥85` → 空間 FAIL，在本 PREREG 範圍內不進候選池。即使另一固定索引分數未取得，觀測到的最大值已是有效下界。
- 所有必算基線齊備且 `B_fix <75` → 符合入池條件。若 `R ≥90`，加註限固定索引／單向量情境。缺 `R` 時，範圍未定。
- 所有必算基線齊備且 `75 ≤ B_fix <85` → 灰區，由使用者決定。
- 未達第一條且缺必算固定索引分數 → 未能測試。CI、E2 或新模型現有排行榜名次均不改判。

## 實作來源與執行前核對

- [CORE 作者程式，400cf919](https://github.com/Alibaba-NLP/core-emb/tree/400cf919838ceeecf4c94e3ac178a4270abcc4c5)。模型卡的連結未導到作者 repo，但該 repo 公開實際的 embedding wrapper 與 reranker。補量在現有 Transformers 上直接用 `Qwen3VLModel` 及 `Qwen3VLForConditionalGeneration`，沿用角色、最後位置讀取及二元分數，不要求執行外部載入程式。輸入圖像的 patch 配置與 resize 用 checkpoint processor。
- [CORE-EMBED](https://huggingface.co/Alibaba-NLP/core-emb-2b/tree/cc9870ea92d64ebd5f8f9fc368d1120453564158) 發布兩份 PyTorch `.bin` shard。[CORE-RERANKER](https://huggingface.co/Alibaba-NLP/core-reranker-2b/tree/64e1a36cd6ecace459a4c7cd8f7d0ef5d6fee30b)、[SigLIP2](https://huggingface.co/google/siglip2-so400m-patch14-384/tree/e8e487298228002f3d8a82e0cd5c8ea9c567f57f)、[Qwen3-VL 備援](https://huggingface.co/Qwen/Qwen3-VL-2B-Instruct/tree/89644892e4d85e24eaac8bacfd4f463576704203) 均發布 safetensors。[CLIC 作者權重表](https://github.com/AmitPeleg/CLIC#evaluation) 指定選定 checkpoint。
- 執行環境為群組共用 `icml/envs/vllm033` 的 Python，已確認 torch 2.14.0+cu130、torchvision 0.29.0+cu130、Transformers 5.20.2 可匯入所需模型類別。由可連網的準備環境取得凍結的模型版本，再執行 `scripts/headroom.py --preflight`，核對 Half-Truths、CS-CLIP 既有向量及選定分支的完整權重。使用 `--smoke <model>` 跑合成影像並核對載入層及有限分數。必要的固定索引 checkpoint 缺失時先回報。
- 每次正式作業只向新建的 `results/headroom/<Slurm jobid>/` 寫檔。各模型先保存主子集，再逐一保存描述性子集。某模型錯誤時繼續其餘模型，並以本次來源可驗證的結果做機械判讀。E2 的檔案檢查、查詢生成與評分在判讀完成後進行，E2 缺件或失敗寫入日誌。

```bash
export HF_HOME=/work/ChihChungHsu/cactus8603/icml/hf_home
export TMPDIR=/work/ChihChungHsu/cactus8603/icml/tmp
PY_HEADROOM=/work/ChihChungHsu/cactus8603/icml/envs/vllm033/bin/python
HF_HUB_OFFLINE=0 "$PY_HEADROOM" -I scripts/headroom.py --stage-models
HF_HUB_OFFLINE=1 "$PY_HEADROOM" -I scripts/headroom.py --preflight
```

若依上文採用備援，兩個命令同時加 `--reranker vlm_fallback`，正式工作用 `HEADROOM_RERANKER=vlm_fallback`。

## 讀取紀錄

- [codex] 已讀 `results/headroom/26393/` 內的 `verdict_headroom.json`、`cs_clip.json`、`core_embed.json`、`siglip2.json`、`clic.json`、`core_reranker.json` 及上述五個模型的 `e2_*.json`，2026-11-15 03:15。判準沿用 `c834e87`。
- [codex] 2026-11-15 03:35–03:41，再核對 26393 輸出、Probe 2 正負控制與資料版本。CPU anchor 診斷的逐檔讀取記錄保存在 `codex/retrieval-anchor-audit/results/anchor_audit.json` 的 `reads`，包括 `results/ht` 的原輸出、`results/probe2` 的同批與 COCO5000 影像池、26393 結果及三個凍結 Parquet。診斷沿用已看過的子集；未改入池判準。
