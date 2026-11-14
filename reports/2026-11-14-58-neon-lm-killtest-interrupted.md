<!-- 子代理 a7fe7cf5d8596b6c2 的回報原文，2026-11-14 -->

我在跑到一半時停下來。A、B、C 三個設定的實驗結果都還沒有。原因是在我送出的 slurm 工作被取消的同時，環境與大檔輸出被第三方刪掉了。需要你決定要不要重建。

## 發生了什麼（時間序）
1. 19:39 前後，群組磁碟配額（1536G，整個 ChihChungHsu 群組共用）被寫滿，我的 A、C 工作因 `Errno 122 Disk quota exceeded` 失敗。
   - 當時群組用量是 1534G／1536G，我占約 150G：pilot 41G、A/C 的 merged 模型各約 43G，加上 θ_s。
   - 我立刻刪掉 pilot 與 A/C 的 merged（群組降到 1399G），並把 `run_setting.py` 改成按需合併、用完即刪，再重送 A（job 26249）、C（26250）、B（26251，依賴 A）。
2. 19:47:24，26249 與 26250 被 `CANCELLED by 1047` 取消。1047 是我這個帳號，但不是我下的指令。
   - 同一時間之後，`/work/ChihChungHsu/cactus8603/icml/runs/neon-lm/`（資料切分、自生成資料、θ_s 全部、所有 dev 生成結果）整個消失。
   - 新建的 `/work/ChihChungHsu/cactus8603/icml/envs/neon` 也消失了。
   - 專案 repo 目錄 `/work/ChihChungHsu/cactus8603/icml27-s2-gated-sink-kv` 也不見了（我的 shell cwd 被刪）。
   - `icml/runs`、`icml/models` 底下的其他專案目錄也大量消失，`icml/ICML2027/` 在重組中。`icml/envs` 的內容在 19:49 還在被改動。
3. 看起來是同帳號下的另一個代理或使用者在做大掃除，可能因為配額，也可能是別的原因。我沒有確認。我不確定是誰，也不確定是不是故意針對我的實驗。

我沒有重建。理由是：
- 重建會再吃約 2.7 GPU 小時（A、C 的 dev 網格），而且如果清理還在進行，可能再被刪。
- 群組配額幾乎滿，我自己再寫幾十 GB 也可能害到別人。
- 這超出我的權限範圍，不是我該自己決定的取捨。

## 還在的東西
- 程式與設定（killtests 目錄沒被動）：`/work/ChihChungHsu/cactus8603/icml27-topic-search/killtests/neon-lm/`
  - `scripts/`：common.py、prep_data.py、gen.py（vLLM）、train_selfft.py、merge.py、nll_analysis.py（NLL 加判準 5）、mauve_eval.py、select_best.py、run_setting.py（整條流程）、job.sh、aggregate.py（出表與判準填表）、pilot_*.sh。
  - `results/` 只剩 `train_log_A.json`、`train_log_C.json`、`timing_A.json`、`timing_C.json`。
  - `logs/`：各工作的 slurm 輸出。
- 模型權重（`/work/ChihChungHsu/cactus8603/models/Qwen2.5-0.5B`、`Qwen2.5-1.5B`、`Qwen2.5-1.5B-Instruct`）與資料集（`icml/data/openwebtext`、`icml/data/gsm8k-hf`）還在。
- 環境需重建：`uv venv` 加 `uv pip install vllm mauve-text datasets accelerate scikit-learn faiss-cpu`。我裝出來的是 torch 2.14.0+cu130、vllm 0.33.0、transformers 5.20.2。
- 我沒存到 requirements-lock.txt，因為環境在我存之前就被刪了。

## 已量到的數字（來自 pilot 與被刪前的 log；原始檔已不在，只有 log 與我當時讀到的值）
- 原論文（arXiv 2510.03597）與官方 repo 確認：
  - 公式是 θ_neon = (1+w)θ_r − wθ_s，w>0。
  - w 掃描範圍約 0 到 3。VAR-d16 的最佳 w 約 1.0。
  - 合成資料量從 1k 到 750k；論文只說「少於 1% 訓練預算、低學習率」，沒給 LM 的步數。
  - 官方程式（https://github.com/VITA-Group/Neon）只有影像模型（EDM、VAR、xAR、IMM），沒有語言模型版本，所以 LM 版是我用 transformers 自己寫的。
- A 的自訓練（Qwen2.5-1.5B，OpenWebText 的 10,000 筆自生成，bs 32）：400 步用了 184 秒，loss 約 1.35 降到 1.30，第 312 步之後（第 2 個 epoch）降到約 1.0。
- A 的 dev 生成吞吐約 42k token/s，單一 (模型, 24 個取樣設定) 約 6 到 7 分鐘。
- C 的 dev 生成（200 題 ×16 次）每個取樣設定約 22 秒，一個模型約 9.5 分鐘。pilot 在 100 題上 base 的 pass@1 約 0.68 到 0.79。
- pilot 上 MAUVE 與 NLL 的程式碼都跑通：
  - base 的 NLL 約 2.534（T=1.0）。
  - 20 步、w=1 的 Neon 在 T=1.0、top-p=1.0 下 MAUVE 只有 1.9（500 筆）。同一設定在 T=0.8、top-p=0.9 下是 93.5。
  - 同一個 Neon 模型對 base 做逐位置最佳溫度擬合，T_i 平均 1.36、標準差 0.10，擬合後殘餘 KL 占原本 KL 的 14.8%。
  - 這些只是 20 步的 pilot，不能當正式結果。

## GPU 用量
- 全部合計約 3.3 GPU 小時（sacct 的 Elapsed 加總，含 pilot 與失敗的工作）。
  - 其中 A 約 1.41 小時、C 約 1.29 小時，但產物都已被刪。
- 配額是 30 GPU 小時，沒有逼近。

## 流程裡踩到的坑（重建時要帶著）
1. vLLM 0.33.0 的取樣器用 flashinfer 時要 nvcc 做 JIT。計算節點沒有 nvcc，所以要設 `VLLM_USE_FLASHINFER_SAMPLER=0`。我已寫進 `gen.py`。
2. `srun`／`sbatch` 要加 `-n1`，否則預設起 2 個 task；`-c` 最多 16。
3. 訓練時 padded batch 加 `attention_mask`（SDPA）會讓梯度出現 NaN，C 的第 41 步就爆。我已改成不傳 mask（右側 padding 加 causal 本來就不需要），測到第 60 步正常。`nll_analysis.py` 也同步改了。
4. mauve 在 fork 的進程池裡要把 `OMP_NUM_THREADS` 設成 1。否則 faiss 的 kmeans 吃滿 128 核、卡 12 分鐘以上。設了之後 pilot 全程 16 秒。
5. 磁碟：θ_s 用 fp32，每個設定 2 個約 12G。merged 模型不要全存。`run_setting.py` 已改成逐模型按需合併、用完即刪。

## 與 brief 不一致或未確認之處
- 自訓練資料在 A、B 上 400 步 ×32 = 12,800，超過 10,000 筆，所以第 2 個 epoch 重複用資料。C 也有類似情況（7,273 題）。
- C 的 dev 集：我用 GSM8K 訓練集裡隨機留出的 200 題，因此自訓練只用剩下 7,273 題。理由是 brief 說測試集取前 500 題、dev 取 200 題，這樣可避免與測試集重疊。
- w<0 的正向對照我對兩個 θ_s 都做，所以共 4 個，加上 10 個 Neon，每個設定共 15 個模型。
- 取樣：grid 一律 top_k 關閉、不加 repetition penalty；A、B 用 `ignore_eos` 強制生成 256 token。C 的自生成資料用 Qwen 預設取樣（temp 0.7、top-p 0.8、top-k 20、rep penalty 1.1），評測 grid 則沒有 top-k 與 rep penalty。
- dev 網格每個條件只用 1 個種子。
- 權重以 fp16 存放與推論，不是 bf16，因為 bf16 會吃掉小的 delta。我在 pilot 沒做 fp16 對 fp32 的 NLL 差異檢查。
- 沒有做 B（0.5B）的任何步驟。

## 需要你決定
1. 是誰刪了 `icml/runs/neon-lm` 與 `envs/neon`、取消了我的工作。如果是故意的整理，我該在哪個位置放大檔，才不會再被清掉？
2. 群組配額（1536G）幾乎滿。重建後我的峰值用量約 40G：θ_s 加生成結果，merged 按需。這樣可以接受嗎？
3. 要不要重建並重跑。預估再需要約 12 GPU 小時（A 約 4 小時、B 約 3 小時、C 約 4.5 小時），總計約 15 小時，仍在 30 小時內。我可以在你確認後，用已修好的 scripts 照 `scripts/run_setting.py` 的 stages 重跑。

REPORT.md 我沒有寫，因為沒有結果可填。
