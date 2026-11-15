# biased-gen-cfg：「條件擴散模型偏差泛化的兩個時鐘」第 1 週停損實驗

**已淘汰。** 以下是原實驗的執行紀錄。[歷史程式與設定](https://github.com/cactus8603/icml27-topic-search/tree/ed59264408851afa1e4b9689aa2a1660f079b3ee/killtests/biased-gen-cfg)

對應回報：`reports/2026-11-14-53-deepcheck-biased-generalization-cfg.md`（題目、原判準）、`reports/2026-11-14-52-relaxed-icml-scan.md` 候選 2、
`reports/2026-11-14-56-killtest-criteria-review.md` 第 1 節（判準審查，**以此為準**）。凍結判準見 `PREREG.md`；數字集中在 `configs/criteria_week1.json`。
原作：arXiv 2603.03469，程式 github.com/davide-beltrame/biased-generalization（clone 於 `/work/ChihChungHsu/cactus8603/icml/ext/biased-generalization`，含 `wdmdm` 子模組；未改動）。

最新評估與後續判讀見 `../../codex/2026-11-14-two-clocks-week1-codex.md`。作業編號見 `../SUBMITTED.log`。

## 目的

第 1 週兩件事（53 號 §6）：
- (a) 重現原作 CelebA 32×32 灰階的偏差曲線（6 個模型、不相交資料），確認「偏差最低點早於 test loss 最低點」。
- (b) 階層資料加類別條件（根符號）與 condition dropout（p=0.1），用 BP 算精確的條件／無條件／CFG 分數，輸出 oracle 偏差曲線。

## 53 號第 1 週停損判準（原文，僅供對照；已被 56 號與 PREREG 取代）

> (a) 兩個最低點的 epoch 差 > 2 倍種子標準誤；(b) 程式能算出條件分支與 CFG oracle 的偏差曲線。(a) 失敗代表 pipeline 有問題，先修再往下。

56 號修正：t\*(w) 定義為 sample-split 分數距離曲線最低點 epoch（平滑方式事先固定）、checkpoint 對數間距相鄰 ≤10%、唯一主指標 `log[t*(w=3)/t*(w=0)]`、
(d) 每個 n_c 6 份不相交資料 × 1 seed、三態結果、閘門（BP 暴力後驗與 Σ_c p(c)·條件分數＝無條件分數、正控制、負控制）、(a) 重現不過記「未能測試」。

## 檔案

| 路徑 | 說明 |
|---|---|
| `PREREG.md` | 凍結判準、資料／seed、閘門順序、成本、待審項目 |
| `configs/week1.env` | 路徑與超參數（env 路徑、資料、epoch 數等） |
| `configs/criteria_week1.json` | 判準數字（aggregate 唯一讀取處） |
| `scripts/prep_celeba.py` | 從 HF `nielsr/CelebA-faces` 轉出 `CelebA32.pt`（前 16384 張 + 最後 2048 張） |
| `scripts/train_celeba.py` | CelebA U-Net 訓練（原作模組；checkpoint 改為對數間距 ≤10%） |
| `scripts/eval_celeba.py` | 測試損失 + 成對 ε cosine 相似度（所有 checkpoint） |
| `scripts/hier_cond.py` | 階層資料條件版：精確條件／無條件 BP、CFG 組合、擴散、含類別嵌入的 transformer |
| `scripts/test_bp.py` | BP 單元測試（CPU，輸出 `results/bp_unit_test.json`） |
| `scripts/train_hier_cond.py` | 階層資料條件版訓練（condition dropout） |
| `scripts/eval_hier_oracle.py` | BP oracle 偏差曲線與兩模型 sample-split 距離 |
| `scripts/aggregate_week1.py` | 依判準輸出 PASS／FAIL／未能測試與數字 |
| `scripts/schedule.py` | checkpoint 排程（相鄰比值 ≤1.10） |
| `scripts/w1_*.sbatch`, `scripts/gen_hier_data.sh`, `scripts/common.sh` | slurm 與資料產生 |

## 環境

既有環境都不能直接用（c5 缺 torchvision/scipy/matplotlib；neon、vllm033 缺 matplotlib/natsort；s2 缺 torchvision）。新建 `envs/biasgen`（原作 repo 的 setup_env.sh 沒有執行，只讀過）：

```bash
export TMPDIR=/work/ChihChungHsu/cactus8603/icml/tmp PIP_CACHE_DIR=/work/ChihChungHsu/cactus8603/icml/tmp/pip-cache
module load miniconda3
conda create -y -p /work/ChihChungHsu/cactus8603/icml/envs/biasgen --override-channels -c conda-forge python=3.11 pip   # 不用預設 channel，避開 conda ToS 提示
E=/work/ChihChungHsu/cactus8603/icml/envs/biasgen/bin/pip
$E install torch==2.8.0 torchvision==0.23.0 --index-url https://download.pytorch.org/whl/cu128   # 與 c5 相同的 torch 2.8.0+cu128
$E install numpy scipy numba matplotlib natsort tqdm pillow pyarrow huggingface_hub pandas pyyaml
```
`HF_HOME=/work/ChihChungHsu/cactus8603/icml/hf_home`。原作的 pytorch-fid、seaborn、PyWavelets 第 1 週用不到，未安裝（原作 `scripts/utils.py` 會 import seaborn，所以本實驗不用它的 utils，只用 `modules.embeddings`、`modules.noise_schedules` 與 BP 檢查用的 numba 模組）。

## 資料準備與核對

1. CelebA（HF 快取約 1.39 GB，輸出 `CelebA32.pt` 約 75 MB）：
   ```bash
   source configs/week1.env
   $BIASGEN_ENV/bin/python scripts/prep_celeba.py --out-dir $CELEBA_DATA_ROOT
   ```
   現用 `icml/data/celeba32/CelebA32.pt` 的 18,432 張張量逐張與 HF 快取重轉結果完全相同。
   已確認 HF 影像順序與原作自然檔名順序不同。全庫影像的 CRC32 與位元組長度（含重複次數）相符，前 3 列也與定位到的原始 JPEG 逐位元相符。
   六塊訓練資料和測試保留集之間沒有相同 JPEG 簽章。本次 (a) 使用不同的影像切分做統計重現。版本與核對方法見 PREREG §8。
2. 階層資料已生成（40000 筆），正式訓練 26280 已讀入並產生 checkpoint。原始葉節點為 `(16, 40000)` 的 `int8`，訓練讀入後形狀為 `(40000, 16)`。
   重建可用 `bash scripts/gen_hier_data.sh`（`WORKERS` 可調）；文法 M 與原作 1M 樣本檔相同（seed 0）。
3. 先跑 `python scripts/test_bp.py`（CPU，數秒；已跑過，全過）。

## 送出指令（寫出來，**不要在審查前執行**）

```bash
cd /work/ChihChungHsu/cactus8603/icml27-topic-search/killtests/biased-gen-cfg
# (a) CelebA：6 個 task 各 1 GPU；跑完後再送 eval
J1=$(sbatch --parsable scripts/w1_celeba_train.sbatch)
sbatch --dependency=afterok:$J1 scripts/w1_celeba_eval.sbatch
# (b) 階層資料條件版：3 個 task 各 1 GPU
J2=$(sbatch --parsable scripts/w1_hier_train.sbatch)
sbatch --dependency=afterok:$J2 scripts/w1_hier_eval.sbatch
# 全部完成後（登入節點、CPU）
$BIASGEN_ENV/bin/python scripts/aggregate_week1.py
```
先做 10 分鐘吞吐實測（PREREG 要求）：對 `w1_celeba_train` 與 `w1_hier_train` 各先送 task 0 並看 log 的 `epoch N ... s`，估總時間，超過下表 2 倍就重排。
訓練可續跑（CelebA 找最大 epoch checkpoint + `opt.pt`）；hier 訓練**不支援續跑**（逾時要整個重跑，已把 time 抓寬）。

## 縮減選項（若 5000 epochs 太貴）

原作 CelebA 預設 5000 epochs（n=1024、batch 512 → 約 10,000 步）。縮減：`CELEBA_EPOCHS=2500`（configs/week1.env）。風險：n=1024 的 test loss 最低點若落在 2500 之後，網格邊界會截斷，aggregate 會標記 `test_loss_min_at_grid_end`。
階層資料縮減：`HIER_EPOCHS=10000`（原作 n=5000 為 20000）；同樣要看最低點是否在網格末端。**縮減都偏離原作設定，PREREG 要註記。**

## 預估 GPU 小時（皆為推估；FLOP 以 `torch.utils.flop_counter` 在 CPU 實測，吞吐為假設，未在 GPU 上實測）

| job | 規模 | 預估 | `--time` |
|---|---|---|---|
| celeba_train（×6） | U-Net 4.06M 參數，前向 1.2 GFLOP/張；每模型 10k 步×512 張，約 18.5 PFLOP；H200 fp32 有效 12–20 TFLOPs 加上小 kernel 效率損失 → 每模型 0.5–1.5 h | 3–9 h（6 個） | 6 h |
| celeba_eval（×1） | 6 模型 × 72 checkpoint × 約 47k 次前向，約 25 PFLOP | 0.5–1 h | 3 h |
| hier_train（×3） | 16.9M 參數 transformer，前向 0.54 GFLOP/序列；20000 epochs×10 步=200k 步，約 165 PFLOP；bf16、eager，吞吐受 Python 開銷限制 → 每個 1.5–3 h | 4.5–9 h（3 個） | 10 h |
| hier_eval（×1） | 3 模型 × 71 checkpoint；BP 在 float64 | 0.3–1 h | 1.5 h |
| 合計 | | 約 8–20 GPU h（53 號估 (a) 5–10、(b) 10–20） | |

第 2 週追加（判準見 `PREREG.md` 第 8 節；程式與 sbatch 已備好、未送出）：K=6 補 pick 3–5 三個 run（4.5–9 h，`w2_hier_train_k6.sbatch`）、
K=1 常數條件負控制 × N 個（每個 1.5–3 h，6 個約 9–18 h；`w2_hier_train_k1.sbatch`）、類別 0 專用無條件模型 × N 個（每 epoch 約 10 步、每步 batch 85，約 1.5–3 h，未實測；`w2_hier_train_u833.sbatch`）、
6 個 run 一起重評（`w2_hier_eval.sbatch`，含 `aggregate_week2.py`）。N 由第 1 週 3 個階層 run 的實測平均 GPU 小時 h 決定（h ≤ 2.0 時 N=6，h > 2.0 時 N=4 並標「低檢定力」）。使用者決定入池後再安排 (d) CIFAR，仍須檢查總 GPU 時數預算。
**已刪除 n=70000 負控制。** 送 (d) 前若「已用 GPU 小時 + 依 (d) 實測吞吐推估」> 115 GPU 小時，先停下來回報使用者。
訓練用 `train_hier_cond_w2.py`（新增 `--const-label`、`--class-only`；預設值下與 `train_hier_cond.py` 逐位元相同），第 1 週訓練檔未動。

## 預估磁碟（本題上限 80 GB，含 env、checkpoint、結果）

| 項目 | 大小 |
|---|---|
| `envs/biasgen`（torch 2.8+cu128 與 nvidia 函式庫） | 約 8–9 GB（安裝完成後請 `du -sh` 確認） |
| HF 快取（CelebA 轉檔來源，轉完可刪） | 約 0.9 GB（暫時） |
| `CelebA32.pt` | 75 MB |
| CelebA checkpoint（6 模型 × 72 個 × 16 MB；`opt.pt` 每模型覆寫 32 MB） | 約 7 GB |
| 階層資料 npy | 約 20 MB |
| 階層 checkpoint（fp16，34 MB × 71 × 3 個） | 約 7.3 GB（週 2 再 +7.3 GB） |
| `results/*.npz` | < 1 GB |
| 第 1 週合計 | 約 25 GB；含週 2 約 50 GB |

checkpoint 放 `results/ckpt/`（`.gitignore` 排除）。只保留判準需要的：CelebA 只存 `Model_epoch_*`（本包的 `train_celeba.py` 不另存原作的 `Model_<step>`）。評估完成並記錄 aggregate 結果後可刪 `results/ckpt/**/opt.pt`。

## 與原作的差異（審查請看）

- CelebA 訓練不用原作 `run_Unet.py`：原作 `Diffusion.train` 把 `max_epoch=5000` 與 checkpoint 網格 `linspace(10,4999,50)` 寫死，無法滿足「對數間距相鄰 ≤10%」。`train_celeba.py` 用同一份 U-Net、擴散、損失、優化器、資料切分與中心化，但：資料整份放 GPU 自己洗牌、不做每 10 epoch 的 validation 與出圖、不生成樣本。這些不影響優化軌跡，但**同 seed 位元級重現未驗證**。
- `eval_celeba.py` 不是原作 `compare_scores.py`/`loss_compute.py`：輸入集合改為 1024 張（原作 `-Ns 1000` 實際只用 1 個 512 batch）、共同 mean 取前 6144 張的平均、測試損失用 t=5,15,…,995 各 256 張固定噪聲（原作 eval_N=50）。
- 階層資料訓練不用原作 `train_transformer.py`（它需要 seaborn、torch.compile、DataLoader 在 CPU 逐筆噪聲化），改用 GPU 批次噪聲化；模型加類別嵌入（null=索引 q）；bf16 autocast 取代 fp16 GradScaler；checkpoint 以 fp16 存檔（評估時轉回 float，位元不等於訓練時權重）。排程、LR、batch、損失與原作相同（`lr_schedule=default`）。
- 階層 oracle 是自寫 BP（機率空間、float64），已與原作 numba BP（無條件）、暴力窮舉（條件／無條件）核對到 1e-15。CFG 以 x0-hat 空間線性組合 `(1+w)p_c − w p_u` 實作，w=0 為條件模型（與 53 號 w∈{0,1,3} 的記法一致；不是 `w` 對應 ε_u+w(ε_c−ε_u) 的另一種記法）。

## 已知風險與未確認事項

1. CelebA 使用 HF 列順序切分，與原作 `img_align_celeba` 按自然檔名切出的樣本不同。判讀 (a) 時須保留這項差異。
2. 所有 GPU 小時與 `--time` 都是推估；hier 訓練不能續跑，逾時代價高。
3. 單一網格解析度：epoch 邊界問題（最低點落在網格頭尾時 t\* 被截斷）。
4. 3 個 run 只能做管線檢查；(c) 主指標在 R=3 時 bootstrap 幾乎沒有意義，aggregate 已標為探索性。
5. 週 2 的負控制已改為 K=1（成本約等於一般 n=5000 run）；batch 85 的吞吐與 1.10 回升門檻都沒有實測依據。
6. CelebA 的 sample-split 距離使用 `1−cos(ε_A, ε_B)`（輸入來自 A 的訓練集）；階層資料主指標也改用 ε 空間 `1−cos`（條文 1，逐 t 正規化並套用 `configs/t_mask_hier.json`；x0-hat `pair_cos` 只留存不判定）。(a) 用原始平均、階層資料用逐 t 正規化，尺度不同，不可互比。
7. 磁碟：/work 群組額度吃緊時，checkpoint 寫入失敗請停下回報，不要重試。
8. 53 號「(a) 失敗代表 pipeline 有問題」與 56 號「記為未能測試」一致；但若 n=1024 本身在我們的 6 模型設定下不重現原作 U 形，無法區分是 pipeline 問題還是原作現象不穩，需審查者決定。

## Smoke test 結果（CPU，登入節點；用合成資料或極小設定，僅確認程式能跑）

見最終回報。
