# PREREG：多保真度資料歸因 停損實驗（回報 45；判準依回報 56 第 4 節與通用規則）

凍結日期：2026-11-14（送 slurm 前須 commit；事後任何修改須在本文件末尾「修改紀錄」記日期並標為**探索性**）。
所有數值門檻同時寫在 `configs/criteria.json`，`scripts/aggregate.py` 只讀該檔，兩者必須一致。

## 1. 唯一要回答的問題
CIFAR-10 ResNet-9 上，便宜歸因方法離精確 metagradient 還有沒有足夠空間，且殘差是否能被少量精確量測捕捉。

## 2. 設定與資料（固定）
- 模型／訓練：ResNet-9（寬度 W=80，約 10M 參數），SGD momentum 0.85，20 epoch、批次 500（2000 步），LR 三角排程，增強為 reflect-pad 4 裁切加水平翻轉；訓練資料為 CIFAR-10 全部 50,000 筆。與 simple-magic-jax 同超參數；實作改寫為純 JAX（見 README 的差異清單）。
- 訓練 seed：主模型 seed 0（初始化與資料順序固定；重訓只改資料權重）。TRAK 集成另用 seed 1–4。跨種子重訓用 seed 1。
- 查詢：測試集以 `query_seed=20261114` 排列後取前 200 個；階段 0 用其中前 50 個（k=50），階段 1 用全部 200 個（k=200）。查詢損失＝eval 模式（running BN）下的交叉熵。
- 移除子集：每次重訓每個樣本以 p=0.01 獨立移除（`mask_seed=1234`，第 j 個子集的遮罩是 j 的函數）。M=200（階段 0 與階段 1 共用同一批重訓，每個重訓模型都記錄全部 200 個查詢的損失，所以階段 1 不必重訓）。另有 4 次「無移除」重複估雜訊底線。
- 只存每個重訓模型在 200 個查詢上的損失與 2000 筆測試準確率，不存權重。
- 約定：Y、C 的元素皆為 ∂(查詢損失)/∂(樣本權重) 的（近似）值，故移除子集 S 的預測損失變化為 −Σ_{i∈S} Ŝ_{iq}。

## 3. 主指標（公式）
- 逐查詢 LDS_q(Ŝ)：Spearman(預測 Δ_{jq} = −Σ_{i∈S_j}Ŝ_{iq}，實測重訓後查詢損失 L_{jq})，跨 j=1..M；LDS = 對查詢平均。同軌跡 LDS（重訓 seed 與 Y 的訓練 seed 相同）為主指標；跨種子 LDS 只作階段 1 的附帶條件。
- 精確 Y：以 JAX 反向重播對每個查詢算 ∂L_q/∂w（SGD+momentum 展開；MAGIC 類，不調 metasmoothness）。
- 便宜方法（固定這 7 個變體，不得增減）：grad_dot（最終模型梯度內積）、tracin（5 個 epoch 檢查點 lr 加權）、trak1（單模型）、trak5（5 模型集成；投影維度 2048）、ekfac_auto／ekfac_1e-8／ekfac_1e-3（kronfluence，damping 為 0.1×平均特徵值／1e-8／1e-3）。UGN：Bergson 的 gnh-unroll 不支援 conv 與 SGD，**不在 CIFAR 上跑**（見 README），UGN 數字只能來自可選的 GPT-2 重現。「最佳便宜方法」＝這 7 個中同軌跡 LDS 最高者（7 選 1 的最大值有輕微樂觀偏誤，對「便宜方法已夠好就停」是偏保守方向，報告時註明）。
- 混合估計（離線，無自由度可調）：U＝查詢梯度 Gram G_V（k×k，最終模型的查詢損失梯度內積）的前 B 個特徵向量，B = round(0.2·k)（k=50→B=10；k=200→B=40）；Spell 基線 Ŷ_Spell = Y U Uᵀ；混合 Ŷ = Y U Uᵀ + α C (I − U Uᵀ)，α = ⟨YU, CU⟩/⟨CU, CU⟩（純量、不收縮、只用量測子空間）；C＝最佳便宜方法。隨機 U 版本只作次要報告。這是 Spell 的「以 G_V 貪婪選方向」之簡化重現，不是其原程式（原程式未公開）。
- κ = ‖YU‖_F²/‖Y‖_F²；殘差 R = Y − αC，κ_res = ‖RU‖_F²/‖R‖_F²。
- 增益 gain = LDS(混合) − max(LDS(Spell 基線), LDS(最佳便宜方法))。

## 4. 閘門（依序；前三關失敗＝「未能測試」，不判題目；限 3 個工作天修，修不好記為可行性停損）
1. 程式正確性（已知答案）：
   - `gate_mlp.py`：MAGIC 類 metagradient 在小 MLP（float64）上對中央有限差分的最大相對誤差 < 1e-4，且對實際留一重訓的 Pearson 平均 > 0.95。
   - `gate_hybrid.py`：三個恆等式——α=0 時混合＝Spell；B=k 時混合＝Y（任意 C、任意 α）；C=Y 時輸出 Y 且 α̂=1；另檢查投影冪等與 α̂ 為最小平方解。
   - 匯出保真度：JAX 與 PyTorch 前向 logits 最大絕對差 < 5e-3（`cheap_torch.py` 內檢查）。
2. 重現：CIFAR 上精確 Y 的同軌跡 LDS（k=50、M=200、移除 1%）≥ 0.88（參照 Spell 論文表 1 的 MAGIC 0.909，容許差 0.03）。
3. 雜訊底線：無移除重複的查詢損失標準差 / 移除重訓的查詢損失標準差，查詢平均 ≤ 0.5。
4. 假設檢定（下兩節）。

## 5. 階段 0（k=50、M=200；約 20–25 GPU 小時）判準，依序檢查，第一個成立者即結論
- STOP（便宜方法已夠好）：最佳便宜方法 LDS ≥ 0.8。
- STOP（殘差沒有可捕捉結構）：κ_res/κ < 0.5。
- STOP（混合沒有增益）：gain < 0.04。
- 否則 CONTINUE，進階段 1。
階段 0 的最佳便宜方法在此刻固定，階段 1 沿用，不重選。

## 6. 階段 1（僅階段 0 為 CONTINUE 才做）
擴到 k=200（精確 Y 再算 150 個查詢），B=0.2k=40 固定。另加跨種子重訓 100 個（seed 1，沿用 j=0..99 的子集）。逐查詢配對 bootstrap（對 200 個查詢重抽 10,000 次，seed 0，雙尾 95% 百分位 CI）。
- PASS：混合比 Spell 基線與最佳便宜方法各高 ≥ 0.08，且兩個差的 CI 下界皆 > 0.03；並且跨種子 LDS 混合 − 最佳便宜 ≥ 0。
- 增益在 [0.04, 0.08)：只看可靠度那條——(i) 最佳便宜方法逐查詢 LDS 的折半信度（偶數／奇數子集，Spearman-Brown 校正）≥ 0.7；(ii) 可靠度估計 ρ̂_q（量測子空間內 Y 與 C 的逐查詢餘弦）與逐查詢真實 LDS 的 Spearman ≥ 0.6，且大於兩個平凡預測量（基準模型查詢損失、‖C_q‖）的 |Spearman|；加跨種子條件。三者成立為 PASS（標註「僅可靠度路線」）。
- 灰區（三態之一，只准一次、事先定好的加碼）：(a) 增益 ≥ 0.08 但 CI 下界 ≤ 0.03；(b) 增益在 [0.04,0.08) 且 (i) 成立、(ii) 的 Spearman 在 [0.45, 0.6)。唯一加碼：再跑 200 個重訓（j=200..399，M=400），以相同規則重判一次，之後不得再加碼。
- FAIL：其餘情形（含增益 < 0.04，或跨種子輸給便宜方法）。
- 比較次數：階段 1 的 PASS 是「兩個比較同時成立」的合取，不需 α 校正；可靠度路線僅在增益落入 [0.04,0.08) 時啟用，所以沒有「m 擇一」的情況。

## 7. 可重試次數與成本上限
- 一般：閘門失敗可修程式重跑，總共最多 3 個工作天；每次重跑（含失敗）的 job id 與 log 全留，不刪。
- 超參數：無（所有超參數已在上面固定）；不得因結果不佳調整 B、U 選法、α 收縮、查詢集、p_drop 等；要改只能記為探索性。
- GPU 小時上限：階段 0 約 20–25（推估）；若 10 分鐘吞吐實測（`s0_bench.sbatch`）顯示超過推估 2 倍，先停下重排，不照原計畫送出。全題上限 80 GPU 小時（含階段 1、跨種子、灰區加碼）。磁碟上限 80 GB。

## 8. 判讀流程
`aggregate.py` 機械輸出（結論寫入 `results/<tag>/verdict_stage{0,1}.json`）→ Opus 對照本文件審 → 使用者定案。實作者（Sonnet）與提案者不判。

## 修改紀錄
- 2026-11-14（接手實作者，**非探索性、無任何門檻或設計變更**，僅基礎設施）：(1) attrib-torch 改重用 t2irare 的 site-packages（neon 已刪除）並補裝 einops；(2) 新增 `scripts/check_qchunk.py`（qchunk=1 與 vmap 版精確 Y 一致性檢查，非判準）；(3) `exact_y.py` 啟動印 `jax.devices()`，`s0_bench.sbatch` 設 `REQUIRE_GPU=1` 在無 GPU 時中止；(4) README 重寫補全。已核對 `configs/criteria.json` 與本文各數值一致（1e-4／0.95／0.88／0.5／0.8／0.5／0.04／0.08／0.03／0.7／0.6／0.45／0／B_frac 0.2／bootstrap 10000 seed 0／7 個變體）。
- 2026-11-14（Opus 審查，**非探索性、無任何門檻數值或判準邏輯變更**；詳見 `REVIEW.md`）：
  (1) `aggregate.py`：精確 Y 只接受由 0 起、首尾相接、不重疊的區塊（舊版 glob 全收，多一個抽查檔會靜默錯接欄位）；7 個便宜變體缺一即不判（舊版在子集合上選最佳）；任何判讀量為 NaN/inf 時輸出 `INVALID(程式錯誤)` 並 exit 4（舊版 NaN 會讓三個 STOP 條件都為假而輸出 CONTINUE，或讓 NaN 方法被選為「最佳」）；MLP 與匯出保真度閘門改依 criteria.json 的數值重新判定；階段 1 需階段 0 結論為 CONTINUE；可靠度路線在缺跨種子重訓時輸出「未完成」而非 FAIL；輸出加 git hash 與 criteria.json 的 sha256；階段 0 另輸出**僅描述、不參與判讀**的配對 bootstrap CI 與折半相關。
  (2) `configs/criteria.json` 加 `gates.export_fidelity_max_abs_logit_diff = 5e-3`（數值與本文 §4 相同，原本寫死在 `cheap_torch.py`），`cheap_torch.py` 改讀此值。
  (3) GPU 防呆：`common.sh` 設 `REQUIRE_GPU=1`；`jax_common.setup` 與 `cheap_torch.py` 在看不到 GPU 時中止（舊版 torch 會靜默改用 CPU）。
  (4) `s0_bench.sbatch` 加 torch（cu124）CUDA 檢查與 42 GB bf16 TRAK 投影矩陣配置測試；`exact_y.py --bench` 先暖機再以差分計時（舊版把 JIT 編譯與段重算攤進 20 步，外推高估）。
  (5) `check_qchunk.py` 改為從包含 [qstart,qend) 的已存區塊切片（舊版找 `Y_0_2.npy`，README 的 `--qend 2` 用法必定失敗）。
  (6) 新增 `scripts/check_resnet_fd.py`（非判準）：ResNet9 含 BN/增強的 metagradient 對有限差分，float64 CPU 最大相對誤差 1.8e-8。
