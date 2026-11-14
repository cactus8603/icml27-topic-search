# 多保真度資料歸因 — 停損實驗（回報 45）

準備日期 2026-11-14。**狀態：程式與 slurm 腳本已寫好、CPU smoke test 完成（見第 9 節）；尚未送出任何 GPU 工作。** 判準以 `PREREG.md`（依回報 56 第 4 節）為準；`configs/criteria.json` 與它一致，`scripts/aggregate.py` 只讀該檔，改門檻只改這一處。

## 1. 目的與對應回報
回報 45：便宜歸因分數當 control variate、少量精確 metagradient 校正殘差、加逐查詢可靠度。本實驗只做**第 1 週停損**：CIFAR-10 ResNet-9 上便宜方法離精確 Y 還有多少空間（最佳便宜 LDS ≥ 0.8 就停），以及殘差能量比例 κ_res 與 Y 本身 κ 的比較。
- 回報 45 原判準：「CIFAR k=100～200、B ≤ 0.2k 時混合法比 Spell 和最佳便宜方法都高至少 0.08 ……；κ_res 不小於 κ 的一半」；「直接停：CIFAR 上最佳便宜方法已有 0.8 以上」。
- 回報 56 改寫為階段 0（k=50、M=200：≥0.8 停；κ_res/κ<0.5 停；增益<0.04 停）→ 通過才階段 1（k=200，PASS＝逐查詢配對 bootstrap 兩個差 ≥0.08 且 CI 下界 >0.03，另有可靠度路線與跨種子條件）。**以新版為準**，全文在 PREREG §5–6。
- GPT-2/WikiText（原判準 2）不在第 1 週內，只附可選重現腳本（第 7 節）。

## 2. 外部程式（`/work/ChihChungHsu/cactus8603/icml/ext/`，皆 `git clone --depth 1`；沒有執行任何安裝腳本）
| 套件 | 授權 | 狀況 |
|---|---|---|
| simple-magic-jax | CC BY-NC-SA 4.0 | 讀過 `main.py`、`cifar.py`。不直接用：import 時下載 CIFAR 到 `/tmp`、每步把狀態搬 CPU（2000 步約 200 GB 主記憶體）、每次重播一個查詢。`scripts/mf_core.py` 為改寫版（同授權、非商業學術、標註來源）。 |
| Bergson | MIT | 讀過 `gnh_unroll/{unroll,cli}.py`、`magic/`、排行榜 yaml。**gnh-unroll 不支援 conv**：`unrolled_names()` 只收 `nn.Linear` 與 HF `Conv1D`（GPT-2 轉置線性層），不收 `nn.Conv2d`；梯度-牛頓步 `_gauss_newton_and_scores` 要 `input_ids/labels/logits[:, :-1]` 與 vocab softmax（LM 專用）；展開式是 AdamW 的 (γ,ν) 遞迴，不是 SGD+momentum；介面綁 HF tokenizer/datasets。`bergson magic` 同為 LM 訓練器。**所以 CIFAR 上沒有 UGN 對手**，便宜方法只有 TRAK、EK-FAC、gradient dot（加 TracIn）；要 UGN-on-CIFAR 得自己寫 SGD 版 GGN 展開，不在本週範圍。 |
| kronfluence | Apache-2.0 | 讀過 API 與 `examples/cifar`（含 ResNet-9 EK-FAC）。支援 Conv2d/Linear，BN 不追蹤。以 `.pth` 路徑使用，未安裝。 |
| TRAK | MIT | 只 clone 參照。未使用其套件（需編譯 `fast_jl` CUDA）；`cheap_torch.py` 以同公式自寫（margin 輸出函數、bf16 高斯投影 d=2048、(ΦᵀΦ+λI)⁻¹、(1−p) 權重、5 模型集成）。 |

## 3. 環境（`/work/ChihChungHsu/cactus8603/icml/envs/`）
既有 envs（uv venv 風格，無 conda）都沒有 JAX 與 kronfluence，**分兩個環境**（JAX 與 PyTorch 的 cu13 輪子易互撞；JAX 負責重訓與 Y，PyTorch 負責 kronfluence/TRAK）：
- `attrib-jax`：`jax[cuda13]` 0.12.0.1、numpy、scipy、optax、equinox（後兩者未用到）。約 4 GB。
- `attrib-torch`：**重用 `t2irare` 環境（torch 2.5.1+cu124、numpy 1.26.4、scipy、accelerate、safetensors、tqdm）的 site-packages**，以 `lib/python3.12/site-packages/zz_reuse_t2irare.pth` 指向它（只讀、不複製，省約 6 GB）與 `icml/ext/kronfluence`（kronfluence 以 .pth 使用、未安裝）；另在本環境內裝 `opt_einsum`、`einconv`、`einops`（`--no-deps`）。**修復紀錄（2026-11-14）**：原本 .pth 指向已刪除的 `envs/neon`，環境壞掉；改指 t2irare 後 `torch`、`kronfluence.analyzer`、`cheap_torch.py`（完整 smoke）皆可用；補裝 einops（kronfluence 的 conv2d 模組需要）。風險：t2irare 若被升降級或刪除會再度壞掉；torch 為 cu124 輪子，H200（驅動需支援 CUDA 12.4+）上 cheap_torch 首次執行請看 `torch.cuda.is_available()`。要隔離就在 attrib-torch 內另裝 torch（約 6 GB），目前不需要。
- 建立：`pip install --target` 取得 uv（`icml/tmp/uvtool`）→ `uv venv` → `uv pip install`；所有快取與 TMPDIR 指向 `icml/tmp`，沒寫 `/tmp`。紀錄：`icml/setup-logs/attrib-jax{,2}.log`。第一次安裝因額度滿而損毀，已 `--reinstall-package` 修復。
- JAX 與 PyTorch 以檔案交換：`train_main.py` 存最終模型/檢查點 npz（OIHW 權重），`cheap_torch.py` 載入並先驗證前向 logits（閘門）。
- `HF_HOME=/work/ChihChungHsu/cactus8603/icml/hf_home`（本實驗不用 HF 資料）；CIFAR-10 在 `icml/data/cifar10`（sha256 與官方相符）。

## 4. 資料夾
```
PREREG.md  README.md  .gitignore
configs/  stage0.json（正式） smoke.json（CPU 極小） criteria.json（全部門檻）
scripts/  mf_core.py（精確 metagradient, JAX）  hybrid.py（混合估計）  train_main.py  retrain.py
          exact_y.py  cheap_torch.py  aggregate.py  gate_mlp.py  gate_hybrid.py  check_qchunk.py  common.py  jax_common.py  common.sh
          s0_bench/train/retrain/exact/cheap.sbatch  s1_exact/crossseed/addon.sbatch  optional_bergson_gpt2.sbatch
results/  gates/*.json、<tag>/verdict_stage*.json、retrain_query_losses_*.npy（小檔）
logs/     slurm 輸出
```
大檔在 `icml/runs/multifid/stage0/{models,exact,retrain,cheap}`。重訓每個模型只存 201 個浮點（200 個查詢損失＋準確率），不存權重。

## 5. 預估（**皆為推估，未在 GPU 實測**；先送 `s0_bench` 實測，超過 2 倍就重排）
| 步驟 | 腳本 | GPU 小時 | 磁碟 |
|---|---|---|---|
| 吞吐實測 | s0_bench | ~0.3 | 0 |
| 主模型＋4 集成模型訓練（含 5 個 TracIn 檢查點） | s0_train | ~0.3 | ~0.4 GB |
| M=200 重訓＋4 雜訊重複，每次 2000 步約 1–3 分鐘 | s0_retrain（8 tasks） | 4–10 | <1 MB |
| 精確 Y k=50（5 tasks×10 查詢，vmap 反向）每 task 0.3–1.5 h | s0_exact | 2–8 | 10 MB |
| 便宜方法 7 變體（逐樣本梯度、TRAK×5、EK-FAC） | s0_cheap | 2–4 | ~0.3 GB |
| **階段 0 合計** | | **約 10–25**（回報 56：20–25） | **約 1 GB** |
| 階段 1：Y 再 150 查詢、跨種子 100 重訓、（灰區 +200 重訓） | s1_* | 8–25／3–5／4–10 | <0.1 GB |
全題上限 80 GPU 小時、80 GB（PREREG §7）。

## 6. 送出指令（寫出來，**未執行**）
```bash
cd /work/ChihChungHsu/cactus8603/icml27-topic-search/killtests/multifidelity-attribution
# 閘門 G1/G2（CPU，已過）：JAX_PLATFORMS=cpu ../../../icml/envs/attrib-jax/bin/python scripts/gate_mlp.py ; envs/attrib-torch 的 python scripts/gate_hybrid.py
sbatch scripts/s0_bench.sbatch                       # 先看 logs/mf-bench-*.out：jax.devices() 是否為 GPU、各項耗時
J1=$(sbatch --parsable scripts/s0_train.sbatch)
J2=$(sbatch --parsable --dependency=afterok:$J1 scripts/s0_cheap.sbatch)
J3=$(sbatch --parsable scripts/s0_retrain.sbatch)    # 與訓練無依賴
J4=$(sbatch --parsable scripts/s0_exact.sbatch)
# 全部完成後（CPU，數秒）：
/work/ChihChungHsu/cactus8603/icml/envs/attrib-torch/bin/python scripts/aggregate.py --cfg configs/stage0.json --stage 0
# 只有結論為 CONTINUE 才送：
sbatch scripts/s1_exact.sbatch; sbatch scripts/s1_crossseed.sbatch
.../attrib-torch/bin/python scripts/aggregate.py --cfg configs/stage0.json --stage 1
# 灰區唯一加碼：sbatch scripts/s1_addon.sbatch；之後再跑一次 aggregate --stage 1
```

## 7. 可選：Bergson GPT-2 排行榜重現（UGN 0.819／MAGIC 0.931）
`scripts/optional_bergson_gpt2.sbatch`（未驗證、不屬階段 0）。需先建 `attrib-bergson`（重用 t2irare 的 site-packages（neon 已刪除）＋補裝 jaxtyping、natsort、peft、petname、pyyaml、simple-parsing、torchopt、bitsandbytes＋`--no-deps -e ext/bergson`；**尚未建立**，先讀 pyproject）。原設定 8 卡，腳本改為單卡會更慢；100 個重訓模型各約 0.5 GB 須清理。成本 15–30 H200 小時以上（推估）。建議階段 0 之後再說。

## 8. 與 simple-magic-jax 的差異（審查請看）
1. 純 JAX、無 equinox；BN 手寫（batch 統計、running momentum 0.99、eps 1e-3）；init 同 eqx 預設（均勻 ±1/√fan_in）。非逐位元相同。
2. 整個訓練步（含 BN running stats 更新）一次 `jax.vjp`；狀態＝(params, momentum, bn buffers)；查詢損失在 eval 模式。
3. 資料排列與增強 key 是訓練 seed 的函數，重訓只改資料權重（loss＝Σℓ_i w_i/BS；被移除樣本仍進 BN 批次統計，與原作相同）。
4. 檢查點每 50 步一個、反向時重算段內狀態；一次處理 10 個查詢（vmap）。
5. 移除遮罩為 Bernoulli(0.01)（約 500 筆）。
6. 精度：預設 matmul 精度（H200 上可能是 TF32）；`matmul_precision` 設 "highest" 較準較慢。

## 9. Smoke 紀錄（2026-11-14，CPU、`configs/smoke.json`：寬度 4、160 筆、10 步、k=5、M=12；全程 `JAX_PLATFORMS=cpu`、nice）
腳本 `icml/tmp/smoke.sh`、`smoke2.sh`；輸出 `results/smoke/`。
- `gate_mlp.py`：有限差分最大相對誤差 4.7e-10、留一重訓 Pearson 0.998（通過）；`gate_hybrid.py` 三個恆等式與冪等、最小平方檢查通過。
- `train_main.py`、`retrain.py`（含 vmap 外的無移除重複）：通過。
- `exact_y.py`（qchunk=3，vmap 版，兩個 chunk）：通過；`--bench`：通過（467 ms/步，CPU，僅供流程驗證，與 GPU 無關）。
- `check_qchunk.py`（qchunk=1 對 qchunk=3）：相對誤差 1.0e-7，通過（門檻 1e-4；GPU 預設精度下請改 `--tol 1e-2` 抽查 2 個查詢）。
- `cheap_torch.py` 完整流程（grad_dot、tracin、trak1、trak5、ekfac×3、匯出保真度）：通過；保真度 max|Δlogit|=1.5e-8（門檻 5e-3）。
- `aggregate.py --stage 0 --smoke`：閘門 ALL_PASS，輸出 `verdict_stage0.json`（smoke 數字毫無統計意義，結論 STOP 僅表示流程走通）。
- 修復：attrib-torch 因 neon 被刪而損壞，改重用 t2irare（見第 3 節）；`exact_y.py` 新增啟動時印 `jax.devices()`，`REQUIRE_GPU=1` 且只有 CPU 時中止（`s0_bench.sbatch` 已設）。
- **未覆蓋**：任何 GPU 上的行為（jax GPU 偵測、記憶體、吞吐、TF32 精度、kronfluence/TRAK 在寬度 80 的耗時）。

## 送出前檢查（未執行；全部需人工確認後才送）
1. `quota -g -s` 剩餘額度充足（本題上限 80 GB，階段 0 約 1 GB）。
2. 先送 `s0_bench`，看 log 第一行 `jax.devices()` 為 GPU（`REQUIRE_GPU=1` 會自動中止）、`forward`／`bwd` 耗時外推不超過推估 2 倍。
3. bench 之後在 GPU 上以 `check_qchunk.py --cfg configs/stage0.json --qend 2 --tol 1e-2` 抽查 vmap 與逐查詢一致（需先有 Y_0_10.npy；2026-11-14 審查後會從該區塊切片）。bench log 另需確認 torch `avail True` 與「TRAK 投影 … 配置成功」。
4. PREREG 凍結後先 commit（由使用者決定；實作者未 commit）。

## 10. 風險與未確認事項
- **GPU 上完全沒跑過**：時間與「CIFAR 上 LDS 重現 ≥0.88」皆未驗證；`jax[cuda13]` 能否在 H200 節點偵測 GPU 未驗證（bench 第一步看 `jax.devices()`）。CPU 只驗證流程、形狀與梯度正確性（MLP 閘門：有限差分相對誤差 4.7e-10、留一重訓 Pearson 0.998）。
- 精確 Y 的 vmap 記憶體（10 查詢）與 TRAK 的 bf16 投影矩陣（42 GB，d=2048）需整張 H200；OOM 時降 `qchunk` 或 `trak_d`。EK-FAC 在寬度 80 的 ResNet-9 上耗時與記憶體未知。
- 便宜方法算的是「eval 模式 BN 的最終模型」，Y 是「訓練軌跡」導數，兩者估計對象不同（回報 45 反駁 C），低 LDS 不全是 bug。
- 階段 0 只有 k=50（B=10），κ 與增益的標準誤大；最佳便宜 7 選 1 有輕微樂觀偏誤。
- Spell 重現只是「以 G_V 取前 B 個特徵向量」的簡化版（原程式未公開）。
- 匯出保真度容差 5e-3；若不過先把 `matmul_precision` 改 "highest"。
- 額度：階段 0 只需約 1 GB；環境 attrib-jax 4 GB。
