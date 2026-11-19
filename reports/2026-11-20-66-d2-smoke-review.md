# D2 smoke 審查（修正 bea1c19 + smoke 全流程）— 2026-11-20

審查者：Claude（非實作者），只做 CPU，未送 GPU job，未讀任何 truth／結果檔（只看 28360 的 smoke log）。
對象：`icml27-multifidelity-attribution` @ 3aa9dcd，`code/experiments/multifidelity-d2/{smoke.sbatch,d2_panel.py,d2_gpu.py}`。

## 結論：PASS（可送 smoke）

## 1. 修正 bea1c19（d2_gpu.py L217–221）
- 正確：`phase_exact` 改用 `ref_losses[:len(final_losses)]`，並先 `require(len(ref) >= len(final))`。smoke 在 L561 把 `env.xq` 截成前 2 題，`phase_fwd` 存的是 10 題，因此取前 2 列，順序也對得上。
- 正式面板行為不變：`pred-exact`（d2_panel L147，`phase_exact(env, …, 0, 50, own_js=…)`）和 `pred-exact-b`（L149）都用完整的 100 題 env，`final_losses` 與參考同長，slice 不起作用。
- 前提：`query_losses` 對每一題分別計算，與 batch 內的其他題無關。乾跑時，2 題的 rmse 為 0.0，證實這點。

## 2. smoke 全流程逐行追蹤
- smoke-jax：28360 已在 GPU 上跑完 fwd、truth、half，到 exact 才失敗。
  - exact（2 題，1 個 block）：L237 的 own retrain 對截斷後的 env 回傳 (1,2)。
  - L243–245：只有 1 個 mask 時 LDS 無定義，記為 0，不會崩潰。這只是描述值，不是 gate。
  - `phase_trak_models` 只用 `ref_x`，不受截斷影響，產出 `model_s39002.npz`。
  - receipt 寫在 `run/`，不是 `run/smoke/`，與 `smoke_report` 讀的位置一致。
- smoke-torch 讀的檔案：`work/model.npz` 與 `schedule.json`（fwd 寫出）、`model_s39002.npz`（trak-models 寫出），都在同一個 bundle 的 `work` 下。
  - TRAK 用完 pos1 的 `model.npz` 後不刪，ASTRA 接著用它。pos2 的 `model_s39002.npz` 用完即刪。
  - `work/kron` 由 EK-FAC 建立後 rmtree，ASTRA 再重建。
  - fidelity 參考的 64 題與 JAX export 所用的 `ref` 來自同一個 RandomState，兩邊一致。
- smoke_report 讀的鍵都存在：`truth.retrain_mean_s`、`exact.exact_block_s`、`ekfac.fit_s`、`trak`、`wall_s`。乾跑以合成的 torch receipt 驗證過。
- 其他：
  - run 目錄依 job id 分開，`publish` 不會因檔案已存在而撞名。
  - bundle 由 EXIT trap 刪除，模型權重不會留下。
  - quota 在 archive 前後各檢查一次（≥4 GiB）。
- 無法在 CPU 驗證：torch 三個 phase 需要 CUDA 與 Kronfluence。靜態追蹤未發現形狀或路徑錯誤。剩餘風險：smoke 的 miniature sweep 若選到不穩定的 alpha，會依設計 HOLD（L472）。

## 3. CPU 乾跑
腳本 `/work/ChihChungHsu/cactus8603/icml/tmp/claude-d2review/dry.py`，用 tiny_problem 建環境。

- smoke 路徑：先以 10 題做 fwd、truth、half，再截成 2 題。
  - 舊的比較方式重現 28360 的錯誤：`(2,) vs (10,)`。
  - 修正後的 `phase_exact` 通過，rmse 0.0；`exact_Y` 為 (1000,2)，`own` 為 (1,2)。
  - trak-models 與 smoke_report 也都通過。
- 面板路徑：24 題、`phase_exact(0,20)`，新舊公式都得 rmse 0.0，own LDS 0.9993。

結果：DRY RUN OK。
