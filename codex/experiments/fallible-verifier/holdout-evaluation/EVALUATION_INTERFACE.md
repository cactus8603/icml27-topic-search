# 留出評估的候選 CPU 工具

`evaluation.py` 提供資料驗證、二元 LP 暖機和已固定策略的 CPU 比較。這份工具等候開發診斷及獨立評估規格。輸入是無題文的收據，測試為合成資料。 [固定面板](../real_holdout/PREREG.md) · [沿用的風險與成本規格](../round2/PREREG.md)

## 資料介面

`ingest_split` 接受固定面板的 `gold.jsonl` 完整位元組，核對既有 SHA-256 後，只解碼呼叫端指定的 train 或 test。核對題目 ID、候選 ID、分割與每題的四種變體；需要完整 250 或 747 題。每個 verifier 傳一份 JSON 收據，以及由正式執行紀錄獨立核對的 SHA-256。收據包含 `schema: 1`、`role`、`model`、`model_revision`、`split`、`binding`、`prompt_manifest_sha256` 和 `records`。每筆 `records` 只含 `id`、`verdict`（0、1、null）及 `status`（`observed`、`abstain`、`missing`）。預期每一候選都有一筆收據；沒有有效分數明列 null。

`Binding` 需要正式凍結的 commit、量測協定 SHA-256、分數轉成 verdict 的規則 SHA-256，並加入面板既有來源指紋。CPU 入口獨立重建兩個分割的 prompt manifest，核對兩個成功的 Slurm 工作及其原始日誌所列的四份推論收據雜湊。 [GPU 收據入口](../holdout-scoring/README.md)

## 比較介面

- `fit_round2_binary` 只接收 train 結構。呼叫端需指定訓練風險目標、成本、暖機比例、探測比例、攤銷分母、平滑和種子。沿用原 LP，包含等額暖機的無強／可詢問強、同強探測配全部訓練 gold、全部訓練 gold 的無強／可詢問強。已付費但答案未知的 gold 仍列暖機成本。完整訓練 gold 說明可用資訊上限，測試部署不取得那些標籤。
- `compare_round2_binary` 用固定後的策略評估相同 test，核對訓練與測試 binding 一致且題目不重疊。另可把已凍結的其他策略轉成每個候選的六種期望動作，交給 `compare_actions`。依序為直接接受、拒絕、專家，以及付強模型費後接受、拒絕、專家。弱模型缺測或棄權時直接轉專家；強模型缺測或棄權時累計查詢及專家成本。沿用原協定中付費專家可修正判決的成本模型。
- 分別回報誤收、誤拒、未知 gold 的保守上界、配對惡化界、部署成本與暖機攤銷。兩側共用以題為單位的重抽，維持一題四候選的相依結構。必要時用 `assess_equal_risk` 接收另行凍結的風險、覆蓋率與省費門檻。只有六項風險皆支持且省費有支持，才回報有等風險省費。無法校準、風險不相配、零基線成本或比例未定義，須依報表保留限制。

本工具不決定連續似然或 verdict 的最終用法、棄權門檻、暖機政策參數與正式風險標準。待開發集完成後，另由審查者確認輸入協定與比較策略，再把已選規格固定並接上正式留出流程。

## 現象與條件訊號

`diagnostics.measure_binary_signal` 以 Round 2 的原始定義計算強模型誤收率（已知錯答中的強接受）和共錯份額（強誤收中的弱接受），也回報弱誤收及未知 gold 的保守下界。另列各角色只取 observed 的誤收比例，以及強誤收中弱角色為 observed 的共錯比例。這些 observed 分母與原分母分開。兩角色的 abstain 與 missing 各自保留。

配對 Brier 使用已有 gold 且 weak 已觀測的同一候選。兩臂都用 weak、資料集和 solver variant 作條件，僅加強端的模型允許用 strong verdict 修正。擬合只取 train。測試缺強判決或 train 中相應條件格缺任一種強判決時，使用同一弱端預測。以題群重抽共用分母，回報兩臂 Brier、相對降幅、候選數及缺強覆蓋。`assess_binary_signal` 可在另行固定標準後帶入原現象／訊號門檻。該判讀使用原分母和未知下界，observed-only 比例供描述。

## CPU 作業入口

`run_evaluation.py` 沿用已鎖版的 Round 2 數值與原比較順序，產出候選方法的三項機械判讀。等暖機主要比較為可查強模型的策略對只買 gold，包含暖機攤銷。完整訓練資訊的對照也同時計算，作為指定策略與訓練資訊範圍的參考。預檢只測合成資料、收據格式及 criteria 雜湊。實跑須等共同凍結的 train/test GPU 作業成功，並逐份核對日誌綁定後才載入封存面板。只寫聚合 `icml/runs/fallible-verifier/gsm8k-policy-evaluation/<job>/summary.json`。

```bash
/work/ChihChungHsu/cactus8603/icml/envs/t2irare/bin/python -IB \
  codex/experiments/fallible-verifier/holdout-evaluation/test_evaluation.py
bash codex/experiments/fallible-verifier/holdout-evaluation/run.sbatch FULL_FREEZE_COMMIT --check-only
```
