# 三組同 cohort 的早期時鐘對照

接續 u833 單臂早期診斷 `26513`，這次把 K=6c0、K=1c0（u5000）及 u833c0 放在同一組六個訓練 pick、同一份類別 0 評估樣本及同一時間網格。這是已知原 26383 結果後的探索性對照，用來觀察共同加密網格是否改變既有正對照的可判讀性。原 26383 判定、訊號、入池紀錄及正式門檻沿用原文件。 [單臂診斷設計](../two-clocks-prefix/PROTOCOL.md) · [原判準](https://github.com/cactus8603/icml27-topic-search/blob/ed59264408851afa1e4b9689aa2a1660f079b3ee/killtests/biased-gen-cfg/configs/criteria_week2.json)

## 來源及前置條件

- 三臂均用原 pick 0–5。K=6 前三個 pick 的原 trainer 和依賴依週一來源 `df7cdb2`，後三個依第 2 週來源 `9e92804`。K=1 和 u833 用 `9e92804` 的控制組 trainer，原始外部 embedding 固定 `df08bc6`。訓練來源從原工作日誌的版本 metadata 和凍結檔案核對。前三個舊 run 依原規則允許無 completion receipt，其餘 15 個需完整收據。
- K=6 用真實類別、dropout 0.1、batch 512；K=1 用常數標籤 0、dropout 0.1、batch 512；u833 只取類別 0、dropout 1.0、batch 85。各自重播原訓練程式 36 epochs，保留 20,000 epoch 的學習率排程。三組評估樣本一致，訓練資料量與條件策略依原對照各自設定。
- 延用 `26513` 的 epoch 1、2、4、8、12、18、24、30、33、36。每組每個 pick 在 30、33、36 epoch 比對原 checkpoint SHA-256，若不同再比較權重張量 dtype、shape 和精確值。18 個 run 的 54 個 anchor 都過才算曲線；首個不吻合的 run 即停止。原 checkpoint 只讀。相符驗證的是存下的權重狀態。

## 對照方式

- 同一份固定階層資料、類別 0 各區塊前 700 筆、各 pick 作來源、測試區塊 7、評估 seed 123、原八個擴散時間。兩個 K 模型均使用 cond；u833 使用 uncond。保留正式分支配置與相同評估器。
- 每組先在原 71 個 checkpoint 重算，再與加入七個早期點的共同網格比較。第三種計算在共同網格保留各組、配對和擴散時間的原 71 點正規化分母，只改取樣網格。該分母須在原 71 點重建正規化曲線。原 u833 第一格配對集合須重現已記錄的小樣本診斷核對。
- 依原遮罩、三點平滑及最低點後至少上升 10% 判斷每組有效配對。各組同樣以 15 對為完整範圍，任何一組無效超過 5 對則本網格無法套用對照。回報三組有效分類和共同有效對數。依原式在各組有效配對取最低點 log epoch 平均，算 `L_u5000 − L_u833`、`λ=(L_c−L_u5000)/(L_u833−L_u5000)`、`|L_c−L_u833|`，並沿用重抽六個 pick 的 2,000 次配對 bootstrap、95% CI 及原門檻。結果僅顯示探索性網格上的既有門檻是否可套用、是否吻合。

## 資源及執行

一張 H200、90 分鐘上限。18 個短前綴最多約 5.66 GiB 暫存 checkpoint。回播、評估的中介檔及權重連結在同一暫存快照，作業結束清除；獨立 JSON 彙總留於 `icml/runs/two-clocks-cohort/<job>/`。

```bash
bash codex/experiments/two-clocks-cohort/run.sbatch --check-only
bash codex/experiments/two-clocks-cohort/run.sbatch FULL_COMMIT_SHA --check-only
sbatch --parsable codex/experiments/two-clocks-cohort/run.sbatch FULL_COMMIT_SHA
```
