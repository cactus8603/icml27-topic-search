# 61｜Claude 交接給 Codex（2026-11-14 約 00:00）

使用者指示：Claude 實作完手上的部分就暫停，接下來由 Codex 接手。本份記錄交接時的狀態。Claude 已停止所有子代理，不再送出、hold、release 或取消任何 job。

## 0. Codex 接手的授權範圍

- 使用者把 Claude 的工作交給 Codex。下面列為 [claude] 的 job（含 hold 中的 26286），Codex 可依 COORDINATION.md 照常操作，並在 SUBMITTED.log 記錄（註明「接手 Claude」）。
- LOCKS.md 中 Claude 的 lock 已全部移除。
- 判準依據：報告 60（現象、訊號、空間；方法對 baseline 只作描述）與通用條文 G。結果出來前必須先把新判準實作並 commit。

## 1. 各題狀態

| 題目 | 新判準（報告 60） | job 狀態 | 下一步 |
|---|---|---|---|
| rl-sparse-distill | 已實作並 commit（bbe8dc2） | gate 0 pair A：26292 base、26293 rl 執行中；26290、26291 greedy 已送 | 結果出來後跑 aggregate。pair B 的 template 尚未確認；只有 pair A 未過 gate 0 時才需要決定 |
| t2i-rare-failure | 已實作並 commit（ddc1b8d） | bench 26288 完成；gate_b 26289 執行中 | 26289 完成後跑 aggregate |
| chameleon-defense | 已實作並 commit（6185167）。selftest `--verdict-only` 26 項全過，`--pc-only` 全吻合 | extract 26285 完成；analyze 26286 **仍 hold**（解除條件已滿足） | `scontrol release 26286` 並記錄 log、刪 HOLDS.md 那一列。`slurm/05_extra_signal.sbatch` 只在訊號為灰區時才送，送前須使用者同意 |
| multifidelity-attribution | **未實作**（子代理剛開始就停止，沒有改動檔案） | train 26324 完成；cheap 26325、retrain 26326（array 1 執行中）、exact 26327 排隊；Codex 的 26353 排隊 | 依報告 60 第 4 節實作，**在 stage 0 aggregate 執行前** commit |
| compositional-retrieval | **未實作**（補量測的檔案都還沒建立） | stage1 26274、probe2 26314 完成 | 依報告 60 第 7 節實作補量測：B_fix = max(CORE-EMBED-2B, SigLIP2, CS-CLIP，若找得到再加 CLIC)；R = CORE-RERANKER-2B（備案：VLM yes-logprob）。B_fix ≥ 85 淘汰；< 75 進入候選；75～85 灰區。檔案預計為 PREREG_supp.md、scripts/supp_baselines.py、supp_reranker.py、aggregate_supp.py、slurm/supp.sbatch。先凍結 PREREG_supp 再送 |
| biased-gen-cfg | 未實作（Codex lock 中） | 訓練與評估 26294、26295 完成；week 2 尚未送 | 依報告 60 第 1 節：Γ 一律計算、G3 空間、97.5% CI；G3 的符號要對齊「正值表示訓練優勢」（見 HOLDS.md 的 Codex 請求） |
| inoculation-lambda | 未實作（Codex lock 中） | 26333→26334→26335→26336 鏈排隊；Codex 診斷 26337 | 依報告 60 第 2 節：criteria.yaml／phaseb_extras 的訊號門檻、範圍、ρ_exist |

## 2. 工作區注意事項

- `killtests/compositional-retrieval/README.md` 有未 commit 的修改（26314 與判讀範圍說明）。這份修改不是 Claude 這次交接前改的，Claude 沒有動它。
- Claude 沒有讀過任何新結果，包括 biased-gen-cfg week 1、inoc 診斷、multifid stage 0。
- 磁碟：群組配額 1536G，約 1379G 已用。暫存檔放 `/work/ChihChungHsu/cactus8603/icml/tmp`，不要寫 /tmp。

## 3. 判讀流程（沿用）

結果 → 機械式 aggregate → 獨立檢查 → 使用者決定。每個判定要寫出三個層級：實作、機制、方向。淘汰一個題目需要以下其中一項：不依賴實作的上限或 oracle 失敗；或兩個不同原理的實作都失敗；或依報告 60 G4，現象、訊號、空間三項在範圍內 FAIL。
