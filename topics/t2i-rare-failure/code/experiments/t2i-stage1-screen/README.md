# 候選 Stage 1 完整性與 CPU 選格

此入口供同一啟用提交的六個模型完成後執行。只讀候選 Stage 1 的分數、側錄、操作收據及 Slurm 紀錄，核對 84 格、336 個分塊、168,000 筆種子及各分塊的模型工作身分。原 Stage 2a 內部函式在驗證完成後執行，沿用原凍結事件、合格條件、排序及覆蓋規則。 [原選格程式](../../killtest/scripts/stage_selection.py)

`--check-only` 比對凍結程式與配置，供啟用前使用。CPU 合成測試建立完整替身面板、不使用候選工作產物。

```bash
shared=/work/ChihChungHsu/cactus8603/icml
stage=codex/experiments/t2i-stage1-screen
export TMPDIR="$shared/tmp"
"$shared/envs/t2irare/bin/python" -IB "$stage/screen.py" --check-only
"$shared/envs/t2irare/bin/python" -IB "$stage/test_screen.py"
```

六個模型完成且分析程式凍結後，在 repo 執行下列 CPU 指令。候選 array 的母工作編號及啟用提交須依協作規則登記在 `killtests/SUBMITTED.log`，包含因失敗而重送的 array。

```bash
"$shared/envs/t2irare/bin/python" -IB "$stage/screen.py" \
    --analysis-freeze FULL_ANALYSIS_COMMIT --activation-freeze FULL_ACTIVATION_COMMIT
```

成本從帳本綁定的陣列任務計算，將每段實際配置的 GPU 數與 Slurm `ElapsedRaw` 相乘，再加總成功、失敗及重試的已歸帳配置。另報仍保留之分塊記錄時間和分數/側錄位元組，便於區分有效執行和額外配置時間。含未完成或無法歸屬的工作時，入口停止。

輸出只會寫到該啟用提交的 `screening/candidate-2a-<analysis-freeze>.json`。文件綁定原始分數/側錄雜湊及帳本工作身分，可供獨立檢查候選順序。正式 Stage 2、Gate B 和盲標沿用原入口與定案程序。
