# GCD 四家族正式結果的 CPU 銜接

`run.sbatch` 只在指定來源作業都成功後讀取收據。它使用資料作業 `26821` 的凍結清冊、PHM 分數 `26822`、技術驗收 `26823`，以及提交時指定的兩個 51 格陣列和各自的 CPU 驗收。核對 Slurm 作業名稱、成功狀態、登記的凍結版本、固定輸出路徑及各日誌末行的檔案 SHA-256。訓練結果須逐格指向同一份資料、步 0 預測及 LoRA 初始值。比較程式直接沿用 [共同支撐評估器](README.md)。

```bash
stage=codex/experiments/lambda-gcd-family-design/evaluation
bash "$stage/run.sbatch" "$freeze" --check-only
sbatch --parsable --dependency="afterok:26821:26822:26823:$step_array:$step_join:$outcome_array:$outcome_judge" \
    --kill-on-invalid-dep=yes "$stage/run.sbatch" "$freeze" \
    "$step_array" "$step_join" "$outcome_array" "$outcome_judge"
```

步 0 與行為分數在每臂以三個相同種子等權。全防禦臂比較和所有種子均通過 ID 匹配的敏感度分析各自報告。PHM 的單一前綴原公式、銀行加權及只看前綴的對照各限適用家族。`phm_gcd_adaptation_all` 另列使用各家族適配分數的整體對照，範圍依實際納入的共同臂與家族。所有 PHM 分數均使用 GCD 校準語料與凍結 Qwen2.5。全防禦臂與 ID 子集都沿用未知漏出界，並報同分母實際所選設定的漏出界。 [凍結協議](../PROTOCOL.md) · [PHM 校準入口](../phm/README.md)

摘要寫到 `icml/runs/lambda-gcd-family-comparison/<job>/summary.json`，stdout 只提供檔案路徑與 SHA-256。預檢使用合成收據與模擬 Slurm 狀態。測試完整銜接、檔案雜湊不符、LoRA 初始值不符和陣列尚未終止時的拒絕路徑。正式作業使用 CPU，保留數字與來源摘要。
