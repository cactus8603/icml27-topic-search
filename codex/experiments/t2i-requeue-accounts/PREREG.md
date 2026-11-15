# 文生圖 Slurm 重排的配置成本修復

[codex] 2026-11-16（台北）。Slurm 的 `--duplicates` 會把同一索引先前已結束的 GPU 配置列為 `REQUEUED`。這個狀態原先未列入封存成本程式所接受的終止紀錄。保留原作業、資料、分段、選格及算費，由選格、成本排程及監看共同接受這一種已結束狀態。每個原分段仍各自需要 `COMPLETED`／`0:0`，以及對應的原完成收據。前後配置的 GPU 秒均計入原成本上限。[原正式取樣](../t2i-formal-replay/PREREG.md)

程式只從指定 Git 版本載入。原 4351 的入口、樣本和既存配置紀錄維持原位。CPU 入口在暫存封存副本中呼叫原程式並增列歷史狀態；監看入口對原封存成本程式設定同一終止狀態。原 CPU 選格 `26802` 若替換，需確認其已終止，並在共享提交紀錄用同一試驗版本登記第二個 `formal-plan=2a`。新 CPU 選格使用同一個 `afterok:26765`，JobName `t2ir-formal-plan-2a`，輸出 `icml/tmp/t2irare/t2ir-plan-2a-%j.out`。原監看退出後，用相同排他鎖啟用綁定新作業的監看，避免兩份監看同時送出。

```bash
repair=codex/experiments/t2i-requeue-accounts/run.sh
bash "$repair" "$repair_commit" --check-only
bash "$repair" "$repair_commit" plan 2a
bash "$repair" "$repair_commit" watch "$replacement_planner" \
    --schedule-freeze 8e49e19c8fcc60e9a71f85c1811f8b88315868cb --watch
```

預檢以兩個索引各重排一次的合成配置驗證成本累加、真成功的要求及仍在執行的拒絕條件。未開啟正式計分。後續若要評人工標註，用同一入口跑 `review-stats <label-fix-commit>` 或 `aggregate <label-fix-commit>`，即可在不改既有取樣或評分條件的情況下同時套用已封存的合法 `0` 標籤修復。
