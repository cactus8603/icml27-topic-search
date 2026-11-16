# CORE 2B／8B 空間量測

[codex] 固定既有 8,191 張圖、136 query，在同平台比較兩種召回與兩種重排的四個組合。方法、標籤與判讀範圍見 [PREREG](PREREG.md)。這是已曝光資料的開發診斷。

公開模型由 `acquire.py` 按固定 revision 取得並驗證。`build_plan.py` 結合舊平台與新資產 SHA 建立 `plan.json`；程式、plan 與協定提交後才執行研究推論。

```bash
python3 -B codex/retrieval_scaling/acquire.py
python3 -B codex/retrieval_scaling/build_plan.py
```

用原排隊器等待足夠空閒的顯卡。下列 `<freeze-commit>` 為包含本輪程式及 plan 的完整提交；輸出目錄不可使用舊方法或封存資料的路徑。

```bash
python3 -B codex/retrieval3090/resources.py queue \
  --min-free-mib 20480 --max-utilization 20 \
  --queue dl/retrieval3090/queues/core-scaling-20261116.json -- \
  bash codex/retrieval_scaling/run.sh <freeze-commit> \
  "$PWD/dl/retrieval3090/runs/core-scaling-20261116"
```

runner 依序做 CPU 來源驗收、純色圖片技術 smoke、8B 索引與四格分數，產出 canonical NPY、每批進度與來源收據，不產生研究指標。完成後才執行 CPU 分析：

```bash
dl/retrieval3090/envs/retrieval3090-torch214/bin/python -B \
  codex/retrieval_scaling/analyze.py \
  --run dl/retrieval3090/runs/core-scaling-20261116 \
  --plan codex/retrieval_scaling/plan.json \
  --freeze-commit <freeze-commit> \
  --out dl/retrieval3090/runs/core-scaling-20261116/results/analysis
```

分析只接受 COMPLETE 收據與完整 SHA；結果首次讀取須登記，並由分析實作者之外的審查者核對。研究數字放在 `results/` 或 `reports/`，本頁只記使用方式。
