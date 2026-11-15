# 強驗證器重播

[預先登記](PREREG.md) 定義公開資料、切分、費用與判讀。程式使用共享 Python 3.12 環境的 NumPy 1.26.4、SciPy 1.17.1 及 PyArrow 17.0.0。

從實驗目錄執行。以下第一條只用合成資料。第二條下載並校驗公開輸入，不產生分數。第三條是凍結後的 CPU 工作，需先依 [協作規則](../../../../../COORDINATION.md) 檢查佇列及做好登記。

```bash
work=/work/ChihChungHsu/cactus8603
python="$work/icml/envs/t2irare/bin/python"
export PYTHONPATH="$work/icml/envs/t2irare_extra_cscomp"
export TMPDIR="$work/icml/tmp" PYTHONDONTWRITEBYTECODE=1
"$python" replay.py --synthetic
"$python" replay.py --fetch --data-dir "$TMPDIR/codex-fallible-verifier-08f42e1"
sbatch run.sbatch "$TMPDIR/codex-fallible-verifier-08f42e1"
```

申請 1 個 CPU、4 GiB、20 分鐘。不需 GPU、模型推論或 API 金鑰。工作讀取四個相加約 28 MiB 的資料檔。可用 checksum 相同的現有共享目錄代替下載。Slurm 工作只讀已準備的資料，並把 log 與結果寫入 `results/`。結果檔包含程式 commit、設定 SHA-256、輸入 revision、每一組政策成本與風險、題目群集區間及機械檢查。現有結果檔不覆寫。
