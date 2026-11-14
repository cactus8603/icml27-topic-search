#!/bin/bash
#SBATCH --job-name=cham-extra
#SBATCH --partition=h200q
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=96G
#SBATCH --time=01:00:00
#SBATCH --output=logs/%x-%j.out
# 訊號灰區的唯一加碼（PREREG 第 6 節；2026-11-14 使用者原則變更後、結果前定案）：抽取 harm_test_extra（421 正／625 負，約 0.25 GPU 小時），
# 與 harm_test 合併後只重算 monitors（不重跑 PC、偵測器），寫到 results/extra/，再以 --extra-results 重判一次 S-b。
# 本檔寫好但未送出；送出須使用者同意；主分析的 RESULT.json 訊號不是「灰區」就拒絕執行。
set -euo pipefail
source slurm/env.sh
$PY -I - <<'PYEOF'
import json, sys
r = json.load(open("results/RESULT.json"))
st = r.get("signal", {}).get("state"); used = r.get("signal", {}).get("used_extra")
if st != "灰區" or used:
    sys.exit(f"加碼只在訊號為灰區且尚未加碼時執行；目前訊號={st}，used_extra={used}。拒絕執行。")
PYEOF
$PY -I scripts/extract_acts.py --sets harm_test_extra --batch-size ${BS:-16} --out $RUN/cache_extra
mkdir -p results/extra
$PY -I scripts/analyze.py monitors --cache $RUN/cache --cache-extra $RUN/cache_extra --merge-extra --results $EXP/results/extra
$PY -I scripts/aggregate.py --results $EXP/results --extra-results $EXP/results/extra
