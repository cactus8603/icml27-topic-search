#!/bin/bash
#SBATCH --job-name=cham-inv
#SBATCH --partition=h200q
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=96G
#SBATCH --time=01:00:00
#SBATCH --output=logs/%x-%j.out
# 唯一的第二探針 INV（PREREG 第 8 節 C3；2026-11-14 定案）。只在主分析 ROB=FAIL 時才可送出；
# 抽取 harm_test_extra（421 正／625 負，上限 0.5 GPU 小時）→ 與 harm_test 合併判 INV vs ENS（97.5% CI）→ 重跑 aggregate。
# 本檔寫好但未送出；送出須使用者同意。灰區走 PREREG 第 6 節加碼，不走本檔（互斥）。
set -euo pipefail
source slurm/env.sh
$PY -I - <<'PYEOF'
import json, sys
r = json.load(open("results/RESULT.json"))
st = r.get("gate4", {}).get("monitor", {}).get("state")
if st != "FAIL":
    sys.exit(f"INV 探針只在 ROB=FAIL 時觸發；目前主指標狀態={st}。拒絕執行。")
PYEOF
$PY -I scripts/extract_acts.py --sets harm_test_extra --batch-size ${BS:-16} --out $RUN/cache_extra
$PY -I scripts/analyze.py inv --cache $RUN/cache --cache-extra $RUN/cache_extra --results $EXP/results
$PY -I scripts/aggregate.py --results $EXP/results
