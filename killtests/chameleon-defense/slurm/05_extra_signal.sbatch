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
base=${CHAM_BASE_RESULTS:?set CHAM_BASE_RESULTS to the registered analysis directory}
cache=${CHAM_CACHE:-$RUN/cache}
extra="$EXP/results/extra_${SLURM_JOB_ID:?run through Slurm}"
extra_cache="$RUN/cache_extra_${SLURM_JOB_ID}"
exec 9>"$base/extra_attempt.lock"
flock -n 9
$PY -I - "$base" "$extra" "$SLURM_JOB_ID" "${CHAM_EXTRA_RETRY_OF:-}" <<'PYEOF'
import json, sys
from pathlib import Path
base = Path(sys.argv[1])
r = json.loads((base / "RESULT.json").read_text())
st = r.get("signal", {}).get("state"); used = r.get("signal", {}).get("used_extra")
if st != "灰區" or used:
    sys.exit(f"加碼只在訊號為灰區且尚未加碼時執行；目前訊號={st}，used_extra={used}。拒絕執行。")
record = base / "extra_attempt.json"
previous = json.loads(record.read_text()) if record.exists() else None
retry_of = sys.argv[4]
if previous:
    previous_output = Path(previous["output"])
    if (previous["state"] == "completed" or (previous_output / "RESULT.json").exists()
            or retry_of != previous["job_id"]):
        sys.exit("已有加碼記錄。只有沒有發布判讀的技術重試可指定前一次 job ID。")
elif retry_of:
    sys.exit("找不到可重試的加碼記錄。")
pending = record.with_suffix(".pending")
pending.write_text(json.dumps({"output": sys.argv[2], "job_id": sys.argv[3],
                               "retry_of": retry_of or None, "state": "running"}, indent=2))
pending.replace(record)
PYEOF
update_attempt() {
  local result=$?
  "$PY" -I - "$base/extra_attempt.json" "$result" <<'PYEOF'
import json, sys
from pathlib import Path
record = Path(sys.argv[1])
value = json.loads(record.read_text())
value["state"] = "completed" if sys.argv[2] == "0" else "failed"
out = Path(value["output"])
if out.is_dir():
    (out / "attempt.json").write_text(json.dumps(value, indent=2))
pending = record.with_suffix(".pending")
pending.write_text(json.dumps(value, indent=2))
pending.replace(record)
PYEOF
}
trap update_attempt EXIT
mkdir "$extra"
$PY -I scripts/extract_acts.py --sets harm_test_extra --batch-size "${BS:-16}" --out "$extra_cache"
$PY -I scripts/analyze.py monitors --cache "$cache" --cache-extra "$extra_cache" --merge-extra --results "$extra"
$PY -I scripts/aggregate.py --results "$base" --extra-results "$extra" --out "$extra"
