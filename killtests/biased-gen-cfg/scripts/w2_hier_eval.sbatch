#!/bin/bash
#SBATCH --job-name=bg-w2-eval
#SBATCH --partition=h200q
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=40G
#SBATCH --time=04:00:00
#SBATCH --output=/work/ChihChungHsu/cactus8603/icml27-topic-search/killtests/biased-gen-cfg/logs/%x-%j.out
# 第 2 週統一評估：所有 run 訓練完後才送（同一個判定內的所有 run 必須一次評完）。W2_N = K=1 / u833 的 run 數（6 或 4，由 h 決定）。
# 用法：sbatch --export=ALL,W2_N=6,W2_H=1.8 w2_hier_eval.sbatch
source /work/ChihChungHsu/cactus8603/icml27-topic-search/killtests/biased-gen-cfg/scripts/common.sh
: "${W2_H:?必須指定第一週實測平均 GPU 小時}"
: "${W2_N:?必須指定依 W2_H 決定的控制組 run 數}"
"$PY" - "$W2_H" "$W2_N" "$EXP_DIR/configs/criteria_week2.json" <<'PY'
import json, math, sys
from pathlib import Path
h, n = float(sys.argv[1]), int(sys.argv[2])
rule = json.loads(Path(sys.argv[3]).read_text())["negative_control"]
key = "n_runs_if_h_le_threshold" if h <= rule["h_gpu_hours_threshold"] else "n_runs_if_h_gt_threshold"
if not math.isfinite(h) or h <= 0 or n != rule[key]:
    raise SystemExit("W2_H/W2_N 與凍結配置不符")
PY

W2_ROOT=${W2_RUNS_ROOT:-$BG_RUNS}
k6=() k1=() u833=()
for ((i=0; i<6; i++)); do
    if ((i < 3)); then root=$BG_RUNS; else root=$W2_ROOT; fi
    k6+=("$root/hier/pick$i")
    if ((i < W2_N)); then
        k1+=("$W2_ROOT/hier_k1/pick$i")
        u833+=("$W2_ROOT/hier_u833/pick$i")
    fi
done

R=${W2_RESULTS_DIR:-"$EXP_DIR/results"}
"$PY" "$S/eval_hier_oracle.py" --week2-role k6 --data "$HIER_DATA" --runs "${k6[@]}" --guidance "$HIER_GUIDANCE" --gamma --out "$R/hier_oracle_w2_k6.npz"
"$PY" "$S/eval_hier_oracle.py" --week2-role k1 --data "$HIER_DATA" --runs "${k1[@]}" --guidance "$HIER_GUIDANCE" --out "$R/hier_oracle_w2_k1.npz"
"$PY" "$S/eval_hier_oracle.py" --week2-role k6c0 --data "$HIER_DATA" --runs "${k6[@]:0:W2_N}" --guidance 3 --class-filter 0 --n-eval 700 --out "$R/hier_oracle_w2_k6c0.npz"
"$PY" "$S/eval_hier_oracle.py" --week2-role u833c0 --data "$HIER_DATA" --runs "${u833[@]}" --guidance 0 --class-filter 0 --n-eval 700 --out "$R/hier_oracle_w2_u833c0.npz"
"$PY" "$S/eval_hier_oracle.py" --week2-role k1c0 --data "$HIER_DATA" --runs "${k1[@]}" --guidance 3 --class-filter 0 --n-eval 700 --out "$R/hier_oracle_w2_k1c0.npz"
"$PY" "$S/aggregate_week2.py" --k6 "$R/hier_oracle_w2_k6.npz" --k1 "$R/hier_oracle_w2_k1.npz" --k6c0 "$R/hier_oracle_w2_k6c0.npz" \
    --u833c0 "$R/hier_oracle_w2_u833c0.npz" --k1c0 "$R/hier_oracle_w2_k1c0.npz" --h "$W2_H" --out "$R/week2_verdict.json"
