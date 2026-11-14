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
N=${W2_N:-6}
runs(){ d=$1; n=$2; for ((i=0;i<n;i++)); do printf "%s/%s/pick%d " "$BG_RUNS" "$d" "$i"; done; }
R=$EXP_DIR/results
$PY $S/eval_hier_oracle.py --data $HIER_DATA --runs $(runs hier 6)     --guidance $HIER_GUIDANCE --out $R/hier_oracle_w2_k6.npz
$PY $S/eval_hier_oracle.py --data $HIER_DATA --runs $(runs hier_k1 $N) --guidance $HIER_GUIDANCE --out $R/hier_oracle_w2_k1.npz
$PY $S/eval_hier_oracle.py --data $HIER_DATA --runs $(runs hier $N)    --guidance 3 --class-filter 0 --n-eval 700 --out $R/hier_oracle_w2_k6c0.npz
$PY $S/eval_hier_oracle.py --data $HIER_DATA --runs $(runs hier_u833 $N) --class-filter 0 --n-eval 700 --out $R/hier_oracle_w2_u833c0.npz
$PY $S/eval_hier_oracle.py --data $HIER_DATA --runs $(runs hier_k1 $N) --guidance 3 --class-filter 0 --n-eval 700 --out $R/hier_oracle_w2_k1c0.npz
$PY $S/aggregate_week2.py --k6 $R/hier_oracle_w2_k6.npz --k1 $R/hier_oracle_w2_k1.npz --k6c0 $R/hier_oracle_w2_k6c0.npz \
    --u833c0 $R/hier_oracle_w2_u833c0.npz --k1c0 $R/hier_oracle_w2_k1c0.npz ${W2_H:+--h $W2_H} --out $R/week2_verdict.json
