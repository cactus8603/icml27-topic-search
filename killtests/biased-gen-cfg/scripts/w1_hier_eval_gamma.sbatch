#!/bin/bash
#SBATCH --job-name=bg-hier-eval-gamma
#SBATCH --partition=h200q
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=40G
#SBATCH --time=01:30:00
#SBATCH --output=/work/ChihChungHsu/cactus8603/icml27-topic-search/killtests/biased-gen-cfg/logs/%x-%j.out
# 第 1 週階層評估重跑，額外輸出 pair_cos_gamma（條文 7 的 γ 曲線）。輸出另存 hier_oracle_w1_gamma.npz，舊檔 hier_oracle_w1.npz 保留不動。
# 寫好未送出；送出須使用者同意。
source /work/ChihChungHsu/cactus8603/icml27-topic-search/killtests/biased-gen-cfg/scripts/common.sh
$PY $S/eval_hier_oracle.py --data $HIER_DATA --runs $BG_RUNS/hier/pick0 $BG_RUNS/hier/pick1 $BG_RUNS/hier/pick2 \
    --guidance $HIER_GUIDANCE --gamma --out $EXP_DIR/results/hier_oracle_w1_gamma.npz
