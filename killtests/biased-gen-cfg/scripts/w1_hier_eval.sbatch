#!/bin/bash
#SBATCH --job-name=bg-hier-eval
#SBATCH --partition=h200q
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=40G
#SBATCH --time=01:30:00
#SBATCH --output=/work/ChihChungHsu/cactus8603/icml27-topic-search/killtests/biased-gen-cfg/logs/%x-%j.out
# (b) oracle 偏差曲線（條件／無條件／CFG w=1,3）。需等 hier train 全部完成。
source /work/ChihChungHsu/cactus8603/icml27-topic-search/killtests/biased-gen-cfg/scripts/common.sh
$PY $S/eval_hier_oracle.py --data $HIER_DATA --runs $BG_RUNS/hier/pick0 $BG_RUNS/hier/pick1 $BG_RUNS/hier/pick2 \
    --guidance $HIER_GUIDANCE --out $EXP_DIR/results/hier_oracle_w1.npz
