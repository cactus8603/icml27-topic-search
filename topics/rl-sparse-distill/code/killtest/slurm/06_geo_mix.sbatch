#!/bin/bash -l
#SBATCH --partition=h200q
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=16
#SBATCH --mem=170G
#SBATCH --output=/work/ChihChungHsu/cactus8603/icml27-topic-search/killtests/rl-sparse-distill/slurm/logs/%x-%j.out
#SBATCH --job-name=rsd-geomix
#SBATCH --time=24:00:00
# 【只寫不送】第二探針 R3（PREREG「2026-11-14 代理落差稽核後」）：λ=0.5 幾何混合取樣。
# 觸發條件：合格配對的 (i) 成立而 (ii) FAIL（aggregate.py 輸出 probe2.triggered=true 與 pair）；只做一個配對，A 符合就做 A，否則做 B。
# 用法：sbatch slurm/06_geo_mix.sbatch PAIR SHARD NSHARD   （與 02_rl_replace 相同：test 1,053 題 × n=128、同 seed 與分片規則、temperature 0.6、top-p 0.95、3000 token）
# 預算：探針上限 25 GPU 小時；全題上限 120 不變——送出前先以實測吞吐推估「已用＋本探針」，預估超出 120 就先回報，不自行刪減。
# 結果只有通過或未通過，不加碼；輸出 results/<P>/geo_mix（graded.json）。
export PAIR=$1; SH=$2; NS=$3
set -e
source /work/ChihChungHsu/cactus8603/icml27-topic-search/killtests/rl-sparse-distill/slurm/env.sh
$PY -BI scripts/artifacts.py "$PAIR" geo_mix
export RSD_RUN_ID=${RSD_RUN_ID:-${SLURM_ARRAY_JOB_ID:-$SLURM_JOB_ID}}
OUT=results/$PAIR/geo_mix
$PY -I scripts/gen_replace.py --cond geo_mix --lam 0.5 --base-model $BASE --rl-model $RL --out $OUT --split test --n 128 --rows-per-batch 256 --shard $SH/$NS || exit 1
$PY -I scripts/grade.py "$OUT" --pair "$PAIR" --condition geo_mix --n-shards "$NS" --workers 16
