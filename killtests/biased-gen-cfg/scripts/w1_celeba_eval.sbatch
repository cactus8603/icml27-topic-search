#!/bin/bash
#SBATCH --job-name=bg-celeba-eval
#SBATCH --partition=h200q
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=40G
#SBATCH --time=03:00:00
#SBATCH --output=/work/ChihChungHsu/cactus8603/icml27-topic-search/killtests/biased-gen-cfg/logs/%x-%j.out
# (a) 評估：6 模型 × 全部 checkpoint 的測試損失與成對 ε cosine 相似度。需等 train 全部完成（sbatch --dependency=afterok:<id>）。
source /work/ChihChungHsu/cactus8603/icml27-topic-search/killtests/biased-gen-cfg/scripts/common.sh
$PY $S/eval_celeba.py --save-root $BG_RUNS/celeba --data-root $CELEBA_DATA_ROOT --n-models $CELEBA_NMODELS \
    -n $CELEBA_N -W $CELEBA_WIDTH -B $CELEBA_BATCH -LR $CELEBA_LR --out $EXP_DIR/results/celeba_eval.npz
