#!/bin/bash
#SBATCH --job-name=mf-cheap
#SBATCH --partition=h200q
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=128G
#SBATCH --time=06:00:00

#SBATCH --output=/work/ChihChungHsu/cactus8603/icml27-topic-search/killtests/multifidelity-attribution/logs/%x-%A_%a.out
source /work/ChihChungHsu/cactus8603/icml27-topic-search/killtests/multifidelity-attribution/scripts/common.sh
# 便宜方法（PyTorch）：grad_dot、tracin、trak1/trak5、ekfac×3。需 s0_train 完成（afterok）。
$PYT $S/cheap_torch.py --cfg $CFG
