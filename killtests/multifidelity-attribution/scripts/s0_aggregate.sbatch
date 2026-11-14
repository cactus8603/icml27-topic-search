#!/bin/bash
#SBATCH --job-name=mf-stage0-aggregate
#SBATCH --partition=defq
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=16G
#SBATCH --time=00:30:00
#SBATCH --output=/work/ChihChungHsu/cactus8603/icml27-topic-search/killtests/multifidelity-attribution/logs/%x-%j.out
set -euo pipefail
source /work/ChihChungHsu/cactus8603/icml27-topic-search/killtests/multifidelity-attribution/scripts/common.sh
unset REQUIRE_GPU
export PYTHONDONTWRITEBYTECODE=1 OMP_NUM_THREADS=4 OPENBLAS_NUM_THREADS=4
"$PYT" "$S/aggregate.py" --cfg "$CFG" --stage 0
