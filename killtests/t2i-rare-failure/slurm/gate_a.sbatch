#!/bin/bash
#SBATCH --job-name=t2ir-gatea
#SBATCH --partition=defq
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=32G
#SBATCH --time=02:00:00
#SBATCH --output=logs/%x-%j.out
# 閘門 (a)（純 CPU，不要 GPU）：d=16384 高斯半空間，p=1e-3..1e-6，各 20 次重複。登入節點負載高時改在此跑。
# 登入節點實測：subset_sim 4 個 p 約 20 分鐘、ce_is（必然失敗，僅留作記錄）約 30 分鐘以上。
set -euo pipefail
source slurm/env.sh
export OMP_NUM_THREADS=8 OPENBLAS_NUM_THREADS=8
$PY -I scripts/gate_a_gaussian.py
