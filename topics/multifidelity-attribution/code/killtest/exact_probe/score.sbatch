#!/bin/bash
#SBATCH --job-name=mf-score-probe
#SBATCH --partition=h200q
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=2
#SBATCH --mem=12G
#SBATCH --time=00:15:00
#SBATCH --output=/work/ChihChungHsu/cactus8603/icml/runs/multifid/exact_probe/logs/%x-%j.out
set -euo pipefail
export PYTHONDONTWRITEBYTECODE=1
export CUDA_VISIBLE_DEVICES=''
export OPENBLAS_NUM_THREADS=2 OMP_NUM_THREADS=2 MKL_NUM_THREADS=2
python=/work/ChihChungHsu/cactus8603/icml/envs/attrib-jax/bin/python
cd "${SLURM_SUBMIT_DIR:?Run sbatch from the repository root}"
exec "$python" killtests/multifidelity-attribution/exact_probe/probe.py score "${1:?Run ID required}" "${2:?Frozen commit required}"
