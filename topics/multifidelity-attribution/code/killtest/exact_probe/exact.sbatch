#!/bin/bash
#SBATCH --job-name=mf-exact-probe
#SBATCH --partition=h200q
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=96G
#SBATCH --time=00:45:00
#SBATCH --array=0-1%1
#SBATCH --output=/work/ChihChungHsu/cactus8603/icml/runs/multifid/exact_probe/logs/%x-%A_%a.out
set -euo pipefail
export PYTHONDONTWRITEBYTECODE=1
export XLA_PYTHON_CLIENT_MEM_FRACTION=0.90
export TMPDIR=/work/ChihChungHsu/cactus8603/icml/tmp
python=/work/ChihChungHsu/cactus8603/icml/envs/attrib-jax/bin/python
cd "${SLURM_SUBMIT_DIR:?Run sbatch from the repository root}"
exec "$python" killtests/multifidelity-attribution/exact_probe/probe.py produce "${1:?Run ID required}" "${2:?Frozen commit required}" "$SLURM_ARRAY_TASK_ID"
