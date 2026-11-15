#!/usr/bin/env bash
#SBATCH --job-name=fv-replay
#SBATCH --partition=defq
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=1
#SBATCH --mem=4G
#SBATCH --time=00:20:00
#SBATCH --output=/dev/null
#SBATCH --error=/dev/null
set -euo pipefail

work=/work/ChihChungHsu/cactus8603
cd "${SLURM_SUBMIT_DIR:?Submit from the experiment directory}"
test -f replay.py
mkdir -p results/logs
export TZ=Asia/Taipei PYTHONDONTWRITEBYTECODE=1 OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1
export TMPDIR="$work/icml/tmp"
export PYTHONPATH="$work/icml/envs/t2irare_extra_cscomp"
python="$work/icml/envs/t2irare/bin/python"
data_dir=${1:-"$TMPDIR/codex-fallible-verifier-08f42e1"}
output="results/replay.${SLURM_JOB_ID}.json"

printf '[codex] %s Starting frozen fallible-verifier replay\n' "$(date '+%Y-%m-%d %H:%M')" |
    tee "results/logs/job.${SLURM_JOB_ID}.log"
"$python" replay.py --public --data-dir "$data_dir" --output "$output" 2>&1 |
    awk '{sub(/^\[codex\] /, ""); print "[codex] " strftime("%Y-%m-%d %H:%M") " " $0; fflush()}' |
    tee -a "results/logs/job.${SLURM_JOB_ID}.log"
