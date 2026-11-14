#!/usr/bin/env bash
#SBATCH --job-name=fv2-analysis
#SBATCH --partition=h200q
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=1
#SBATCH --mem=4G
#SBATCH --time=00:10:00
set -euo pipefail

root=/work/ChihChungHsu/cactus8603
shared="$root/icml"
topic="$root/icml27-topic-search/codex/experiments/fallible-verifier/round2"
freeze=${1:?Pass reviewed freeze commit}
input_job=${2:?Pass inference job ID}
[[ "$input_job" =~ ^[0-9]+$ ]]
export TZ=Asia/Taipei PYTHONDONTWRITEBYTECODE=1
export TMPDIR="$shared/tmp" OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1
"$shared/envs/t2irare/bin/python" -B "$topic/analysis.py" --freeze "$freeze" \
  --run-dir "$topic/results/$input_job" 2>&1 |
  awk '{sub(/^\[codex\] /, ""); print "[codex] " strftime("%Y-%m-%d %H:%M") " " $0; fflush()}'
