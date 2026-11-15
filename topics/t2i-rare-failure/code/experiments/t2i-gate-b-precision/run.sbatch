#!/usr/bin/env bash
#SBATCH --job-name=t2ir-gate-b-precision
#SBATCH --partition=h200q
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=48G
#SBATCH --time=02:00:00
#SBATCH --output=/work/ChihChungHsu/cactus8603/icml/runs/t2irare/t2ir-gate-b-precision-%j.out
set -euo pipefail

if [[ $# != 1 || ! $1 =~ ^[0-9a-f]{40}$ ]]; then
    printf 'Usage: sbatch codex/experiments/t2i-gate-b-precision/run.sbatch FROZEN_COMMIT\n' >&2
    exit 2
fi
freeze=$1
reference=37e6ae9a63cd3035822dccca2acf90ef12efec93
topic=killtests/t2i-rare-failure
experiment=codex/experiments/t2i-gate-b-precision
repo=$(git -C "${SLURM_SUBMIT_DIR:?}" rev-parse --show-toplevel)
job_id=${SLURM_JOB_ID:?}
[[ $job_id =~ ^[0-9]+$ ]]

shared=/work/ChihChungHsu/cactus8603/icml
export HF_HOME="$shared/hf_home" HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 PYTHONUNBUFFERED=1
export TMPDIR="$shared/tmp/t2irare"
export XDG_CACHE_HOME="$TMPDIR/xdg" TORCH_HOME="$TMPDIR/torch" TRITON_CACHE_DIR="$TMPDIR/triton" MPLCONFIGDIR="$TMPDIR/mpl"
export OMP_NUM_THREADS=${SLURM_CPUS_PER_TASK:-8}
mkdir -p -- "$TMPDIR" "$XDG_CACHE_HOME" "$TORCH_HOME" "$TRITON_CACHE_DIR" "$MPLCONFIGDIR"
source_dir=$(mktemp -d -- "$TMPDIR/gate-b-precision.$job_id.XXXXXXXX")
trap 'rm -r --one-file-system -- "$source_dir"' EXIT
git -C "$repo" archive "$freeze" "$experiment" | tar -xf - -C "$source_dir"
git -C "$repo" archive "$reference" "$topic"/configs/{criteria,detectors,sampling,models,prompts}.json \
    "$topic"/scripts/{clip_precision_probe,common,detectors,evidence}.py | tar -xf - -C "$source_dir"

out="$shared/runs/t2irare/gate-b-precision/$job_id.json"
"$shared/envs/t2irare/bin/python" -IB "$source_dir/$experiment/run.py" \
    --upstream "$source_dir/$topic" --freeze "$freeze" --out "$out"
