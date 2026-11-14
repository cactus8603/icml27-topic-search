#!/bin/bash
#SBATCH --job-name=t2ir-clip-precision
#SBATCH --partition=h200q
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=48G
#SBATCH --time=00:30:00
#SBATCH --output=logs/%x-%j.out
set -euo pipefail
source slurm/env.sh
freeze="${1:?Pass the full frozen commit}"
[[ "$freeze" =~ ^[a-f0-9]{40}$ ]]
topic=killtests/t2i-rare-failure
repo=$(git rev-parse --show-toplevel)
snapshot=$(mktemp -d "$TMPDIR/clip-precision-$SLURM_JOB_ID.XXXXXXXX")
trap 'rm -r --one-file-system -- "$snapshot"' EXIT
git -C "$repo" archive "$freeze" "$topic/configs" "$topic/scripts" | tar -xf - -C "$snapshot"
out="/work/ChihChungHsu/cactus8603/icml/runs/t2irare/clip-precision/$SLURM_JOB_ID.json"
"$PY" -B "$snapshot/$topic/scripts/clip_precision_probe.py" --freeze "$freeze" --out "$out"
