#!/usr/bin/env bash
#SBATCH --job-name=t2ir-tie-secondary-key
#SBATCH --partition=h200q
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gres=none
#SBATCH --cpus-per-task=2
#SBATCH --mem=4G
#SBATCH --time=00:05:00
#SBATCH --output=/work/ChihChungHsu/cactus8603/icml/tmp/t2ir-tie-secondary-key-%j.log
set -euo pipefail

freeze=${1:?Pass the reviewed full freeze}
[[ "$freeze" =~ ^[0-9a-f]{40}$ ]]
[[ "$#" == 1 || "$#" == 2 && "$2" == --check-only ]]
repo=$(git -C "${SLURM_SUBMIT_DIR:-$PWD}" rev-parse --show-toplevel)
stage=codex/experiments/t2i-tie-secondary-key
previous=codex/experiments/t2i-tied-scores/run.py
source=killtests/t2i-rare-failure/scripts/estimators.py
previous_commit=bf261c35a5296f6625e8dff889e76997bfb3adfa
source_commit=b54a86fb05e845e2c11139822beac8846c63c77a
[[ -z "$(git -C "$repo" status --porcelain --untracked-files=all -- "$stage")" ]]
git -C "$repo" diff --quiet "$freeze" HEAD -- "$stage"

shared=/work/ChihChungHsu/cactus8603/icml
export TZ=Asia/Taipei CUDA_VISIBLE_DEVICES='' TMPDIR="$shared/tmp"
export OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1
umask 077
bundle=$(mktemp -d "$TMPDIR/t2ir-tie-key-${SLURM_JOB_ID:-check}.XXXXXXXX")
trap 'rm -r --one-file-system -- "$bundle"' EXIT
git -C "$repo" archive "$freeze" "$stage" | tar -x -C "$bundle"
git -C "$repo" archive "$previous_commit" "$previous" | tar -x -C "$bundle"
git -C "$repo" archive "$source_commit" "$source" | tar -x -C "$bundle"
python="$shared/envs/t2irare/bin/python"
entry="$bundle/$stage/run.py"
if [[ ${2:-} == --check-only ]]; then
    "$python" -IB "$entry" --source "$bundle/$source" --check-only
    "$python" -IB "$bundle/$stage/test_geometry.py"
    exit 0
fi

job=${SLURM_JOB_ID:?Submit through Slurm}
[[ "$job" =~ ^[1-9][0-9]*$ ]]
base="$shared/runs/t2irare/tie-secondary-key"
mkdir -p -- "$base"
out="$base/$job"
mkdir -- "$out"
"$python" -IB "$entry" --source "$bundle/$source" --run --freeze "$freeze" > "$out/summary.partial"
mv -- "$out/summary.partial" "$out/summary.json"
sha256sum -- "$out/summary.json"
