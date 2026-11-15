#!/usr/bin/env bash
#SBATCH --job-name=cham-clean-labels
#SBATCH --partition=h200q
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gres=none
#SBATCH --cpus-per-task=1
#SBATCH --mem=2G
#SBATCH --time=00:05:00
#SBATCH --output=/work/ChihChungHsu/cactus8603/icml/tmp/cham-clean-labels-%j.log
set -euo pipefail

[[ "$#" == 2 && "$1" =~ ^[0-9a-f]{40}$ ]]
freeze=$1
phase=$2
[[ "$phase" == --check-only || "$phase" == --run && "${SLURM_JOB_ID:-}" =~ ^[0-9]+$ ]]
repo=$(git -C "${SLURM_SUBMIT_DIR:-$PWD}" rev-parse --show-toplevel)
stage=codex/experiments/chameleon-clean-labels
source=codex/experiments/chameleon-source-audit/audit.py
[[ -z "$(git -C "$repo" status --porcelain --untracked-files=all -- "$stage")" ]]
git -C "$repo" diff --quiet "$freeze" HEAD -- "$stage"
[[ "$(git -C "$repo" rev-parse "$freeze:$source")" == 0493728ee2e03939d1120c75c886a7303b592181 ]]

shared=/work/ChihChungHsu/cactus8603/icml
export TMPDIR="$shared/tmp" TZ=Asia/Taipei OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1
bundle=$(mktemp -d "$TMPDIR/cham-clean-labels-${SLURM_JOB_ID:-check}.XXXXXXXX")
trap 'rm -r --one-file-system -- "$bundle"' EXIT
git -C "$repo" archive "$freeze" "$stage" "$source" | tar -x -C "$bundle"
"$shared/envs/inoc/bin/python" -IB "$bundle/$stage/analyze.py" "$phase" --freeze "$freeze"
