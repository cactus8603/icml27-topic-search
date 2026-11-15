#!/usr/bin/env bash
#SBATCH --job-name=lambda-gcd-family-comparison
#SBATCH --partition=h200q
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gres=none
#SBATCH --cpus-per-task=1
#SBATCH --mem=4G
#SBATCH --time=00:05:00
#SBATCH --output=/work/ChihChungHsu/cactus8603/icml/tmp/lambda-gcd-family-comparison-%j.log
set -euo pipefail
umask 077

root=/work/ChihChungHsu/cactus8603
repo="$root/icml27-topic-search"
shared="$root/icml"
stage=codex/experiments/lambda-gcd-family-design/evaluation
freeze=${1:?Pass the full frozen Git commit}
[[ "$freeze" =~ ^[0-9a-f]{40}$ ]]
[[ -z "$(git -C "$repo" status --porcelain --untracked-files=all -- "$stage")" ]]
git -C "$repo" diff --quiet "$freeze" HEAD -- "$stage"

option=${2:-}
if [[ "$option" == --check-only ]]; then
    [[ "$#" == 2 && -z "${SLURM_JOB_ID:-}" ]]
else
    [[ "$#" == 5 && "${SLURM_JOB_ID:-}" =~ ^[1-9][0-9]*$ && -z "${SLURM_ARRAY_JOB_ID:-}" ]]
fi
export TZ=Asia/Taipei TMPDIR="$shared/tmp" PYTHONDONTWRITEBYTECODE=1 CUDA_VISIBLE_DEVICES=''
bundle=$(mktemp -d "$TMPDIR/lambda-gcd-family-comparison-${SLURM_JOB_ID:-check}.XXXXXXXX")
trap 'rm -r --one-file-system -- "$bundle"' EXIT
export GCD_JOIN_TEST_TMP="$bundle"
git -C "$repo" archive "$freeze" "$stage" | tar -xf - -C "$bundle"
if [[ "$option" == --check-only ]]; then
    python3 -B -m unittest discover -s "$bundle/$stage"
else
    python3 -IBu "$bundle/$stage/join.py" --freeze "$freeze" --ledger "$repo/killtests/SUBMITTED.log" \
        --step0-array "$2" --step0-join "$3" --outcome-array "$4" --outcome-judge "$5"
fi
