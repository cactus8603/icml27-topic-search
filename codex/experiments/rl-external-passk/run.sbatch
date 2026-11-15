#!/usr/bin/env bash
#SBATCH --job-name=rsd-external-passk
#SBATCH --partition=h200q
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gres=none
#SBATCH --cpus-per-task=1
#SBATCH --mem=2G
#SBATCH --time=00:05:00
#SBATCH --output=/work/ChihChungHsu/cactus8603/icml/tmp/rsd-external-passk-%j.log
set -euo pipefail

[[ "$#" == 2 && "$1" =~ ^[0-9a-f]{40}$ ]]
freeze=$1
phase=$2
case "$phase" in
    --check-only) job=check ;;
    --run) job=${SLURM_JOB_ID:?Submit via Slurm}; [[ "$job" =~ ^[1-9][0-9]*$ ]] ;;
    *) exit 2 ;;
esac

repo=$(git -C "${SLURM_SUBMIT_DIR:-$PWD}" rev-parse --show-toplevel)
stage=codex/experiments/rl-external-passk
source=killtests/rl-sparse-distill
baseline=5808cb676c8fc5ab52b74d7adb0601caa4cfedd6
shared=/work/ChihChungHsu/cactus8603/icml
assets="$shared/data/rl-external-passk/ac7c3fc58d5e325447607e778bdcbc897182cf7e"
python="$shared/envs/attrib-jax/bin/python"
[[ -z "$(git -C "$repo" status --porcelain --untracked-files=all -- "$stage")" ]]
git -C "$repo" diff --quiet "$freeze" HEAD -- "$stage"

export TMPDIR="$shared/tmp" TZ=Asia/Taipei OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1
export CUDA_VISIBLE_DEVICES=''
umask 077
bundle=$(mktemp -d "$TMPDIR/rsd-external-passk-$job.XXXXXXXX")
trap 'rm -r --one-file-system -- "$bundle"' EXIT
git -C "$repo" archive "$freeze" "$stage" | tar -x -C "$bundle"
git -C "$repo" archive "$baseline" "$source"/scripts/{aggregate,artifacts,common}.py "$source/configs/criteria.json" | tar -x -C "$bundle"

entry="$bundle/$stage/analyze.py"
inputs=(--assets "$assets" --source "$bundle/$source" --freeze "$freeze")
if [[ "$phase" == --check-only ]]; then
    "$python" -IB "$bundle/$stage/test_analyze.py"
    "$python" -IB "$entry" check "${inputs[@]}"
    exit 0
fi

outdir="$shared/runs/rsd-external-passk"
mkdir -p "$outdir"
[[ ! -e "$outdir/$job.json" && ! -L "$outdir/$job.json" ]]
"$python" -IB "$entry" run "${inputs[@]}" --job "$job" --output "$bundle/summary.json"
ln "$bundle/summary.json" "$outdir/$job.json"
sha256sum -- "$outdir/$job.json"
