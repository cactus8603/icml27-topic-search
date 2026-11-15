#!/usr/bin/env bash
#SBATCH --job-name=t2ir-detector-probe
#SBATCH --partition=h200q
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=16
#SBATCH --mem=48G
#SBATCH --time=00:15:00
#SBATCH --output=/work/ChihChungHsu/cactus8603/icml/tmp/t2ir-detector-probe-%j.log
set -euo pipefail

freeze=${1:?Pass the frozen commit}
[[ $freeze =~ ^[0-9a-f]{40}$ ]]
[[ $# == 1 || $# == 2 && $2 == --check-only ]]
repo=$(git -C "${SLURM_SUBMIT_DIR:-$PWD}" rev-parse --show-toplevel)
stage=codex/experiments/t2i-detector-probe
topic=killtests/t2i-rare-failure
reference=b54a86fb05e845e2c11139822beac8846c63c77a
changes=$(git -C "$repo" status --porcelain --untracked-files=all -- "$stage")
[[ -z "$changes" ]]
git -C "$repo" diff --quiet "$freeze" HEAD -- "$stage"
shared=/work/ChihChungHsu/cactus8603/icml
export TZ=Asia/Taipei TMPDIR="$shared/tmp" HF_HOME="$shared/hf_home"
export HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 PYTHONUNBUFFERED=1
export OMP_NUM_THREADS=${SLURM_CPUS_PER_TASK:-4} OPENBLAS_NUM_THREADS=1
export SLURM_CPUS_PER_TASK=${SLURM_CPUS_PER_TASK:-4}
bundle=$(mktemp -d "$TMPDIR/t2ir-detector-probe-${SLURM_JOB_ID:-check}.XXXXXXXX")
trap 'rm -r --one-file-system -- "$bundle"' EXIT
export XDG_CACHE_HOME="$bundle/xdg" TORCH_HOME="$bundle/torch"
git -C "$repo" archive "$freeze" "$stage" | tar -x -C "$bundle"
git -C "$repo" archive "$reference" \
    "$topic"/configs/{detectors,sampling}.json "$topic"/scripts/{common,detectors}.py | tar -x -C "$bundle"
args=(--preflight)
if [[ ${2:-} != --check-only ]]; then
    : "${SLURM_JOB_ID:?Submit through Slurm}"
    quota -g -s || true
    args=(--gpu --freeze "$freeze")
fi
"$shared/envs/t2irare/bin/python" -IB "$bundle/$stage/probe.py" --upstream "$bundle/$topic" "${args[@]}"
