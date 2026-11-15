#!/usr/bin/env bash
#SBATCH --job-name=t2ir-s1-candidate
#SBATCH --partition=h200q
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=16
#SBATCH --mem=96G
#SBATCH --time=04:00:00
#SBATCH --array=0-5
#SBATCH --output=/work/ChihChungHsu/cactus8603/icml/tmp/t2ir-s1-candidate-%A_%a.log
set -euo pipefail

freeze=${1:?Pass the full launch freeze}
[[ "$freeze" =~ ^[0-9a-f]{40}$ ]]
[[ $# == 1 || $# == 2 && $2 == --check-only ]]
repo=$(git -C "${SLURM_SUBMIT_DIR:-$PWD}" rev-parse --show-toplevel)
stage=codex/experiments/t2i-stage1-candidate
probe=codex/experiments/t2i-detector-probe
repair=codex/experiments/t2i-gate-repair
topic=killtests/t2i-rare-failure
source_commit=b54a86fb05e845e2c11139822beac8846c63c77a
probe_commit=0d86f961a09954cec06434999e63b136bf0a83b1
changes=$(git -C "$repo" status --porcelain --untracked-files=all -- "$stage")
[[ -z "$changes" ]]
git -C "$repo" diff --quiet "$freeze" HEAD -- "$stage"
mode=--check-only
if [[ ${2:-} != --check-only ]]; then
    : "${SLURM_JOB_ID:?Submit through Slurm}"
    if ! git -C "$repo" cat-file -e "$freeze:$stage/activation.json"; then
        printf '[codex] Candidate Stage 1 awaits the committed user decision.\n' >&2
        exit 2
    fi
    quota -g -s || true
    mode=--execute
fi

shared=/work/ChihChungHsu/cactus8603/icml
export TZ=Asia/Taipei TMPDIR="$shared/tmp" HF_HOME="$shared/hf_home"
export HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 PYTHONUNBUFFERED=1
export OMP_NUM_THREADS=${SLURM_CPUS_PER_TASK:-4} OPENBLAS_NUM_THREADS=1
bundle=$(mktemp -d "$TMPDIR/t2ir-s1-candidate-${SLURM_JOB_ID:-check}.XXXXXXXX")
trap 'rm -r --one-file-system -- "$bundle"' EXIT
export XDG_CACHE_HOME="$bundle/xdg" TORCH_HOME="$bundle/torch"
export TRITON_CACHE_DIR="$bundle/triton" MPLCONFIGDIR="$bundle/mpl"
git -C "$repo" archive "$freeze" "$stage" | tar -x -C "$bundle"
git -C "$repo" archive "$probe_commit" "$probe"/{probe,stage1_check}.py | tar -x -C "$bundle"
git -C "$repo" archive "$source_commit" "$repair/review_evidence.py" \
    "$topic"/configs/{criteria,detectors,models,prompts,sampling}.json \
    "$topic"/scripts/{common,detectors,evidence,generate,verify_gate_b}.py \
    "$topic"/slurm/stage1.sbatch | tar -x -C "$bundle"
"$shared/envs/t2irare/bin/python" -IB "$bundle/$stage/run.py" "$mode" \
    --freeze "$freeze" --repo "$repo" --upstream "$bundle/$topic"
