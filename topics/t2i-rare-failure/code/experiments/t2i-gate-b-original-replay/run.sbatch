#!/usr/bin/env bash
#SBATCH --job-name=t2ir-gate-b-original-replay
#SBATCH --partition=h200q
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=16
#SBATCH --mem=64G
#SBATCH --time=03:00:00
#SBATCH --output=/work/ChihChungHsu/cactus8603/icml/tmp/t2irare/t2ir-gate-b-original-replay-%j.out
set -euo pipefail

if [[ $# -eq 1 && $1 =~ ^[0-9a-f]{40}$ && ${SLURM_JOB_ID:-} =~ ^[0-9]+$ ]]; then
    mode=(--run)
elif [[ $# -eq 2 && $1 =~ ^[0-9a-f]{40}$ && $2 == --check-only ]]; then
    mode=(--preflight)
elif [[ $# -eq 3 && $1 =~ ^[0-9a-f]{40}$ && $2 == --verify-job && $3 =~ ^[0-9]+$ ]]; then
    mode=(--verify-job "$3")
else
    printf 'Usage: run.sbatch FULL_FREEZE_COMMIT [--check-only | --verify-job JOB_ID]\n' >&2
    exit 2
fi

freeze=$1
topic=killtests/t2i-rare-failure
experiment=codex/experiments/t2i-gate-b-original-replay
precision=codex/experiments/t2i-gate-b-precision/run.py
generator=f54cb310a921966440754d42f65f86d9f565dc62
reference=37e6ae9a63cd3035822dccca2acf90ef12efec93
precision_repeat=2b1c7f64b8b9013a9a32f8353024c1d6f82b6e53
repo=$(git -C "${SLURM_SUBMIT_DIR:-$PWD}" rev-parse --show-toplevel)
test -z "$(git -C "$repo" status --porcelain --untracked-files=all -- "$experiment")"
git -C "$repo" diff --quiet "$freeze" HEAD -- "$experiment"

shared=/work/ChihChungHsu/cactus8603/icml
export TZ=Asia/Taipei HF_HOME="$shared/hf_home" HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 PYTHONUNBUFFERED=1
export TMPDIR="$shared/tmp/t2irare"
export XDG_CACHE_HOME="$TMPDIR/xdg" TORCH_HOME="$TMPDIR/torch" TRITON_CACHE_DIR="$TMPDIR/triton" MPLCONFIGDIR="$TMPDIR/mpl"
export OMP_NUM_THREADS=${SLURM_CPUS_PER_TASK:-16}
mkdir -p -- "$TMPDIR" "$XDG_CACHE_HOME" "$TORCH_HOME" "$TRITON_CACHE_DIR" "$MPLCONFIGDIR"
bundle=$(mktemp -d -- "$TMPDIR/gate-b-original-replay.${SLURM_JOB_ID:-check}.XXXXXXXX")
trap 'rm -r --one-file-system -- "$bundle"' EXIT

git -C "$repo" archive "$freeze" "$experiment" | tar -xf - -C "$bundle"
git -C "$repo" archive "$generator" "$topic/scripts/generate.py" | tar -xf - -C "$bundle"
git -C "$repo" archive "$precision_repeat" "$precision" | tar -xf - -C "$bundle"
git -C "$repo" archive "$reference" "$topic"/configs/{criteria,detectors,sampling,models,prompts}.json \
    "$topic"/scripts/{clip_precision_probe,common,detectors,evidence,verify_gate_b}.py | tar -xf - -C "$bundle"

python="$shared/envs/t2irare/bin/python"
if [[ ${mode[0]} == --preflight ]]; then
    "$python" -IB "$bundle/$experiment/test_run.py"
fi
"$python" -IB "$bundle/$experiment/run.py" --upstream "$bundle/$topic" --original "$repo/$topic" \
    --precision "$bundle/$precision" --freeze "$freeze" "${mode[@]}"
