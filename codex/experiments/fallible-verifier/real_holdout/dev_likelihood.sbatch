#!/usr/bin/env bash
#SBATCH --job-name=fv-gsm8k-dev-odds
#SBATCH --partition=h200q
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=48G
#SBATCH --time=01:00:00
#SBATCH --output=/work/ChihChungHsu/cactus8603/icml/tmp/fv-gsm8k-dev-odds-%j.log
set -euo pipefail

if [[ $# != 1 || ! $1 =~ ^[0-9a-f]{40}$ ]]; then
    printf 'Usage: sbatch codex/experiments/fallible-verifier/real_holdout/dev_likelihood.sbatch FROZEN_COMMIT\n' >&2
    exit 2
fi
freeze=$1
job=${SLURM_JOB_ID:?Use Slurm for the GPU diagnostic}
[[ $job =~ ^[0-9]+$ ]]
repo=$(git -C "${SLURM_SUBMIT_DIR:?}" rev-parse --show-toplevel)
topic=codex/experiments/fallible-verifier
experiment="$topic/real_holdout"
native=b6023cea71bdfb4abceac3567ab1c0a81e3eea4a
panel=37a8a783bd7b1f7648dfbf0376cbbb99099cad58
changes=$(git -C "$repo" status --porcelain -- "$experiment")
[[ -z "$changes" ]]
git -C "$repo" diff --quiet "$freeze" HEAD -- "$experiment"
git -C "$repo" diff --quiet "$panel" "$freeze" -- "$experiment/prepare.py"

shared=/work/ChihChungHsu/cactus8603/icml
run="$shared/runs/fallible-verifier/gsm8k-dev-likelihood/$job"
export TZ=Asia/Taipei PYTHONDONTWRITEBYTECODE=1 PYTHONUNBUFFERED=1
export TMPDIR="$shared/tmp" HF_HOME="$shared/hf_home"
export HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 TOKENIZERS_PARALLELISM=false
export OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 VLLM_NO_USAGE_STATS=1
export VLLM_USE_FLASHINFER_SAMPLER=0 TORCH_COMPILE_DISABLE=1
snapshot=$(mktemp -d -- "$TMPDIR/fv-gsm8k-dev-odds.$job.XXXXXXXX")
trap 'rm -r --one-file-system -- "$snapshot"' EXIT
export XDG_CACHE_HOME="$snapshot/xdg"
git -C "$repo" archive "$freeze" "$experiment" | tar -x -C "$snapshot"
git -C "$repo" archive "$native" "$topic"/{phase3_likelihood,phase3_feasibility,phase3_synthetic}.py \
    "$topic"/round2/{config.json,protocol.py} | tar -x -C "$snapshot"

for role in weak strong; do
    printf '[codex] %s train-only assay %s\n' "$(date '+%Y-%m-%d %H:%M')" "$role"
    # Native engine failure diagnostics may embed scheduler inputs.
    if ! "$shared/envs/vllm033/bin/python" -IB "$snapshot/$experiment/dev_likelihood.py" \
        --role "$role" --run-dir "$run" --freeze "$freeze" >/dev/null 2>&1; then
        printf '[codex] %s %s failed to complete the train-only assay\n' "$(date '+%Y-%m-%d %H:%M')" "$role" >&2
        exit 1
    fi
done
