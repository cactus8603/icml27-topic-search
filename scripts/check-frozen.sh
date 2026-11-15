#!/usr/bin/env bash
set -euo pipefail

if (( $# < 3 )); then
    printf 'Usage: %s <frozen commit> <original launcher path> [arguments] --check-only\n' "$0" >&2
    exit 2
fi

repo=$(git rev-parse --show-toplevel)
freeze=$(git -C "$repo" rev-parse --verify --end-of-options "$1^{commit}")
launcher=$2
shift 2
[[ "$launcher" != /* && "/$launcher/" != */../* && "$launcher" =~ \.(sbatch|sh)$ ]]
check_only=false
for arg in "$@"; do
    if [[ "$arg" == --check-only ]]; then
        check_only=true
    fi
done
[[ "$check_only" == true ]] || { printf 'The frozen launcher must receive --check-only.\n' >&2; exit 2; }

shared=/work/ChihChungHsu/cactus8603/icml
scratch=$(mktemp -d "$shared/tmp/topic-preflight.XXXXXXXX")
checkout="$scratch/source"
launch_copy="$scratch/launcher"
cleanup() {
    if [[ -e "$checkout/.git" ]]; then
        git -C "$repo" worktree remove --force "$checkout"
    fi
    if [[ -f "$launch_copy" ]]; then
        rm -- "$launch_copy"
    fi
    rmdir -- "$scratch"
}
trap cleanup EXIT

git -C "$repo" worktree add --quiet --detach "$checkout" "$freeze"
entry="$checkout/$launcher"
[[ -f "$entry" && ! -L "$entry" ]]
# Historical launchers with a fixed repo path use this scratch checkout; all
# versioned inputs, frozen path checks, data paths and job options stay intact.
if grep -Fxq 'repo="$root/icml27-topic-search"' "$entry" ||
   grep -Fxq 'repo=/work/ChihChungHsu/cactus8603/icml27-topic-search' "$entry"; then
    sed -e 's|^repo="\$root/icml27-topic-search"$|repo="$FROZEN_CHECKOUT"|' \
        -e 's|^repo=/work/ChihChungHsu/cactus8603/icml27-topic-search$|repo="$FROZEN_CHECKOUT"|' \
        "$entry" > "$launch_copy"
    entry=$launch_copy
fi

if [[ "$launcher" == codex/experiments/rl-fork-budget/run.sbatch ]]; then
    grader="${RL_GRADER_PATH:-/work/ChihChungHsu/cactus8603/icml27-topic-search/killtests/rl-sparse-distill/pylib}"
    [[ -d "$grader" ]] || { printf 'Set RL_GRADER_PATH to the experiment Math-Verify directory.\n' >&2; exit 2; }
    ln -s -- "$(realpath -- "$grader")" "$checkout/killtests/rl-sparse-distill/pylib"
fi

cd "$checkout"
env -u SLURM_SUBMIT_DIR -u SLURM_JOB_ID -u SLURM_ARRAY_JOB_ID -u SLURM_ARRAY_TASK_ID \
    CUDA_VISIBLE_DEVICES="" FROZEN_CHECKOUT="$checkout" LAMBDA_GCD_REPO="$checkout" \
    bash "$entry" "$freeze" "$@"
