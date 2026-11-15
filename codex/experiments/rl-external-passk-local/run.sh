#!/usr/bin/env bash
set -euo pipefail

if [[ "$#" == 1 && "$1" == --check-only ]]; then
    phase=check
elif [[ "$#" == 2 && "$1" == --run && "$2" =~ ^[0-9a-f]{40}$ ]]; then
    phase=run
    launcher_freeze=$2
    [[ -z "${SLURM_JOB_ID:-}" ]]
else
    printf 'Usage: %s --check-only | --run FULL_LAUNCHER_COMMIT\n' "$0" >&2
    exit 2
fi

repo=$(git -C "$(dirname -- "$0")" rev-parse --show-toplevel)
local_stage=codex/experiments/rl-external-passk-local
frozen_stage=codex/experiments/rl-external-passk
experiment=10d42bdf86534a44b47c03d56ea5e1dd749cdbac
source=killtests/rl-sparse-distill
baseline=5808cb676c8fc5ab52b74d7adb0601caa4cfedd6
shared=/work/ChihChungHsu/cactus8603/icml
assets="$shared/data/rl-external-passk/ac7c3fc58d5e325447607e778bdcbc897182cf7e"
python="$shared/envs/attrib-jax/bin/python"

export TMPDIR="$shared/tmp" TZ=Asia/Taipei CUDA_VISIBLE_DEVICES=''
export OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1 BLIS_NUM_THREADS=1 NUMEXPR_NUM_THREADS=1
umask 077
bundle=$(mktemp -d "$TMPDIR/rsd-external-passk-local.XXXXXXXX")
trap 'rm -r --one-file-system -- "$bundle"' EXIT
git -C "$repo" archive "$experiment" "$frozen_stage" | tar -x -C "$bundle"
git -C "$repo" archive "$baseline" "$source"/scripts/{aggregate,artifacts,common}.py "$source/configs/criteria.json" | tar -x -C "$bundle"

if [[ "$phase" == check ]]; then
    timeout --signal=KILL 30s "$python" -IB "$bundle/$frozen_stage/test_analyze.py"
    timeout --signal=KILL 30s "$python" -IB "$bundle/$frozen_stage/analyze.py" check \
        --assets "$assets" --source "$bundle/$source" --freeze "$experiment"
    exit 0
fi

[[ -z "$(git -C "$repo" status --porcelain --untracked-files=all -- "$local_stage")" ]]
git -C "$repo" diff --quiet "$launcher_freeze" HEAD -- "$local_stage"
git -C "$repo" archive "$launcher_freeze" "$local_stage" | tar -x -C "$bundle"
run_id=${bundle##*/}
output_dir="$shared/runs/rsd-external-passk-local"
mkdir -p -- "$output_dir"
timeout --signal=KILL 30s "$python" -IB "$bundle/$local_stage/replay.py" \
    --assets "$assets" --source "$bundle/$source" --analysis "$bundle/$frozen_stage" \
    --freeze "$experiment" --baseline "$baseline" --launcher-freeze "$launcher_freeze" \
    --run-id "$run_id" --output "$bundle/summary.json"
result="$output_dir/$run_id.json"
ln -T -- "$bundle/summary.json" "$result"
printf '[codex] %s 本機 CPU 重播收據\n' "$(date '+%F %R')"
sha256sum -- "$result"
