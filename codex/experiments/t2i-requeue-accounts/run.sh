#!/usr/bin/env bash
set -euo pipefail

revision=${1:?Pass the accounting overlay commit}
operation=${2:?Choose --check-only, plan, aggregate, review-stats, or watch}
shift 2
[[ $revision =~ ^[0-9a-f]{40}$ ]]
repo=$(git -C "${SLURM_SUBMIT_DIR:-$PWD}" rev-parse --show-toplevel)
stage=codex/experiments/t2i-requeue-accounts
formal=4351ebb70f2e7ad72e0ec8b99658d9ed41c06567
runner=codex/experiments/t2i-formal-replay/run.sbatch
receipt=d88b2d37b09d64a57a3102ecd7049c57d616aa19ed3fcc85a69a5dc447f9cffe
shared=/work/ChihChungHsu/cactus8603/icml
export TZ=Asia/Taipei TMPDIR="$shared/tmp/t2irare"
bundle=$(mktemp -d -- "$TMPDIR/t2ir-requeue-repair.XXXXXXXX")
trap 'rm -r --one-file-system -- "$bundle"' EXIT
git -C "$repo" archive "$revision" "$stage" | tar -xf - -C "$bundle"
cmp -- "${BASH_SOURCE[0]}" "$bundle/$stage/run.sh"
python="$shared/envs/t2irare/bin/python"
export T2I_REQUEUE_ENTRY="$bundle/$stage/requeue.py"
unset T2I_REQUEUE_LABEL_FIX
if [[ $operation = watch ]]; then
    "$python" -IB "$T2I_REQUEUE_ENTRY" watch "$repo" "$@"
    exit
fi

git -C "$repo" show "$formal:$runner" > "$bundle/requeue-run.sbatch"
"$python" -IB "$T2I_REQUEUE_ENTRY" runner "$bundle/requeue-run.sbatch"
cd "$repo"
case "$operation" in
    --check-only)
        [[ $# == 0 ]]
        "$python" -IB "$bundle/$stage/test_requeue.py" "$repo"
        bash -n "$bundle/requeue-run.sbatch"
        bash "$bundle/requeue-run.sbatch" "$formal" preflight
        ;;
    plan)
        [[ $# == 1 && $1 =~ ^(2a|2b|2b_extra)$ ]]
        bash "$bundle/requeue-run.sbatch" "$formal" --receipt-sha "$receipt" plan "$1"
        ;;
    aggregate|review-stats)
        [[ $# == 0 && $operation = aggregate || $# == 1 && $1 =~ ^[0-9a-f]{40}$ ]]
        if [[ $# == 1 ]]; then
            label=codex/experiments/t2i-blind-label-zero/fix_labels.py
            git -C "$repo" archive "$1" "$label" | tar -xf - -C "$bundle"
            export T2I_REQUEUE_LABEL_FIX="$bundle/$label"
        fi
        bash "$bundle/requeue-run.sbatch" "$formal" --receipt-sha "$receipt" "$operation"
        ;;
    *) exit 2 ;;
esac
