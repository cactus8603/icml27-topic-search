#!/usr/bin/env bash
set -euo pipefail

fix=${1:?Pass the label parser fix commit}
formal=${2:?Pass the formal sampling commit}
command=${3:?Choose --check-only, review-stats, or aggregate}
[[ $fix =~ ^[0-9a-f]{40}$ && $formal = 4351ebb70f2e7ad72e0ec8b99658d9ed41c06567 ]]
case "$command" in
    --check-only) [[ $# == 3 ]] ;;
    review-stats|aggregate) [[ $# == 4 && $4 =~ ^[0-9a-f]{64}$ ]] ;;
    *) exit 2 ;;
esac

repo=$(git -C "${SLURM_SUBMIT_DIR:-$PWD}" rev-parse --show-toplevel)
extension=codex/experiments/t2i-blind-label-zero
runner=codex/experiments/t2i-formal-replay/run.sbatch
shared=/work/ChihChungHsu/cactus8603/icml
export TZ=Asia/Taipei TMPDIR="$shared/tmp/t2irare"
bundle=$(mktemp -d -- "$TMPDIR/t2ir-label-fix.XXXXXXXX")
trap 'rm -r --one-file-system -- "$bundle"' EXIT
git -C "$repo" archive "$fix" "$extension" | tar -xf - -C "$bundle"
cmp -- "$bundle/$extension/run.sh" "${BASH_SOURCE[0]}"
git -C "$repo" show "$formal:$runner" > "$bundle/review-run.sbatch"
export T2I_REVIEW_LABEL_FIX="$bundle/$extension/fix_labels.py"
python="$shared/envs/t2irare/bin/python"
"$python" -IB "$T2I_REVIEW_LABEL_FIX" runner "$bundle/review-run.sbatch"

cd "$repo"
if [[ $command = --check-only ]]; then
    "$python" -IB "$bundle/$extension/test_fix_labels.py" "$repo" "$bundle"
    bash -n "$bundle/review-run.sbatch"
    bash "$bundle/review-run.sbatch" "$formal" preflight
else
    bash "$bundle/review-run.sbatch" "$formal" --receipt-sha "$4" "$command"
fi
