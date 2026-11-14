#!/usr/bin/env bash
set -euo pipefail

work=/work/ChihChungHsu/cactus8603
repo="$work/icml27-topic-search"
icml="$work/icml"
topic=codex/experiments/gate-optimizer
stage="$topic/round3"
source_commit=4d87430fd7cda6e74ea9e48deb57411911733e04
action=${1:?Pass produce or analyze}
freeze=${2:?Pass the full freeze commit}
option=${3:-}
[[ "$freeze" =~ ^[0-9a-f]{40}$ && $# -le 3 ]]
[[ "$(git -C "$repo" rev-parse --verify "$freeze^{commit}")" == "$freeze" ]]
changes=$(git -C "$repo" status --porcelain --untracked-files=all -- "$topic")
[[ -z "$changes" ]]
git -C "$repo" diff --quiet "$freeze" HEAD -- "$topic"
git -C "$repo" diff --quiet "$freeze" -- "$topic"

dependency=("$topic/config.json" "$topic/"{data,model,run,aggregate}.py
            "$topic/round2/"{config.json,experiment.py,judge.py,validation.py})
git -C "$repo" diff --quiet "$source_commit" "$freeze" -- "${dependency[@]}"
git -C "$repo" cat-file -e "$freeze:$stage/panel.py"
tree=$(git -C "$repo" rev-parse "$freeze:$topic")
python="$icml/envs/vllm033/bin/python"
export TZ=Asia/Taipei TMPDIR="$icml/tmp" HF_HOME="$icml/hf_home" PYTHONDONTWRITEBYTECODE=1
export OMP_NUM_THREADS="${SLURM_CPUS_PER_TASK:-1}" RAYON_NUM_THREADS="${SLURM_CPUS_PER_TASK:-1}"
export OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1
if [[ "$option" == --check-only ]]; then
    export CUDA_VISIBLE_DEVICES=''
    "$python" -IB "$repo/$stage/panel.py" plan --verify-assets
    echo "[codex] gate optimizer freeze and dependency checks passed $freeze $tree"
    exit 0
fi

job=${SLURM_JOB_ID:?Submit through Slurm}
args=()
case "$action" in
    produce)
        [[ -z "$option" && "${SLURM_ARRAY_TASK_ID:?Submit the fixed four-way array}" =~ ^[0-3]$ ]]
        array=${SLURM_ARRAY_JOB_ID:?Submit the fixed four-way array}
        args+=(--shard "$SLURM_ARRAY_TASK_ID")
        ;;
    analyze)
        array=${option:?Pass the producer array job ID}
        export CUDA_VISIBLE_DEVICES=''
        ;;
    *) exit 2 ;;
esac
[[ "$job" =~ ^[0-9]+$ && "$array" =~ ^[0-9]+$ ]]
snapshot=$(mktemp -d "$TMPDIR/gate-optimizer-fresh-$job.XXXXXXXX")
trap 'rm -r --one-file-system -- "$snapshot"' EXIT
git -C "$repo" archive "$source_commit" "${dependency[@]}" | tar -x -C "$snapshot"
git -C "$repo" archive "$freeze" "$stage" | tar -x -C "$snapshot"
output="$icml/runs/gate-optimizer/fresh-$array"
echo "[codex] $(date '+%Y-%m-%d %H:%M') gate optimizer fresh panel $action array=$array commit=$freeze tree=$tree source=$source_commit"
"$python" -IBu "$snapshot/$stage/panel.py" "$action" --commit "$freeze" --tree "$tree" \
    --array-id "$array" --output "$output" --scratch "$snapshot" "${args[@]}"
