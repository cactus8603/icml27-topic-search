#!/usr/bin/env bash
#SBATCH --job-name=gate-nominal-dense
#SBATCH --partition=h200q
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=32G
#SBATCH --time=01:00:00
#SBATCH --array=0-7%1
#SBATCH --output=/work/ChihChungHsu/cactus8603/icml/tmp/gate-nominal-dense-%A_%a.log
set -euo pipefail

root=/work/ChihChungHsu/cactus8603
repo="$root/icml27-topic-search"
shared="$root/icml"
topic=codex/experiments/gate-optimizer
nominal="$topic-nominal"
stage="$topic-nominal-dense"
source_commit=6e16f498bb21044128f8383a684874f3ea12e29c
nominal_commit=d89b15761fbf6be47bb7098d3a088061d76751ab
freeze=${1:?Pass the reviewed full commit}
mode=${2:-produce}
[[ "$freeze" =~ ^[0-9a-f]{40}$ ]]
[[ -z "$(git -C "$repo" status --porcelain --untracked-files=all -- "$stage")" ]]
git -C "$repo" diff --quiet "$freeze" HEAD -- "$stage"

job=${SLURM_JOB_ID:-check}
case "$mode" in
    produce)
        [[ "$#" == 1 && "${SLURM_ARRAY_TASK_ID:-}" =~ ^[0-7]$ ]]
        array=${SLURM_ARRAY_JOB_ID:?Submit the fixed eight-way array}
        options=(--shard "$SLURM_ARRAY_TASK_ID")
        ;;
    analyze)
        [[ "$#" == 3 && -z "${SLURM_ARRAY_TASK_ID:-}" ]]
        array=$3
        options=()
        export CUDA_VISIBLE_DEVICES=''
        ;;
    check)
        [[ "$#" == 2 && "$job" == check ]]
        options=()
        export CUDA_VISIBLE_DEVICES=''
        ;;
    *) exit 2 ;;
esac
if [[ "$mode" != check ]]; then
    [[ "$job" =~ ^[1-9][0-9]*$ && "$array" =~ ^[1-9][0-9]*$ ]]
    options+=(--array "$array" --job "$job" --output "$shared/runs/gate-optimizer/nominal-dense-$array")
fi

export TZ=Asia/Taipei TMPDIR="$shared/tmp" HF_HOME="$shared/hf_home"
export HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1
export RAYON_NUM_THREADS=${SLURM_CPUS_PER_TASK:-1}
umask 077
scratch=$(mktemp -d "$TMPDIR/gate-nominal-dense-$job.XXXXXXXX")
trap 'rm -r --one-file-system -- "$scratch"' EXIT
git -C "$repo" archive "$source_commit" "$topic" | tar -x -C "$scratch"
git -C "$repo" archive "$nominal_commit" "$nominal/measure.py" | tar -x -C "$scratch"
git -C "$repo" archive "$freeze" "$stage" | tar -x -C "$scratch"
printf '[codex] %s nominal dense %s job=%s freeze=%s\n' "$(date '+%F %R')" "$mode" "$job" "$freeze"
python="$shared/envs/vllm033/bin/python"
if [[ "$mode" == check ]]; then
    "$python" -IB "$scratch/$stage/test_dense.py"
fi
"$python" -IBu "$scratch/$stage/dense.py" "$mode" \
    --scratch "$scratch" --freeze "$freeze" "${options[@]}"
