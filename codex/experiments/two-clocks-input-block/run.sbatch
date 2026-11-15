#!/usr/bin/env bash
#SBATCH --job-name=bg-input-block
#SBATCH --partition=h200q
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gres=none
#SBATCH --cpus-per-task=1
#SBATCH --mem=4G
#SBATCH --time=00:05:00
#SBATCH --output=/work/ChihChungHsu/cactus8603/icml/tmp/bg-input-block-%j.log
set -euo pipefail

freeze=${1:?Pass the full reviewed commit}
[[ "$freeze" =~ ^[0-9a-f]{40}$ ]]
mode=${2:-}
case "$mode" in
    --check-only|--local) [[ "$#" == 2 && -z "${SLURM_JOB_ID:-}" ]] ;;
    "") [[ "$#" == 1 && "${SLURM_JOB_ID:-}" =~ ^[1-9][0-9]*$ ]] ;;
    *) exit 2 ;;
esac

root=/work/ChihChungHsu/cactus8603
repo="$root/icml27-topic-search"
shared="$root/icml"
stage=codex/experiments/two-clocks-input-block
topic=killtests/biased-gen-cfg
source_freeze=9e92804a7bfded3c5cafd4fccd4315df2034eca5
source="$repo/$topic/results/week2-9e92804"
[[ -z "$(git -C "$repo" status --porcelain --untracked-files=all -- "$stage")" ]]
git -C "$repo" diff --quiet "$freeze" HEAD -- "$stage"

export TZ=Asia/Taipei PYTHONUNBUFFERED=1 CUDA_VISIBLE_DEVICES=''
export TMPDIR="$shared/tmp" OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1
umask 077
bundle=$(mktemp -d "$TMPDIR/bg-input-block.XXXXXXXX")
trap 'rm -r --one-file-system -- "$bundle"' EXIT
git -C "$repo" archive "$freeze" "$stage" | tar -x -C "$bundle"
git -C "$repo" archive "$source_freeze" "$topic/scripts/"{reach.py,schedule.py,week2_identity.py} \
    "$topic/configs/"{criteria_week2.json,t_mask_hier.json} | tar -x -C "$bundle"

python="$shared/envs/biasgen/bin/python"
entry="$bundle/$stage"
export PYTHONPATH="$entry:$bundle/$topic/scripts" PYTHONNOUSERSITE=1
core=$("$python" -BP -c 'import os; print(min(os.sched_getaffinity(0)))')
if [[ "$mode" == --check-only ]]; then
    timeout --kill-after=1s 29s taskset -c "$core" "$python" -BP "$entry/test_block_gap.py"
    timeout --kill-after=1s 29s taskset -c "$core" "$python" -BP "$entry/block_gap.py" \
        "$source" --freeze "$freeze" --check-only
    exit 0
fi

actual=$(sacct -X -n -P -j 26383 -o JobID,State,ExitCode,JobName)
[[ "$actual" == '26383|COMPLETED|0:0|bg-w2-eval' ]]
awk '$1=="[codex]" && $4=="submit" && $5=="26383" && $6=="biased-gen-cfg" && $NF=="9e92804" { count++ } END { exit count != 1 }' \
    "$repo/killtests/SUBMITTED.log"

if [[ "$mode" == --local ]]; then
    run_id="local-${bundle##*.}"
    time_limit=29s
else
    run_id=$SLURM_JOB_ID
    time_limit=120s
fi
timeout --kill-after=5s "$time_limit" taskset -c "$core" "$python" -BP "$entry/block_gap.py" \
    "$source" --freeze "$freeze" --output "$shared/runs/two-clocks-input-block/$run_id/results.json"
