#!/usr/bin/env bash
#SBATCH --job-name=commit-sets-observable
#SBATCH --partition=h200q
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gres=none
#SBATCH --cpus-per-task=1
#SBATCH --mem=4G
#SBATCH --time=00:05:00
#SBATCH --output=/work/ChihChungHsu/cactus8603/icml/tmp/commit-sets-observable-%j.log
set -euo pipefail

freeze=${1:?Pass the reviewed observable-selector commit}
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
topic=codex/experiments/large-commit-sets
stage="$topic/observable-selector"
source_freeze=c979eba0260a2a6f5bd3a1de5cf6affb4ea7a5af
[[ -z "$(git -C "$repo" status --porcelain --untracked-files=all -- "$stage")" ]]
git -C "$repo" diff --quiet "$freeze" HEAD -- "$stage"

export PILOT_FREEZE_COMMIT="$source_freeze" PILOT_RESULTS_PATH="$repo/$topic/round2/results"
export TZ=Asia/Taipei PYTHONUNBUFFERED=1 CUDA_VISIBLE_DEVICES=''
export TMPDIR="$shared/tmp" HF_HOME="$shared/hf_home"
export HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1
export MKL_NUM_THREADS=1 OMP_THREAD_LIMIT=1
umask 077
bundle=$(mktemp -d "$TMPDIR/commit-sets-observable.XXXXXXXX")
trap 'rm -r --one-file-system -- "$bundle"' EXIT
export XDG_CACHE_HOME="$bundle/xdg"
git -C "$repo" archive "$freeze" "$stage" | tar -x -C "$bundle"
git -C "$repo" archive "$source_freeze" "$topic/round2/"{score.py,design.py,config.json,artifacts.json} \
    "$topic/"{pilot.py,judge.py,read_parquet.py,thresholds.json} | tar -x -C "$bundle"
python="$shared/envs/t2irare/bin/python"
core=$("$python" -IB -c 'import os; print(min(os.sched_getaffinity(0)))')
entry="$bundle/$stage"
if [[ "$mode" != --check-only ]]; then
    actual=$(sacct -X -n -P -j 26441,26442,26443 -o JobID,State,ExitCode,JobName | sort)
    expected=$'26441|COMPLETED|0:0|commit-sets-r2\n26442|COMPLETED|0:0|commit-sets-r2\n26443|COMPLETED|0:0|commit-sets-r2-score'
    [[ "$actual" == "$expected" ]]
fi
case "$mode" in
    --check-only)
        timeout --kill-after=1s 29s taskset -c "$core" "$python" -IB "$entry/test_observable.py"
        timeout --kill-after=1s 29s taskset -c "$core" "$python" -IB "$entry/observable.py" \
            --freeze "$freeze" --check-only
        ;;
    --local)
        timeout --kill-after=1s 29s taskset -c "$core" "$python" -IB "$entry/observable.py" \
            --freeze "$freeze" --local-run-id "local-${bundle##*.}"
        ;;
    "")
        timeout --kill-after=5s 120s taskset -c "$core" "$python" -IB "$entry/observable.py" \
            --freeze "$freeze" --job "$SLURM_JOB_ID"
        ;;
esac
