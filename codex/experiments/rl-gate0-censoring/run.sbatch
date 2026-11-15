#!/usr/bin/env bash
#SBATCH --job-name=rsd-gate0-censoring
#SBATCH --partition=h200q
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gres=none
#SBATCH --cpus-per-task=1
#SBATCH --mem=4G
#SBATCH --time=00:05:00
#SBATCH --output=/work/ChihChungHsu/cactus8603/icml/tmp/rsd-gate0-censoring-%j.log
set -euo pipefail

freeze=${1:?Use the full reviewed commit}
[[ "$freeze" =~ ^[0-9a-f]{40}$ ]]
mode=${2:-}
case "$mode" in
    --check-only) [[ "$#" == 2 && -z "${SLURM_JOB_ID:-}" ]] ;;
    "") [[ "$#" == 1 && "${SLURM_JOB_ID:-}" =~ ^[1-9][0-9]*$ ]] ;;
    *) exit 2 ;;
esac

root=/work/ChihChungHsu/cactus8603
repo="$root/icml27-topic-search"
shared="$root/icml"
stage=codex/experiments/rl-gate0-censoring
topic=killtests/rl-sparse-distill
source=d52d5d737906ddc21f77269eb67d54d2ab8a56e6
python="$shared/envs/oat-vllm072/bin/python"
[[ -z "$(git -C "$repo" status --porcelain --untracked-files=all -- "$stage")" ]]
git -C "$repo" diff --quiet "$freeze" HEAD -- "$stage"

export TZ=Asia/Taipei TMPDIR="$shared/tmp" CUDA_VISIBLE_DEVICES=''
export OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1
umask 077
bundle=$(mktemp -d "$TMPDIR/rsd-gate0-censoring.XXXXXXXX")
trap 'rm -r --one-file-system -- "$bundle"' EXIT
git -C "$repo" archive "$freeze" "$stage" | tar -xf - -C "$bundle"
git -C "$repo" archive "$source" "$topic/scripts/"{aggregate,artifacts,common}.py \
    "$topic/configs/"{criteria,legacy_artifacts,pairs}.json "$topic/data/eval.jsonl" \
    "$topic/results/aggregate_gate0_A.json" | tar -xf - -C "$bundle"
mkdir -p -- "$bundle/$topic/slurm/logs"
for log in rsd-correct-26270.out rsd-vllm-26292.out rsd-vllm-26293.out; do
    cp -- "$repo/$topic/slurm/logs/$log" "$bundle/$topic/slurm/logs/$log"
done
core=$("$python" -IB -c 'import os; print(min(os.sched_getaffinity(0)))')
entry=("$bundle/$stage/analyze.py" --freeze "$freeze" --source "$bundle/$topic"
    --results "$repo/$topic/results" --grader-lib "$repo/$topic/pylib")
if [[ "$mode" == --check-only ]]; then
    timeout 25s taskset -c "$core" "$python" -IB "$bundle/$stage/test_analyze.py"
    timeout 25s taskset -c "$core" "$python" -IB "${entry[@]}" --check-only
    exit
fi
records=$(sacct -X -n -P -j 26270,26292,26293 -o JobID,State,ExitCode | sort)
[[ "$records" == $'26270|COMPLETED|0:0\n26292|COMPLETED|0:0\n26293|COMPLETED|0:0' ]]
timeout 150s taskset -c "$core" "$python" -IB "${entry[@]}" \
    --output "$shared/runs/rsd-gate0-censoring/$SLURM_JOB_ID/summary.json"
