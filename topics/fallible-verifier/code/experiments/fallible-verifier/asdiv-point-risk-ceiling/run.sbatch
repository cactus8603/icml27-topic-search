#!/usr/bin/env bash
#SBATCH --job-name=fv-asdiv-point-risk
#SBATCH --partition=h200q
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gres=none
#SBATCH --cpus-per-task=1
#SBATCH --mem=4G
#SBATCH --time=00:05:00
#SBATCH --output=/work/ChihChungHsu/cactus8603/icml/tmp/fv-asdiv-point-risk-%j.log
set -euo pipefail

freeze=${1:?Pass the reviewed ASDiv-A point-risk commit}
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
topic=codex/experiments/fallible-verifier
stage="$topic/asdiv-point-risk-ceiling"
source_job=26707
asdiv_freeze=885ac701b9b4ed313c3ffb7d6619b70953f0422f
lp_freeze=454e3d7b2bb4aeec07027d9467efd5c2afcc569f
gsm1k_freeze=229cdcf22a60e3f030e1e1edf9347294e9584b02
scoring_freeze=ac3950820be8c7f4f1528ecc553b9cf51aad894b
[[ -z "$(git -C "$repo" status --porcelain --untracked-files=all -- "$stage")" ]]
git -C "$repo" diff --quiet "$freeze" HEAD -- "$stage"

export TZ=Asia/Taipei PYTHONUNBUFFERED=1 CUDA_VISIBLE_DEVICES=''
export TMPDIR="$shared/tmp" HF_HOME="$shared/hf_home"
export HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1
export MKL_NUM_THREADS=1 OMP_THREAD_LIMIT=1
umask 077
bundle=$(mktemp -d "$TMPDIR/fv-asdiv-point-risk.XXXXXXXX")
trap 'rm -r --one-file-system -- "$bundle"' EXIT
export XDG_CACHE_HOME="$bundle/xdg"
git -C "$repo" archive "$freeze" "$stage" | tar -x -C "$bundle"
git -C "$repo" archive "$asdiv_freeze" "$topic/asdiv-replication/"{assay,prepare}.py | tar -x -C "$bundle"
git -C "$repo" archive "$lp_freeze" "$topic/empirical-policy-ceiling/ceiling.py" | tar -x -C "$bundle"
git -C "$repo" archive "$gsm1k_freeze" "$topic/gsm1k-replication/"{panel,score,evaluate}.py | tar -x -C "$bundle"
git -C "$repo" archive "$scoring_freeze" "$topic/holdout-evaluation" "$topic/holdout-scoring" \
    "$topic/real_holdout/"{prepare,dev_likelihood}.py "$topic/"{phase3_likelihood,phase3_feasibility,phase3_synthetic}.py \
    "$topic/round2/"{config.json,protocol.py,analysis.py} | tar -x -C "$bundle"
python="$shared/envs/t2irare/bin/python"
core=$("$python" -IB -c 'import os; print(min(os.sched_getaffinity(0)))')
entry="$bundle/$stage"
if [[ "$mode" != --check-only ]]; then
    [[ "$(sacct -X -n -P -j "$source_job" -o JobID,State,ExitCode,JobName)" == "$source_job|COMPLETED|0:0|fv-asdiv-replication" ]]
fi
case "$mode" in
    --check-only)
        timeout --kill-after=1s 29s taskset -c "$core" "$python" -IB "$entry/test_point_envelope.py"
        timeout --kill-after=1s 29s taskset -c "$core" "$python" -IB "$entry/point_envelope.py" \
            --freeze "$freeze" --check-only
        ;;
    --local)
        timeout --kill-after=1s 29s taskset -c "$core" "$python" -IB "$entry/point_envelope.py" \
            --freeze "$freeze" --local-run-id "local-${bundle##*.}"
        ;;
    "")
        timeout --kill-after=5s 120s taskset -c "$core" "$python" -IB "$entry/point_envelope.py" \
            --freeze "$freeze" --job "$SLURM_JOB_ID"
        ;;
esac
