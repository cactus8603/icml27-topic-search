#!/usr/bin/env bash
#SBATCH --job-name=fv-gsm8k-nested-audit
#SBATCH --partition=h200q
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gres=none
#SBATCH --cpus-per-task=1
#SBATCH --mem=4G
#SBATCH --time=00:05:00
#SBATCH --output=/work/ChihChungHsu/cactus8603/icml/tmp/fv-gsm8k-nested-audit-%j.log
set -euo pipefail

freeze=${1:?Pass the reviewed experimental commit}
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
stage="$topic/gsm8k-nested-audit"
source_freeze=ac3950820be8c7f4f1528ecc553b9cf51aad894b
ceiling_freeze=454e3d7b2bb4aeec07027d9467efd5c2afcc569f
inputs=("$topic/holdout-evaluation" "$topic/holdout-scoring" "$topic/real_holdout/"{prepare,dev_likelihood}.py
        "$topic/"{phase3_likelihood,phase3_feasibility,phase3_synthetic}.py "$topic/round2/"{config.json,protocol.py,analysis.py})

export TZ=Asia/Taipei PYTHONUNBUFFERED=1 CUDA_VISIBLE_DEVICES=''
export TMPDIR="$shared/tmp" HF_HOME="$shared/hf_home"
export HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1
export MKL_NUM_THREADS=1 OMP_THREAD_LIMIT=1
umask 077
bundle=$(mktemp -d "$TMPDIR/fv-gsm8k-nested-audit.XXXXXXXX")
trap 'rm -r --one-file-system -- "$bundle"' EXIT
export XDG_CACHE_HOME="$bundle/xdg"
git -C "$repo" archive "$freeze" "$stage" | tar -x -C "$bundle"
git -C "$repo" archive "$source_freeze" "${inputs[@]}" | tar -x -C "$bundle"
git -C "$repo" archive "$ceiling_freeze" "$topic/empirical-policy-ceiling/ceiling.py" | tar -x -C "$bundle"
python="$shared/envs/t2irare/bin/python"
core=$("$python" -IB -c 'import os; print(min(os.sched_getaffinity(0)))')
entry="$bundle/$stage"
case "$mode" in
    --check-only)
        timeout --kill-after=1s 29s taskset -c "$core" "$python" -IB "$entry/test_nested_audit.py"
        timeout --kill-after=1s 29s taskset -c "$core" "$python" -IB "$entry/nested_audit.py" \
            --freeze "$freeze" --check-only
        ;;
    --local)
        timeout --kill-after=1s 29s taskset -c "$core" "$python" -IB "$entry/nested_audit.py" \
            --freeze "$freeze" --local-run-id "local-${bundle##*.}"
        ;;
    "")
        timeout --kill-after=5s 120s taskset -c "$core" "$python" -IB "$entry/nested_audit.py" \
            --freeze "$freeze" --job "$SLURM_JOB_ID"
        ;;
esac
