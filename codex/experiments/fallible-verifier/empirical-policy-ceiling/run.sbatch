#!/usr/bin/env bash
#SBATCH --job-name=fv-gsm8k-policy-ceiling
#SBATCH --partition=h200q
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gres=none
#SBATCH --cpus-per-task=1
#SBATCH --mem=4G
#SBATCH --time=00:05:00
#SBATCH --output=/work/ChihChungHsu/cactus8603/icml/tmp/fv-gsm8k-policy-ceiling-%j.log
set -euo pipefail

freeze=${1:?Pass the full ceiling freeze commit}
[[ "$freeze" =~ ^[0-9a-f]{40}$ ]]
case ${2:-} in
    --check-only|--local) [[ "$#" == 2 && -z "${SLURM_JOB_ID:-}" ]] ;;
    "") [[ "$#" == 1 && "${SLURM_JOB_ID:?Submit through Slurm}" =~ ^[1-9][0-9]*$ ]] ;;
    *) exit 2 ;;
esac

root=/work/ChihChungHsu/cactus8603
repo="$root/icml27-topic-search"
shared="$root/icml"
topic=codex/experiments/fallible-verifier
stage="$topic/empirical-policy-ceiling"
source_freeze=ac3950820be8c7f4f1528ecc553b9cf51aad894b
inputs=("$topic/holdout-evaluation" "$topic/holdout-scoring" "$topic/real_holdout/prepare.py" "$topic/real_holdout/dev_likelihood.py"
        "$topic/"{phase3_likelihood,phase3_feasibility,phase3_synthetic}.py "$topic/round2/"{config.json,protocol.py,analysis.py})
[[ -z "$(git -C "$repo" status --porcelain --untracked-files=all -- "$stage")" ]]
git -C "$repo" diff --quiet "$freeze" HEAD -- "$stage"

export TZ=Asia/Taipei PYTHONUNBUFFERED=1 CUDA_VISIBLE_DEVICES=''
export TMPDIR="$shared/tmp" HF_HOME="$shared/hf_home"
export HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1
export MKL_NUM_THREADS=1 OMP_THREAD_LIMIT=1
umask 077
bundle=$(mktemp -d "$TMPDIR/fv-gsm8k-policy-ceiling-${SLURM_JOB_ID:-check}.XXXXXXXX")
trap 'rm -r --one-file-system -- "$bundle"' EXIT
export XDG_CACHE_HOME="$bundle/xdg"
git -C "$repo" archive "$freeze" "$stage" | tar -x -C "$bundle"
git -C "$repo" archive "$source_freeze" "${inputs[@]}" | tar -x -C "$bundle"
python="$shared/envs/t2irare/bin/python"
case ${2:-} in
    --check-only)
        timeout --kill-after=5s 120s "$python" -IB "$bundle/$stage/test_ceiling.py"
        timeout --kill-after=5s 120s "$python" -IB "$bundle/$stage/ceiling.py" --freeze "$freeze" --check-only
        ;;
    --local)
        core=$("$python" -IB -c 'import os; print(min(os.sched_getaffinity(0)))')
        timeout --kill-after=1s 29s taskset -c "$core" "$python" -IB "$bundle/$stage/ceiling.py" \
            --freeze "$freeze" --local-run-id "local-${bundle##*.}"
        ;;
    "")
        timeout --kill-after=5s 120s "$python" -IB "$bundle/$stage/ceiling.py" --freeze "$freeze"
        ;;
esac
