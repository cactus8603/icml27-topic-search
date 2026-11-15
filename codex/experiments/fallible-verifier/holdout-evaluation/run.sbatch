#!/usr/bin/env bash
#SBATCH --job-name=fv-gsm8k-policy-eval
#SBATCH --partition=h200q
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gres=none
#SBATCH --cpus-per-task=2
#SBATCH --mem=12G
#SBATCH --time=00:20:00
#SBATCH --output=/work/ChihChungHsu/cactus8603/icml/tmp/fv-gsm8k-policy-eval-%j.log
set -euo pipefail

freeze=${1:?Pass the common full scoring and evaluation freeze}
[[ "$freeze" =~ ^[0-9a-f]{40}$ ]]
if [[ ${2:-} == --check-only ]]; then
    [[ "$#" == 2 ]]
    arguments=(--check-only)
else
    [[ "$#" == 3 && "$2" =~ ^[1-9][0-9]*$ && "$3" =~ ^[1-9][0-9]*$ && "$2" != "$3" ]]
    [[ "${SLURM_JOB_ID:?Submit through Slurm}" =~ ^[1-9][0-9]*$ ]]
    arguments=(--freeze "$freeze" --train-job "$2" --test-job "$3")
fi

root=/work/ChihChungHsu/cactus8603
repo="$root/icml27-topic-search"
shared="$root/icml"
topic=codex/experiments/fallible-verifier
stage="$topic/holdout-evaluation"
inputs=("$stage" "$topic/holdout-scoring" "$topic/real_holdout/prepare.py" "$topic/real_holdout/dev_likelihood.py"
        "$topic/"{phase3_likelihood,phase3_feasibility,phase3_synthetic}.py "$topic/round2/"{config.json,protocol.py,analysis.py})
[[ -z "$(git -C "$repo" status --porcelain --untracked-files=all -- "${inputs[@]}")" ]]
git -C "$repo" diff --quiet "$freeze" HEAD -- "${inputs[@]}"

export TZ=Asia/Taipei PYTHONUNBUFFERED=1 CUDA_VISIBLE_DEVICES=''
export TMPDIR="$shared/tmp" HF_HOME="$shared/hf_home"
export HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1
umask 077
bundle=$(mktemp -d "$TMPDIR/fv-gsm8k-policy-eval-${SLURM_JOB_ID:-check}.XXXXXXXX")
trap 'rm -r --one-file-system -- "$bundle"' EXIT
export XDG_CACHE_HOME="$bundle/xdg"
git -C "$repo" archive "$freeze" "${inputs[@]}" | tar -x -C "$bundle"
python="$shared/envs/t2irare/bin/python"
if [[ ${2:-} == --check-only ]]; then
    "$python" -IB "$bundle/$stage/test_evaluation.py"
    "$python" -IB "$bundle/$stage/test_run_evaluation.py"
fi
"$python" -IB "$bundle/$stage/run_evaluation.py" "${arguments[@]}"
