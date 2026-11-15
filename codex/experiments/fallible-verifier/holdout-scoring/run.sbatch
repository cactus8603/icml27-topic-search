#!/usr/bin/env bash
#SBATCH --job-name=fv-gsm8k-policy-score
#SBATCH --partition=h200q
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=48G
#SBATCH --time=01:30:00
#SBATCH --output=/work/ChihChungHsu/cactus8603/icml/tmp/fv-gsm8k-policy-score-%j.log
set -euo pipefail

freeze=${1:?Pass the full frozen commit}
split=${2:?Pass --check-only, calibrate-dev, train or test}
[[ "$#" == 2 && "$freeze" =~ ^[0-9a-f]{40}$ ]]
root=/work/ChihChungHsu/cactus8603
repo="$root/icml27-topic-search"
shared="$root/icml"
topic=codex/experiments/fallible-verifier
stage="$topic/holdout-scoring"
files=("$stage" "$topic/holdout-evaluation" "$topic/real_holdout/prepare.py" "$topic/real_holdout/dev_likelihood.py"
       "$topic/"{phase3_likelihood,phase3_feasibility,phase3_synthetic}.py "$topic/round2/"{config.json,protocol.py,analysis.py})
[[ -z "$(git -C "$repo" status --porcelain --untracked-files=all -- "${files[@]}")" ]]
git -C "$repo" diff --quiet "$freeze" HEAD -- "${files[@]}"

export TZ=Asia/Taipei PYTHONDONTWRITEBYTECODE=1 PYTHONUNBUFFERED=1
export TMPDIR="$shared/tmp" HF_HOME="$shared/hf_home"
export HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 TOKENIZERS_PARALLELISM=false
export OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 VLLM_NO_USAGE_STATS=1
export VLLM_USE_FLASHINFER_SAMPLER=0 TORCH_COMPILE_DISABLE=1
umask 077
bundle=$(mktemp -d "$TMPDIR/fv-gsm8k-policy-score-${SLURM_JOB_ID:-check}.XXXXXXXX")
trap 'rm -r --one-file-system -- "$bundle"' EXIT
export XDG_CACHE_HOME="$bundle/xdg"
git -C "$repo" archive "$freeze" "${files[@]}" | tar -x -C "$bundle"
python="$shared/envs/vllm033/bin/python"
entry="$bundle/$stage/scoring.py"
if [[ "$split" == --check-only ]]; then
    export CUDA_VISIBLE_DEVICES=''
    for role in weak strong; do
        "$python" -IB "$entry" --role "$role" --check-only
    done
    exit 0
fi

[[ "$split" == calibrate-dev || "$split" == train || "$split" == test ]]
[[ "${SLURM_JOB_ID:?Submit through Slurm}" =~ ^[1-9][0-9]*$ ]]
if [[ "$split" == calibrate-dev ]]; then
    arguments=(--calibrate-dev)
    receipt_prefix="$shared/runs/fallible-verifier/gsm8k-policy-calibration/$SLURM_JOB_ID/"
else
    arguments=(--split "$split")
    receipt_prefix="$shared/runs/fallible-verifier/gsm8k-policy-scoring/$SLURM_JOB_ID/$split-"
    for filename in measurement.json decision.json; do
        git -C "$repo" cat-file -e "$freeze:$stage/$filename"
    done
fi
hash_args=()
for role in weak strong; do
    if FV_REPORT_STAGE_FD3=1 "$python" -IB "$entry" --role "$role" "${arguments[@]}" --freeze "$freeze" 3>&2 >/dev/null 2>&1; then
        :
    else
        status=$?
        printf '[codex] %s %s %s scoring failed\n' "$(date '+%Y-%m-%d %H:%M')" "$split" "$role" >&2
        exit "$status"
    fi
    receipt="$receipt_prefix$role.json"
    [[ -f "$receipt" && ! -L "$receipt" ]]
    receipt_sha=$(sha256sum "$receipt" | cut -d' ' -f1)
    printf '[codex] %s %s %s receipt SHA-256 %s\n' "$(date '+%Y-%m-%d %H:%M')" "$split" "$role" "$receipt_sha"
    hash_args+=("--$role-sha256" "$receipt_sha")
done
if [[ "$split" == calibrate-dev ]]; then
    CUDA_VISIBLE_DEVICES='' "$shared/envs/t2irare/bin/python" -IB "$bundle/$stage/calibration.py" --freeze "$freeze" "${hash_args[@]}"
fi
