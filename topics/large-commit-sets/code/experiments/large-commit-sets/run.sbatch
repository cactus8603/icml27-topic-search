#!/bin/bash -l
#SBATCH --job-name=codex-commit-sets
#SBATCH --partition=h200q
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=48G
#SBATCH --time=00:30:00

set -euo pipefail
mode=${1:?Pass smoke or main}
repo=/work/ChihChungHsu/cactus8603/icml27-topic-search
shared=/work/ChihChungHsu/cactus8603/icml
topic="$repo/codex/experiments/large-commit-sets"
revision=08b83a6feb34df1a6011b80c3c00c7563e963b07
model="${LLADA_MODEL_PATH:-$shared/hf_home/hub/models--GSAI-ML--LLaDA-8B-Instruct/snapshots/$revision}"
python="$shared/envs/t2irare/bin/python"
parquet_python="$shared/envs/biasgen/bin/python"
export TMPDIR="$shared/tmp" HF_HOME="$shared/hf_home"
export HF_MODULES_CACHE="$shared/tmp/large_commit_sets_hf_modules_${SLURM_JOB_ID:?Use sbatch}"
export HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 TOKENIZERS_PARALLELISM=false
export PYTHONUNBUFFERED=1 PYTHONDONTWRITEBYTECODE=1
export TZ=Asia/Taipei

extra=()
case "$mode" in
  smoke)
    data="$shared/data/gsm8k-hf/main/train-00000-of-00001.parquet"
    ;;
  main)
    frozen_commit=${2:?Pass the frozen commit hash as the second argument for main}
    if [[ -n "$(git -C "$repo" status --porcelain -- "$topic")" ]]; then
      printf '%s\n' 'Freeze and commit the experiment directory before submitting main.' >&2
      exit 1
    fi
    git -C "$repo" cat-file -e "$frozen_commit:codex/experiments/large-commit-sets/PREREG.md"
    git -C "$repo" diff --quiet "$frozen_commit" HEAD -- "$topic"
    export PILOT_FREEZE_COMMIT="$(git -C "$repo" rev-parse "$frozen_commit^{commit}")"
    data="$shared/data/gsm8k/main/test-00000-of-00001.parquet"
    extra=(--output "$topic/results/main-${SLURM_JOB_ID}.jsonl")
    ;;
  *)
    printf '%s\n' 'Pass smoke or main.' >&2
    exit 2
    ;;
esac
printf '[codex] %s %s job=%s commit=%s\n' "$(date '+%F %R')" "$mode" "$SLURM_JOB_ID" "$(git -C "$repo" rev-parse HEAD)"
exec "$python" -B "$topic/pilot.py" --mode "$mode" --model "$model" --data "$data" \
  --parquet-python "$parquet_python" "${extra[@]}"
