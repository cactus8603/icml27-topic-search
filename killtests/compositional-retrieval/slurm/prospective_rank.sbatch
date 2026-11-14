#!/usr/bin/env bash
#SBATCH --job-name=cr-prospective
#SBATCH --partition=h200q
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=48G
#SBATCH --time=01:00:00
#SBATCH --output=/work/ChihChungHsu/cactus8603/icml27-topic-search/killtests/compositional-retrieval/logs/%x-%j.out
set -euo pipefail

if [[ $# != 1 || ! $1 =~ ^[0-9a-f]{40}$ ]]; then
    printf 'Usage: sbatch slurm/prospective_rank.sbatch FROZEN_COMMIT\n' >&2
    exit 2
fi
commit=$1
job_id=${SLURM_JOB_ID:?Submit through Slurm}
topic=killtests/compositional-retrieval
submit_dir=$(cd -- "${SLURM_SUBMIT_DIR:?}" && pwd -P)
repo=$(git -C "$submit_dir" rev-parse --show-toplevel)
if [[ $submit_dir != "$repo/$topic" ]]; then
    printf 'Submit from %s.\n' "$topic" >&2
    exit 2
fi

work=/work/ChihChungHsu/cactus8603/icml
export HF_HOME="$work/hf_home" TMPDIR="$work/tmp" HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1
export OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1
python="$work/envs/vllm033/bin/python"
output_parent="$submit_dir/results/prospective_rank"
output="$output_parent/$job_id"
source="$output/_source"

mkdir -p -- "$output_parent"
mkdir -- "$output"
mkdir -- "$source"
git -C "$repo" archive "$commit" "$topic/configs" "$topic/scripts" \
    "$topic/PREREG_prospective_rank.md" "$topic/slurm/prospective_rank.sbatch" | tar -xf - -C "$source"
export EXP="$source/$topic"
printf 'job=%s frozen_commit=%s start=%s\n' "$job_id" "$commit" "$(TZ=Asia/Taipei date -Is)"
printf '%s\n' "$commit" > "$output/frozen_commit.txt"
cd -- "$EXP"
exec "$python" -IB scripts/prospective_rank.py --out-dir "$output"
