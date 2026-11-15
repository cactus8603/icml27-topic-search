#!/usr/bin/env bash
#SBATCH --job-name=cr-controlled-spatial
#SBATCH --partition=h200q
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=24G
#SBATCH --time=00:30:00
#SBATCH --output=/work/ChihChungHsu/cactus8603/icml27-topic-search/killtests/compositional-retrieval/logs/%x-%j.out
set -euo pipefail

if [[ $# != 1 || ! $1 =~ ^[0-9a-f]{40}$ ]]; then
    printf 'Usage: sbatch slurm/controlled_spatial.sbatch FROZEN_COMMIT\n' >&2
    exit 2
fi
commit=$1
job_id=${SLURM_JOB_ID:?Submit through Slurm}
topic=killtests/compositional-retrieval
submit_dir=$(cd -- "${SLURM_SUBMIT_DIR:?}" && pwd -P)
repo=$(git -C "$submit_dir" rev-parse --show-toplevel)
if [[ $submit_dir != "$repo/$topic" || ! $job_id =~ ^[0-9]+$ ]]; then
    printf 'Submit from %s.\n' "$topic" >&2
    exit 2
fi

work=/work/ChihChungHsu/cactus8603/icml
export HF_HOME="$work/hf_home" TMPDIR="$work/tmp" HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1
export OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1
python="$work/envs/vllm033/bin/python"
source_dir=$(mktemp -d -- "$TMPDIR/controlled-spatial.$job_id.XXXXXXXX")
trap 'rm -rf -- "$source_dir"' EXIT
git -C "$repo" archive "$commit" "$topic/configs" "$topic/scripts" \
    "$topic/PREREG_controlled_spatial.md" "$topic/slurm/controlled_spatial.sbatch" | tar -xf - -C "$source_dir"
export EXP="$source_dir/$topic"

output_parent="$submit_dir/results/controlled_spatial"
output="$output_parent/$job_id"
mkdir -p -- "$output_parent"
mkdir -- "$output"
printf '%s\n' "$commit" > "$output/frozen_commit.txt"
printf 'job=%s frozen_commit=%s start=%s\n' "$job_id" "$commit" "$(TZ=Asia/Taipei date -Is)"
cd -- "$EXP"
"$python" -IB scripts/controlled_spatial.py --out-dir "$output"
