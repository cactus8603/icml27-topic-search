#!/usr/bin/env bash
#SBATCH --job-name=gate-nominal
#SBATCH --partition=h200q
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=32G
#SBATCH --time=01:30:00
#SBATCH --output=/work/ChihChungHsu/cactus8603/icml/tmp/gate-nominal-%j.log
set -euo pipefail

work=/work/ChihChungHsu/cactus8603
repo="$work/icml27-topic-search"
shared="$work/icml"
stage=codex/experiments/gate-optimizer-nominal
source=codex/experiments/gate-optimizer
reference=6e16f498bb21044128f8383a684874f3ea12e29c
freeze=${1:?Pass the reviewed full commit}
mode=${2:-}
[[ "$freeze" =~ ^[0-9a-f]{40}$ && $# -le 2 && ( -z "$mode" || "$mode" == --check-only ) ]]
git -C "$repo" diff --quiet "$freeze" HEAD -- "$stage"
[[ -z "$(git -C "$repo" status --porcelain --untracked-files=all -- "$stage")" ]]
export TZ=Asia/Taipei TMPDIR="$shared/tmp" HF_HOME="$shared/hf_home" HF_HUB_OFFLINE=1
export TRANSFORMERS_OFFLINE=1 OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1
export RAYON_NUM_THREADS=${SLURM_CPUS_PER_TASK:-1}
job=${SLURM_JOB_ID:-check}
scratch=$(mktemp -d "$TMPDIR/gate-nominal-$job.XXXXXXXX")
trap 'rm -r --one-file-system -- "$scratch"' EXIT
git -C "$repo" archive "$reference" "$source" | tar -x -C "$scratch"
git -C "$repo" archive "$freeze" "$stage" | tar -x -C "$scratch"
python="$shared/envs/vllm033/bin/python"
args=(--scratch "$scratch" --freeze "$freeze")
if [[ "$mode" == --check-only ]]; then
    export CUDA_VISIBLE_DEVICES=''
    "$python" -IB "$scratch/$source/round3/panel.py" plan --verify-assets >/dev/null
    args+=(--smoke)
else
    [[ "$job" =~ ^[1-9][0-9]*$ ]]
    output="$shared/runs/gate-optimizer/nominal-$job"
    mkdir -p -- "$(dirname "$output")"
    mkdir -- "$output"
    args+=(--output "$output/summary.json")
fi
"$python" -IBu "$scratch/$stage/measure.py" "${args[@]}"
