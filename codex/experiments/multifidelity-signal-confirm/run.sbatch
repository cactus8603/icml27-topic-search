#!/usr/bin/env bash
#SBATCH --job-name=mf-signal-confirm
#SBATCH --partition=h200q
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=96G
#SBATCH --time=02:00:00
#SBATCH --output=/work/ChihChungHsu/cactus8603/icml/tmp/mf-signal-confirm-%j.log
set -euo pipefail

freeze=${1:?Pass the frozen diagnostic commit}
[[ "$freeze" =~ ^[0-9a-f]{40}$ ]]
[[ "$#" == 1 || "$#" == 2 && "$2" == --check-only ]]
root=/work/ChihChungHsu/cactus8603
repo="$root/icml27-topic-search"
shared="$root/icml"
stage=codex/experiments/multifidelity-signal-confirm
source=killtests/multifidelity-attribution
source_commit=599469e3faf5dec7aae854cc929c6d265bdd706f
data_manifest=codex/experiments/multifidelity-second-seed/cifar.sha256
export TMPDIR="$shared/tmp" TZ=Asia/Taipei PYTHONDONTWRITEBYTECODE=1 PYTHONUNBUFFERED=1
export CIFAR_DIR="$shared/data/cifar10/cifar-10-batches-py"
export OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1
[[ -z "$(git -C "$repo" status --porcelain --untracked-files=all -- "$stage" "$source")" ]]
git -C "$repo" diff --quiet "$freeze" HEAD -- "$stage"
bundle=$(mktemp -d "$TMPDIR/mf-signal-confirm-${SLURM_JOB_ID:-check}.XXXXXXXX")
trap 'rm -r --one-file-system -- "$bundle"' EXIT
git -C "$repo" archive "$freeze" "$stage" "$data_manifest" | tar -x -C "$bundle"
git -C "$repo" archive "$source_commit" "$source/configs/"{stage0,criteria}.json \
    "$source/scripts/"{common,exact_y,jax_common,mf_core,hybrid,stage0_criteria}.py | tar -x -C "$bundle"
cd "$CIFAR_DIR"
sha256sum --check "$bundle/$data_manifest"
score="$bundle/$stage/score.py"
cpu="$shared/envs/attrib-torch/bin/python"
jax="$shared/envs/attrib-jax/bin/python"
CUDA_VISIBLE_DEVICES='' "$cpu" -IB "$score" preflight
if [[ "${2:-}" == --check-only ]]; then
    exit 0
fi
job=${SLURM_JOB_ID:?Submit through Slurm}
[[ "$job" =~ ^[1-9][0-9]*$ ]]
run="$shared/runs/multifid/signal-confirm/$job"
mkdir -p "$(dirname "$run")"
mkdir "$run"
CUDA_VISIBLE_DEVICES='' "$cpu" -IB "$score" configure --run "$run"
export REQUIRE_GPU=1
exact="$bundle/$source/scripts/exact_y.py"
"$jax" -IB "$exact" --cfg "$run/config.json" --qstart 0 --qend 10
CUDA_VISIBLE_DEVICES='' "$cpu" -IB "$score" replay --run "$run"
"$jax" -IB "$exact" --cfg "$run/config.json" --qstart 50 --qend 100
CUDA_VISIBLE_DEVICES='' "$cpu" -IB "$score" score --run "$run" --freeze "$freeze"
