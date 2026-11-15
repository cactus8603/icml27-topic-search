#!/usr/bin/env bash
#SBATCH --job-name=mf-heldout-lds
#SBATCH --partition=h200q
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --time=01:00:00
#SBATCH --output=/work/ChihChungHsu/cactus8603/icml/tmp/mf-heldout-lds-%j.log
set -euo pipefail

[[ "$#" == 1 || "$#" == 2 && "$2" == --check-only ]]
freeze=$1
[[ "$freeze" =~ ^[0-9a-f]{40}$ ]]
job=check
if [[ -v SLURM_JOB_ID ]]; then
    job=$SLURM_JOB_ID
    [[ "$job" =~ ^[1-9][0-9]*$ ]]
fi

root=/work/ChihChungHsu/cactus8603
repo="$root/icml27-topic-search"
shared="$root/icml"
stage=codex/experiments/multifidelity-heldout-lds
source=killtests/multifidelity-attribution
projection=codex/experiments/multifidelity-complement/pilot.py
python="$shared/envs/attrib-jax/bin/python"
[[ -z "$(git -C "$repo" status --porcelain --untracked-files=all -- "$stage")" ]]
git -C "$repo" diff --quiet "$freeze" HEAD -- "$stage"

export TMPDIR="$shared/tmp" TZ=Asia/Taipei PYTHONDONTWRITEBYTECODE=1 PYTHONUNBUFFERED=1
export CIFAR_DIR="$shared/data/cifar10/cifar-10-batches-py"
export OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1
umask 077
bundle=$(mktemp -d "$TMPDIR/mf-heldout-lds-$job.XXXXXXXX")
trap 'rm -r --one-file-system -- "$bundle"' EXIT
git -C "$repo" archive "$freeze" "$stage" | tar -x -C "$bundle"
git -C "$repo" archive 599469e3faf5dec7aae854cc929c6d265bdd706f \
    "$source/configs/stage0.json" "$source/scripts/"{common.py,hybrid.py,mf_core.py} | tar -x -C "$bundle"
git -C "$repo" archive 1432b5dffa9527677f3cce697f138bbe437c40cb "$projection" | tar -x -C "$bundle"

entry="$bundle/$stage/pilot.py"
if [[ "$#" == 2 ]]; then
    CUDA_VISIBLE_DEVICES='' "$python" -IB "$bundle/$stage/test_pilot.py"
    CUDA_VISIBLE_DEVICES='' "$python" -IB "$entry" preflight
    exit
fi

[[ "$job" != check ]]
run="$shared/runs/multifid/heldout-lds/$job"
mkdir -p "$(dirname "$run")"
mkdir "$run"
unset JAX_DEFAULT_MATMUL_PRECISION JAX_ENABLE_X64 JAX_DISABLE_JIT XLA_FLAGS
export XLA_PYTHON_CLIENT_MEM_FRACTION=0.90
export XDG_CACHE_HOME="$bundle/xdg"
JAX_PLATFORMS=cuda "$python" -IB "$entry" retrain --freeze "$freeze" --run "$run"
CUDA_VISIBLE_DEVICES='' "$python" -IB "$entry" score --freeze "$freeze" --run "$run"
sha256sum -- "$run/summary.json"
