#!/usr/bin/env bash
#SBATCH --job-name=mf-second-seed
#SBATCH --partition=h200q
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=96G
#SBATCH --time=02:00:00
#SBATCH --output=/work/ChihChungHsu/cactus8603/icml/tmp/mf-second-seed-%j.log
set -euo pipefail

root=/work/ChihChungHsu/cactus8603
shared="$root/icml"
repo="$root/icml27-topic-search"
pilot=codex/experiments/multifidelity-second-seed
topic=killtests/multifidelity-attribution
jax_python="$shared/envs/attrib-jax/bin/python"
torch_python="$shared/envs/attrib-torch/bin/python"
job=${SLURM_JOB_ID:?Submit through Slurm}
freeze=${1:?Pass the reviewed full Git commit}
[[ "$freeze" =~ ^[0-9a-f]{40}$ && "$job" =~ ^[1-9][0-9]*$ ]]
changes=$(git -C "$repo" status --porcelain --untracked-files=all -- "$pilot")
[[ -z "$changes" ]]
git -C "$repo" diff --quiet "$freeze" HEAD -- "$pilot"

export TMPDIR="$shared/tmp" TZ=Asia/Taipei PYTHONDONTWRITEBYTECODE=1 PYTHONUNBUFFERED=1
export CIFAR_DIR="$shared/data/cifar10/cifar-10-batches-py"
export OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1
export XLA_PYTHON_CLIENT_MEM_FRACTION=0.90 REQUIRE_GPU=1
unset JAX_DEFAULT_MATMUL_PRECISION JAX_ENABLE_X64 JAX_DISABLE_JIT XLA_FLAGS
snapshot=$(mktemp -d "$TMPDIR/mf-second-seed-$job.XXXXXXXX")
trap 'rm -r --one-file-system -- "$snapshot"' EXIT
git -C "$repo" archive "$freeze" "$pilot" | tar -x -C "$snapshot"
config="$snapshot/$pilot/config.json"
source_commit=$("$jax_python" -B -c 'import json, sys; print(json.load(open(sys.argv[1]))["source_commit"])' "$config")
kron_commit=$("$jax_python" -B -c 'import json, sys; print(json.load(open(sys.argv[1]))["kronfluence_commit"])' "$config")
git -C "$repo" archive "$source_commit" "$topic/scripts/"{common.py,mf_core.py,cheap_torch.py} \
  "$topic/configs/"{stage0.json,criteria.json} | tar -x -C "$snapshot"
export MULTIFID_KRON_ARCHIVE="$snapshot/kron"
mkdir "$MULTIFID_KRON_ARCHIVE"
git -C "$shared/ext/kronfluence" archive "$kron_commit" kronfluence | tar -x -C "$MULTIFID_KRON_ARCHIVE"

run="$shared/runs/multifid/second_seed/$job"
mkdir -p "$(dirname "$run")"
mkdir "$run"
runner="$snapshot/$pilot/pilot.py"
args=(--source "$snapshot/$topic" --freeze "$freeze" --run "$run")
printf '[codex] %s second seed job=%s freeze=%s\n' "$(date '+%F %R')" "$job" "$freeze"
JAX_PLATFORMS=cuda "$jax_python" -B "$runner" --stage jax "${args[@]}"
PYTHONPATH="$MULTIFID_KRON_ARCHIVE" "$torch_python" -B "$runner" --stage ekfac "${args[@]}"
CUDA_VISIBLE_DEVICES='' "$jax_python" -B "$runner" --stage score "${args[@]}"
