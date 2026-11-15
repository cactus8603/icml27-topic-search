#!/usr/bin/env bash
#SBATCH --job-name=mf-complement
#SBATCH --partition=h200q
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=96G
#SBATCH --time=00:45:00
#SBATCH --output=/work/ChihChungHsu/cactus8603/icml/tmp/mf-complement-%j.log
set -euo pipefail

freeze=${1:?Pass the frozen diagnostic commit}
[[ "$freeze" =~ ^[0-9a-f]{40}$ ]]
[[ "$#" == 1 || "$#" == 2 && "$2" == --check-only ]]
root=/work/ChihChungHsu/cactus8603
repo="$root/icml27-topic-search"
shared="$root/icml"
stage=codex/experiments/multifidelity-complement
source=killtests/multifidelity-attribution
pilot=codex/experiments/multifidelity-second-seed
kron_commit=19088a899dd333aae086fe82213a972db5e6c493
jax_python="$shared/envs/attrib-jax/bin/python"
torch_python="$shared/envs/attrib-torch/bin/python"
changes=$(git -C "$repo" status --porcelain --untracked-files=all -- "$stage")
[[ -z "$changes" ]]
git -C "$repo" diff --quiet "$freeze" HEAD -- "$stage"
export TMPDIR="$shared/tmp" TZ=Asia/Taipei PYTHONDONTWRITEBYTECODE=1 PYTHONUNBUFFERED=1
export CIFAR_DIR="$shared/data/cifar10/cifar-10-batches-py"
export OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1
bundle=$(mktemp -d "$TMPDIR/mf-complement-${SLURM_JOB_ID:-check}.XXXXXXXX")
trap 'rm -r --one-file-system -- "$bundle"' EXIT
git -C "$repo" archive "$freeze" "$stage" | tar -x -C "$bundle"
git -C "$repo" archive c6ff6aabd5229f3bd65966a741b5ad7789187c94 "$pilot/config.json" "$pilot/cifar.sha256" \
  | tar -x -C "$bundle"
git -C "$repo" archive 599469e3faf5dec7aae854cc929c6d265bdd706f \
  "$source/scripts/"{common.py,mf_core.py,cheap_torch.py} "$source/configs/"{stage0.json,criteria.json} \
  | tar -x -C "$bundle"
git -C "$repo" archive ca8252971d720586b8426cd49ffc0290606293e5 codex/multifidelity-heldout/results.json \
  | tar -x -C "$bundle"
export MULTIFID_KRON_ARCHIVE="$bundle/kron"
export MULTIFID_KRON_OUTPUT="$bundle/kron-output"
mkdir "$MULTIFID_KRON_ARCHIVE"
git -C "$shared/ext/kronfluence" archive "$kron_commit" kronfluence | tar -x -C "$MULTIFID_KRON_ARCHIVE"
if [[ "${2:-}" == --check-only ]]; then
    CUDA_VISIBLE_DEVICES='' "$jax_python" -IB "$bundle/$stage/test_pilot.py"
    CUDA_VISIBLE_DEVICES='' "$jax_python" -IB "$bundle/$stage/pilot.py" preflight
    exit
fi
job=${SLURM_JOB_ID:?Submit through Slurm}
[[ "$job" =~ ^[1-9][0-9]*$ ]]
run="$shared/runs/multifid/complement/$job"
mkdir -p "$(dirname "$run")"
mkdir "$run"
export REQUIRE_GPU=1
script="$bundle/$stage/pilot.py"
args=(--freeze "$freeze" --run "$run")
CUDA_VISIBLE_DEVICES='' "$jax_python" -IB "$script" reuse-exact "${args[@]}"
# The runtime's isolated mode ignores PYTHONPATH. Add only the frozen Kronfluence path.
"$torch_python" -IB -c 'import os,runpy,sys; sys.path.insert(0,os.environ["MULTIFID_KRON_ARCHIVE"]); sys.argv=sys.argv[1:]; runpy.run_path(sys.argv[0],run_name="__main__")' \
    "$script" cheap "${args[@]}"
CUDA_VISIBLE_DEVICES='' "$jax_python" -IB "$script" score "${args[@]}"
