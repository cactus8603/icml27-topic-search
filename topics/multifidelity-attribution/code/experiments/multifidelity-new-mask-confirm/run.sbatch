#!/usr/bin/env bash
#SBATCH --job-name=mf-new-mask-confirm
#SBATCH --partition=h200q
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --time=01:00:00
#SBATCH --output=/work/ChihChungHsu/cactus8603/icml/tmp/mf-new-mask-confirm-%j.log
set -euo pipefail
umask 077

[[ "$#" == 1 || "$#" == 2 && "$2" == --check-only ]]
freeze=$1
[[ "$freeze" =~ ^[0-9a-f]{40}$ ]]
mode=${2:-run}
job=${SLURM_JOB_ID:-check}
if [[ "$mode" == run ]]; then
    [[ "$job" =~ ^[1-9][0-9]*$ ]]
    [[ "$(sacct -X -n -P -j 26734 -o JobID,State,ExitCode,JobName%50)" == '26734|COMPLETED|0:0|mf-signal-confirm' ]]
fi

root=/work/ChihChungHsu/cactus8603
repo="${MULTIFID_FROZEN_REPO:-$root/icml27-topic-search}"
shared="$root/icml"
stage=codex/experiments/multifidelity-new-mask-confirm
source=killtests/multifidelity-attribution
helper=codex/experiments/multifidelity-heldout-lds/pilot.py
signal=codex/experiments/multifidelity-signal-confirm/score.py
jax="$shared/envs/attrib-jax/bin/python"
cpu="$shared/envs/attrib-torch/bin/python"
[[ -z "$(git -C "$repo" status --porcelain --untracked-files=all -- "$stage")" ]]
git -C "$repo" diff --quiet "$freeze" HEAD -- "$stage"

export TMPDIR="$shared/tmp" TZ=Asia/Taipei PYTHONDONTWRITEBYTECODE=1 PYTHONUNBUFFERED=1
export CIFAR_DIR="$shared/data/cifar10/cifar-10-batches-py"
export OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1
bundle=$(mktemp -d "$TMPDIR/mf-new-mask-confirm-$job.XXXXXXXX")
trap 'rm -r --one-file-system -- "$bundle"' EXIT
git -C "$repo" archive "$freeze" "$stage" | tar -xf - -C "$bundle"
git -C "$repo" archive 599469e3faf5dec7aae854cc929c6d265bdd706f \
    "$source/configs/"{stage0,criteria}.json "$source/scripts/"{common,mf_core,hybrid,stage0_criteria}.py | tar -xf - -C "$bundle"
git -C "$repo" archive c7aa443a3d284ec1accbd841d7d383a4b0f52058 "$helper" | tar -xf - -C "$bundle"
git -C "$repo" archive 09661dc26c24ea06623bae30fe6fdc80692dd0aa "$signal" | tar -xf - -C "$bundle"

export XDG_CACHE_HOME="$bundle/xdg"
entry="$bundle/$stage/experiment.py"
if [[ "$mode" == --check-only ]]; then
    CUDA_VISIBLE_DEVICES='' JAX_PLATFORMS=cpu "$cpu" -IB "$bundle/$stage/test_experiment.py"
    CUDA_VISIBLE_DEVICES='' JAX_PLATFORMS=cpu "$jax" -IB - "$bundle/$stage/inputs.json" <<'PY'
from importlib.metadata import version
import json
import sys
with open(sys.argv[1]) as stream:
    expected = json.load(stream)["gpu_packages"]
if any(version(name) != value for name, value in expected.items()):
    raise SystemExit("Frozen GPU environment packages differ")
PY
fi
CUDA_VISIBLE_DEVICES='' JAX_PLATFORMS=cpu "$cpu" -IB "$entry" preflight
[[ "$mode" == run ]] || exit 0

run="$shared/runs/multifid/new-mask-confirm/$job"
mkdir -p "$(dirname "$run")"
mkdir "$run"
unset JAX_DEFAULT_MATMUL_PRECISION JAX_ENABLE_X64 JAX_DISABLE_JIT JAX_DEFAULT_PRNG_IMPL XLA_FLAGS
export XLA_PYTHON_CLIENT_MEM_FRACTION=0.90
JAX_PLATFORMS=cuda "$jax" -IB "$entry" retrain --freeze "$freeze" --run "$run"
CUDA_VISIBLE_DEVICES='' JAX_PLATFORMS=cpu "$cpu" -IB "$entry" score --freeze "$freeze" --run "$run"
printf '[codex] '
sha256sum -- "$run/summary.json"
