#!/usr/bin/env bash
#SBATCH --job-name=mf-mask-span
#SBATCH --partition=h200q
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=96G
#SBATCH --time=02:00:00
#SBATCH --output=/work/ChihChungHsu/cactus8603/icml/tmp/mf-mask-span-%j.log
set -euo pipefail

work=/work/ChihChungHsu/cactus8603
repo="$work/icml27-topic-search"
shared="$work/icml"
topic=killtests/multifidelity-attribution
pilot=codex/experiments/multifidelity-direction-coverage
formal=599469e3faf5dec7aae854cc929c6d265bdd706f
freeze=${1:?Pass the full freeze commit and optionally --check-only}
[[ "$freeze" =~ ^[0-9a-f]{40}$ ]]
[[ "$#" == 1 || "$#" == 2 && "$2" == --check-only ]]
git -C "$repo" cat-file -e "$freeze:$pilot/pilot.py"
git -C "$repo" diff --quiet "$freeze" -- "$pilot"

export TZ=Asia/Taipei PYTHONDONTWRITEBYTECODE=1
export CIFAR_DIR="$shared/data/cifar10/cifar-10-batches-py"
export OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1
export JAX_PLATFORMS=cuda XLA_PYTHON_CLIENT_MEM_FRACTION=0.90
unset JAX_DEFAULT_MATMUL_PRECISION JAX_ENABLE_X64 JAX_DISABLE_JIT XLA_FLAGS
export TMPDIR="$shared/tmp"
snapshot=$(mktemp -d "$TMPDIR/mf-mask-span-${SLURM_JOB_ID:-check}.XXXXXXXX")
trap 'rm -r --one-file-system -- "$snapshot"' EXIT
git -C "$repo" archive "$freeze" "$pilot" | tar -x -C "$snapshot"
git -C "$repo" archive "$formal" "$topic/scripts/common.py" "$topic/scripts/mf_core.py" \
  "$topic/configs/stage0.json" | tar -x -C "$snapshot"
(cd "$CIFAR_DIR" && sha256sum --quiet -c "$snapshot/$pilot/cifar.sha256")

args=(--source "$snapshot/$topic" --commit "$freeze")
if [[ "${2:-}" != --check-only ]]; then
  job=${SLURM_JOB_ID:?Submit through Slurm}
  args+=(--output "$shared/runs/multifid/direction_coverage/$job")
fi
echo "[codex] $(date '+%Y-%m-%d %H:%M') fresh MF direction coverage freeze=$freeze source=$formal"
"$shared/envs/attrib-jax/bin/python" -IBu "$snapshot/$pilot/pilot.py" "${args[@]}"
