#!/usr/bin/env bash
#SBATCH --job-name=mf-cross-seed
#SBATCH --partition=h200q
#SBATCH --qos=h200q_1g
#SBATCH --gres=none
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=2
#SBATCH --mem=4G
#SBATCH --time=00:10:00
#SBATCH --output=/work/ChihChungHsu/cactus8603/icml/tmp/mf-cross-seed-%j.log
set -euo pipefail

freeze=${1:?Pass the reviewed full comparison commit}
[[ "$freeze" =~ ^[0-9a-f]{40}$ ]]
[[ "$#" == 1 || "$#" == 2 && "$2" == --check-only ]]
root=/work/ChihChungHsu/cactus8603
shared="$root/icml"
repo="$root/icml27-topic-search"
stage=codex/experiments/multifidelity-cross-seed
pilot=codex/experiments/multifidelity-second-seed
topic=killtests/multifidelity-attribution
python="$shared/envs/attrib-jax/bin/python"
changes=$(git -C "$repo" status --porcelain --untracked-files=all -- "$stage")
[[ -z "$changes" ]]
git -C "$repo" diff --quiet "$freeze" HEAD -- "$stage"
export MULTIFID_GIT_REPO="$repo" TMPDIR="$shared/tmp" TZ=Asia/Taipei CUDA_VISIBLE_DEVICES=''
export CIFAR_DIR="$shared/data/cifar10/cifar-10-batches-py" OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1
bundle=$(mktemp -d "$TMPDIR/mf-cross-seed-${SLURM_JOB_ID:-check}.XXXXXXXX")
trap 'rm -r --one-file-system -- "$bundle"' EXIT
git -C "$repo" archive "$freeze" "$stage" | tar -x -C "$bundle"
git -C "$repo" archive c6ff6aabd5229f3bd65966a741b5ad7789187c94 "$pilot" | tar -x -C "$bundle"
git -C "$repo" archive 599469e3faf5dec7aae854cc929c6d265bdd706f \
    "$topic/scripts/"{common.py,mf_core.py,cheap_torch.py} "$topic/configs/"{stage0.json,criteria.json} \
    | tar -x -C "$bundle"
git -C "$repo" archive ca8252971d720586b8426cd49ffc0290606293e5 codex/multifidelity-heldout/results.json \
    | tar -x -C "$bundle"
if [[ "${2:-}" == --check-only ]]; then
    "$python" -IB "$bundle/$stage/test_compare.py"
    "$python" -IB "$bundle/$stage/compare.py" preflight
    exit
fi
job=${SLURM_JOB_ID:?Submit through Slurm}
[[ "$job" =~ ^[1-9][0-9]*$ ]]
out="$shared/runs/multifid/cross_seed/$job"
mkdir -p "$(dirname "$out")"
mkdir "$out"
"$python" -IB "$bundle/$stage/compare.py" score --freeze "$freeze" > "$out/summary.partial"
mv -- "$out/summary.partial" "$out/summary.json"
