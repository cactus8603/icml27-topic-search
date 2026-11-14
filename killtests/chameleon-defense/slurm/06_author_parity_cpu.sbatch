#!/bin/bash
#SBATCH --job-name=cham-author-parity
#SBATCH --partition=defq
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=16
#SBATCH --mem=64G
#SBATCH --time=02:00:00
#SBATCH --output=/work/ChihChungHsu/cactus8603/icml27-topic-search/killtests/chameleon-defense/logs/%x-%j.out
set -euo pipefail

work=/work/ChihChungHsu/cactus8603
repo="$work/icml27-topic-search"
topic=killtests/chameleon-defense
commit=${1:?Usage: sbatch slurm/06_author_parity_cpu.sbatch FULL_COMMIT_SHA}
[[ "$commit" =~ ^[0-9a-f]{40}$ ]] || { echo "A full frozen commit hash is required" >&2; exit 1; }
[[ "$#" == 1 || "$#" == 2 && "$2" == --check-only ]] || exit 1
git -C "$repo" cat-file -e "$commit:$topic/scripts/audit_author_probe.py"
git -C "$repo" diff --quiet "$commit" -- "$topic" || {
    echo "Tracked topic files differ from the frozen commit" >&2
    exit 1
}

python="$work/icml/envs/inoc/bin/python"
export HF_HOME="$work/icml/hf_home" HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1
export TMPDIR="$work/icml/tmp"
export OMP_NUM_THREADS="${SLURM_CPUS_PER_TASK:-1}" OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS="${SLURM_CPUS_PER_TASK:-1}"
export TOKENIZERS_PARALLELISM=false
if [[ "${2:-}" == --check-only ]]; then
    exec "$python" -IB "$repo/$topic/scripts/audit_author_probe.py"
fi

job=${SLURM_JOB_ID:?Submit through Slurm}
snapshot=$(mktemp -d "$TMPDIR/cham-author-$job.XXXXXXXX")
trap 'rm -r -- "$snapshot"' EXIT
git -C "$repo" archive "$commit" "$topic/scripts" "$topic/configs" | tar -x -C "$snapshot"

echo "[codex] $(TZ=Asia/Taipei date '+%Y-%m-%d %H:%M') Chameleon clean author parity job=$job freeze=$commit"
"$python" -IB "$snapshot/$topic/scripts/audit_author_probe.py" --freeze "$commit" \
    --out "$repo/$topic/results/author_parity/$job.json"
