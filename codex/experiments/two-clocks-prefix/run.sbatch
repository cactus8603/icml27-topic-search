#!/usr/bin/env bash
#SBATCH --job-name=cfg-prefix
#SBATCH --partition=h200q
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=40G
#SBATCH --time=01:30:00
#SBATCH --output=/work/ChihChungHsu/cactus8603/icml/tmp/cfg-prefix-%j.log
set -euo pipefail

work=/work/ChihChungHsu/cactus8603
repo="$work/icml27-topic-search"
shared="$work/icml"
topic=killtests/biased-gen-cfg
diagnostic=codex/experiments/two-clocks-prefix
formal=9e92804a7bfded3c5cafd4fccd4315df2034eca5
dependency=df08bc634274f78ba1b99e83b07228b4252608ce
freeze=${1:?Pass the full freeze commit and optionally --check-only}
[[ "$freeze" =~ ^[0-9a-f]{40}$ ]]
[[ "$#" == 1 || "$#" == 2 && "$2" == --check-only ]]
git -C "$repo" cat-file -e "$freeze:$diagnostic/replay.py"
git -C "$repo" diff --quiet "$freeze" -- "$diagnostic"

export TZ=Asia/Taipei TMPDIR="$shared/tmp" PYTHONDONTWRITEBYTECODE=1
export PYTHONUNBUFFERED=1 MPLBACKEND=Agg OMP_NUM_THREADS="${SLURM_CPUS_PER_TASK:-4}"
snapshot=$(mktemp -d "$TMPDIR/two-clocks-prefix-${SLURM_JOB_ID:-check}.XXXXXXXX")
trap 'rm -r --one-file-system -- "$snapshot"' EXIT
git -C "$repo" archive "$freeze" "$diagnostic" | tar -x -C "$snapshot"
git -C "$repo" archive "$formal" "$topic/scripts" "$topic/configs" | tar -x -C "$snapshot"
export BG_EXT="$snapshot/external"
mkdir -p "$BG_EXT"
git -C "$shared/ext/biased-generalization" archive "$dependency" modules/embeddings.py | tar -x -C "$BG_EXT"

python="$shared/envs/biasgen/bin/python"
data="$shared/data/biasgen-hier/hier_q6_l4_s1.0_qeff4_seed0_n40000.npy"
original="$repo/$topic/results/ckpt/week2-9e92804/hier_u833"
args=(--source "$snapshot/$topic" --data "$data" --original "$original" --work "$snapshot" --commit "$freeze")
if [[ "${2:-}" != --check-only ]]; then
  job=${SLURM_JOB_ID:?Submit through Slurm}
  args+=(--output "$shared/runs/two-clocks-prefix/$job")
fi
echo "[codex] $(date '+%Y-%m-%d %H:%M') CFG earlier-grid diagnostic freeze=$freeze original=$formal"
"$python" -IB "$snapshot/$diagnostic/replay.py" "${args[@]}"
