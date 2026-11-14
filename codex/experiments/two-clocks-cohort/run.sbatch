#!/usr/bin/env bash
#SBATCH --job-name=cfg-cohort-prefix
#SBATCH --partition=h200q
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=40G
#SBATCH --time=01:30:00
#SBATCH --output=/work/ChihChungHsu/cactus8603/icml/tmp/cfg-cohort-prefix-%j.log
set -euo pipefail

topic=killtests/biased-gen-cfg
experiment=codex/experiments/two-clocks-cohort
prefix=codex/experiments/two-clocks-prefix
formal=9e92804a7bfded3c5cafd4fccd4315df2034eca5
legacy=df7cdb280888512e9006f61f642ad5caf479603c
dependency=df08bc634274f78ba1b99e83b07228b4252608ce
prefix_freeze=f292d94aa65da583ff5a787eed457d3f429cebb1
repo=$(git -C "${SLURM_SUBMIT_DIR:-.}" rev-parse --show-toplevel)
shared=/work/ChihChungHsu/cactus8603/icml
freeze=${1:?Pass the full freeze commit, or --check-only for a draft preflight}
[[ "$#" == 1 || "$#" == 2 && "$2" == --check-only ]]

export TZ=Asia/Taipei TMPDIR="$shared/tmp" PYTHONDONTWRITEBYTECODE=1
export PYTHONUNBUFFERED=1 MPLBACKEND=Agg OMP_NUM_THREADS="${SLURM_CPUS_PER_TASK:-4}"
snapshot=$(mktemp -d "$TMPDIR/two-clocks-cohort-${SLURM_JOB_ID:-check}.XXXXXXXX")
trap 'rm -r --one-file-system -- "$snapshot"' EXIT
if [[ "$freeze" == --check-only && "$#" == 1 ]]; then
    script="$repo/$experiment/contrast.py"
else
    [[ "$freeze" =~ ^[0-9a-f]{40}$ ]]
    git -C "$repo" diff --quiet "$freeze" -- "$experiment"
    changes=$(git -C "$repo" status --porcelain -- "$experiment")
    [[ -z "$changes" ]]
    git -C "$repo" archive "$freeze" "$experiment" | tar -x -C "$snapshot"
    script="$snapshot/$experiment/contrast.py"
fi

git -C "$repo" archive "$prefix_freeze" "$prefix/replay.py" | tar -x -C "$snapshot"
git -C "$repo" archive "$formal" "$topic/scripts" "$topic/configs" | tar -x -C "$snapshot"
export BG_EXT="$snapshot/external"
mkdir -p -- "$BG_EXT" "$snapshot/legacy"
git -C "$repo" archive "$legacy" "$topic"/scripts/{train_hier_cond,hier_cond,schedule}.py |
    tar -x -C "$snapshot/legacy"
git -C "$shared/ext/biased-generalization" archive "$dependency" modules/embeddings.py |
    tar -x -C "$BG_EXT"

data="$shared/data/biasgen-hier/hier_q6_l4_s1.0_qeff4_seed0_n40000.npy"
args=(--source "$snapshot/$topic" --legacy "$snapshot/legacy/$topic" --prefix-source "$snapshot/$prefix"
      --checkpoints "$repo/$topic/results/ckpt" --data "$data" --work "$snapshot" "--commit=$freeze")
if [[ "$freeze" != --check-only && "${2:-}" != --check-only ]]; then
    job=${SLURM_JOB_ID:?Submit through Slurm}
    [[ $job =~ ^[0-9]+$ ]]
    args+=(--output "$shared/runs/two-clocks-cohort/$job")
fi
echo "[codex] $(date '+%Y-%m-%d %H:%M') three-arm early-grid probe freeze=$freeze original=$formal"
"$shared/envs/biasgen/bin/python" -IB "$script" "${args[@]}"
