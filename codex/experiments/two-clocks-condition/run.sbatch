#!/usr/bin/env bash
#SBATCH --job-name=cfg-condition-clock
#SBATCH --partition=h200q
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --time=00:45:00
#SBATCH --output=/work/ChihChungHsu/cactus8603/icml/tmp/cfg-condition-clock-%j.log
set -euo pipefail

root=/work/ChihChungHsu/cactus8603
repo="$root/icml27-topic-search"
shared="$root/icml"
stage=codex/experiments/two-clocks-condition
topic=killtests/biased-gen-cfg
prefix=codex/experiments/two-clocks-prefix
count=codex/experiments/two-clocks-count
formal=9e92804a7bfded3c5cafd4fccd4315df2034eca5
reference_prefix=f292d94aa65da583ff5a787eed457d3f429cebb1
reference_count=24cd3d380592ff0674535bcca3cdf2fa7b538910
external=df08bc634274f78ba1b99e83b07228b4252608ce
freeze=${1:?Pass the full design commit}
[[ "$freeze" =~ ^[0-9a-f]{40}$ ]]
mode=${2:---run}
[[ "$#" -le 2 && ( "$mode" == --run || "$mode" == --check-only || "$mode" == --cpu-smoke ) ]]
changes=$(git -C "$repo" status --porcelain --untracked-files=all -- "$stage")
[[ -z "$changes" ]]
git -C "$repo" diff --quiet "$freeze" HEAD -- "$stage"

export TZ=Asia/Taipei TMPDIR="$shared/tmp" OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1
bundle=$(mktemp -d "$TMPDIR/cfg-condition-${SLURM_JOB_ID:-check}.XXXXXXXX")
trap 'rm -r --one-file-system -- "$bundle"' EXIT
git -C "$repo" archive "$freeze" "$stage" | tar -x -C "$bundle"
git -C "$repo" archive "$formal" "$topic/scripts" "$topic/configs" | tar -x -C "$bundle"
git -C "$repo" archive "$reference_prefix" "$prefix/replay.py" | tar -x -C "$bundle"
git -C "$repo" archive "$reference_count" "$count/prepare.py" | tar -x -C "$bundle"
export BG_EXT="$bundle/biased-generalization"
mkdir "$BG_EXT"
git -C "$shared/ext/biased-generalization" archive "$external" modules/embeddings.py | tar -x -C "$BG_EXT"

if [[ "$mode" == --run ]]; then
    job=${SLURM_JOB_ID:?Use Slurm for the frozen diagnostic}
    [[ "$job" =~ ^[1-9][0-9]*$ && ! -e "$shared/runs/two-clocks-condition/$job" ]]
    quota -g -s "$(id -gn)"
else
    export CUDA_VISIBLE_DEVICES=''
fi
"$shared/envs/biasgen/bin/python" -IBu "$bundle/$stage/condition_clock.py" "$mode" \
    --source "$bundle/$topic" --prefix "$bundle/$prefix" --scratch "$bundle" --commit "$freeze"
