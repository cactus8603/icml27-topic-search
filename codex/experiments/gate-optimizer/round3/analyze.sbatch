#!/usr/bin/env bash
#SBATCH --job-name=gate-fresh-score
#SBATCH --partition=h200q
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=2
#SBATCH --mem=8G
#SBATCH --time=00:20:00
#SBATCH --output=/work/ChihChungHsu/cactus8603/icml/tmp/gate-fresh-score-%j.log
set -euo pipefail
repo=/work/ChihChungHsu/cactus8603/icml27-topic-search
freeze=${1:?Pass the full freeze commit}
[[ "$freeze" =~ ^[0-9a-f]{40}$ && $# -le 2 ]]
stage=codex/experiments/gate-optimizer/round3
launcher=$(git -C "$repo" show "$freeze:$stage/launch.sh")
exec bash -c "$launcher" gate-fresh analyze "$@"
