#!/usr/bin/env bash
#SBATCH --job-name=gate-fresh
#SBATCH --partition=h200q
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=32G
#SBATCH --time=01:00:00
#SBATCH --array=0-3%1
#SBATCH --output=/work/ChihChungHsu/cactus8603/icml/tmp/gate-fresh-%A_%a.log
set -euo pipefail
repo=/work/ChihChungHsu/cactus8603/icml27-topic-search
freeze=${1:?Pass the full freeze commit}
[[ "$freeze" =~ ^[0-9a-f]{40}$ && $# -le 2 ]]
stage=codex/experiments/gate-optimizer/round3
launcher=$(git -C "$repo" show "$freeze:$stage/launch.sh")
exec bash -c "$launcher" gate-fresh produce "$@"
