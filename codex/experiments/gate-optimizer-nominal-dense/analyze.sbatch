#!/usr/bin/env bash
#SBATCH --job-name=gate-nominal-dense-score
#SBATCH --partition=h200q
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=2
#SBATCH --mem=8G
#SBATCH --time=00:20:00
#SBATCH --output=/work/ChihChungHsu/cactus8603/icml/tmp/gate-nominal-dense-score-%j.log
set -euo pipefail

repo=/work/ChihChungHsu/cactus8603/icml27-topic-search
stage=codex/experiments/gate-optimizer-nominal-dense
freeze=${1:?Pass the reviewed full commit}
array=${2:?Pass the producer array job ID}
[[ "$#" == 2 && "$freeze" =~ ^[0-9a-f]{40}$ && "$array" =~ ^[1-9][0-9]*$ ]]
launcher=$(git -C "$repo" show "$freeze:$stage/run.sbatch")
exec bash -c "$launcher" gate-nominal-dense "$freeze" analyze "$array"
