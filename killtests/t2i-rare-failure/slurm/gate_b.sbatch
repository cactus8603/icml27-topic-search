#!/bin/bash
#SBATCH --job-name=t2ir-gateb
#SBATCH --partition=h200q
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=16
#SBATCH --mem=64G
#SBATCH --time=03:00:00
#SBATCH --output=logs/%x-%j.out
# 閘門 (b)：sd14 x {E1,N1,B1} x 10,000 張，只算 CLIP（30,000 張，推估約 0.9 GPU 小時，未實測）。
# 必須在 gate (a) 通過、bench 實測之後才送。
set -euo pipefail
source slurm/env.sh
$PY scripts/generate.py --model sd14 --prompts E1 N1 B1 --seed-start 0 --seed-end 10000 --detectors clip --out results_gateb
$PY scripts/gate_b_check.py results_gateb
