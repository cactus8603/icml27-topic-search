#!/bin/bash
#SBATCH --job-name=rsd-template
#SBATCH --partition=h200q
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=48G
#SBATCH --time=00:30:00
#SBATCH --output=/work/ChihChungHsu/cactus8603/icml/tmp/rsd-template-%j.log
set -euo pipefail

root=/work/ChihChungHsu/cactus8603
repo="$root/icml27-topic-search"
shared="$root/icml"
task=codex/experiments/rl-template
topic=killtests/rl-sparse-distill
freeze=${1:?Pass the full frozen commit}
mode=${2:-}
[[ "$freeze" =~ ^[a-f0-9]{40}$ && $# -le 2 && ( -z "$mode" || "$mode" == --check-only ) ]]
git -C "$repo" diff --quiet "$freeze" -- "$task" "$topic/scripts" "$topic/configs" "$topic/data/eval.jsonl"
export HF_HOME="$shared/hf_home" TMPDIR="$shared/tmp" HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1
export TZ=Asia/Taipei OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 TOKENIZERS_PARALLELISM=false
export VLLM_USE_FLASHINFER_SAMPLER=0 VLLM_NO_USAGE_STATS=1
scratch=$(mktemp -d "$TMPDIR/rsd-template-${SLURM_JOB_ID:-check}.XXXXXXXX")
trap 'rm -r --one-file-system -- "$scratch"' EXIT
export XDG_CACHE_HOME="$scratch/xdg"
git -C "$repo" archive "$freeze" "$task" "$topic/scripts" "$topic/configs" "$topic/data/eval.jsonl" |
  tar -x -C "$scratch"
python="$shared/envs/vllm033/bin/python"
entry="$scratch/$task/probe.py"
args=(--shared "$shared" --scratch "$scratch")
echo "[codex] $(date '+%F %R') template sensitivity freeze=$freeze"
"$python" -IB "$entry" prepare "${args[@]}"
if [[ "$mode" == --check-only ]]; then
  "$python" -IB "$entry" check "${args[@]}"
else
  job=${SLURM_JOB_ID:?Submit through Slurm}
  output="$shared/runs/rsd-template"
  mkdir -p -- "$output"
  [[ -w "$output" && ! -e "$output/$job.json" ]]
  "$python" -IB "$entry" sample "${args[@]}"
  "$python" -IB "$entry" score "${args[@]}" --freeze "$freeze" --output "$output/$job.json"
fi
