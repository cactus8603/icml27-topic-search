#!/bin/bash
#SBATCH --job-name=rsd-author-stop
#SBATCH --partition=h200q
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=48G
#SBATCH --time=00:30:00
#SBATCH --output=/work/ChihChungHsu/cactus8603/icml/tmp/rsd-author-stop-%j.log
set -euo pipefail

root=/work/ChihChungHsu/cactus8603
repo="$root/icml27-topic-search"
shared="$root/icml"
stage=codex/experiments/rl-author-stop
template=codex/experiments/rl-template
topic=killtests/rl-sparse-distill
freeze=${1:?Pass the full frozen commit}
mode=${2:-}
[[ "$freeze" =~ ^[a-f0-9]{40}$ && $# -le 2 && ( -z "$mode" || "$mode" == --check-only ) ]]
paths=("$stage" "$template/probe.py" "$topic/scripts" "$topic/configs" "$topic/data/eval.jsonl")
git -C "$repo" diff --quiet "$freeze" -- "${paths[@]}"
changes=$(git -C "$repo" status --porcelain --untracked-files=all -- "$stage")
[[ -z "$changes" ]]

export HF_HOME="$shared/hf_home" TMPDIR="$shared/tmp" HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1
export TZ=Asia/Taipei OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 TOKENIZERS_PARALLELISM=false
export HF_HUB_DISABLE_IMPLICIT_TOKEN=1 VLLM_USE_FLASHINFER_SAMPLER=0 VLLM_NO_USAGE_STATS=1
unset OPENAI_API_KEY OPENAI_BASE_URL HF_TOKEN HUGGING_FACE_HUB_TOKEN HUGGINGFACE_HUB_TOKEN
umask 077
scratch=$(mktemp -d "$TMPDIR/rsd-author-stop-${SLURM_JOB_ID:-check}.XXXXXXXX")
trap 'rm -r --one-file-system -- "$scratch"' EXIT
export XDG_CACHE_HOME="$scratch/xdg" TORCHINDUCTOR_CACHE_DIR="$scratch/torch"
git -C "$repo" archive "$freeze" "${paths[@]}" | tar -x -C "$scratch"
python="$shared/envs/vllm033/bin/python"
entry="$scratch/$stage/probe_stop.py"
args=(--shared "$shared" --scratch "$scratch")
echo "[codex] $(date '+%F %R') SimpleRL-Zoo stop-string sensitivity freeze=$freeze"
if [[ "$mode" == --check-only ]]; then
    export CUDA_VISIBLE_DEVICES=''
    "$python" -IBu "$entry" check "${args[@]}"
else
    job=${SLURM_JOB_ID:?Submit through Slurm}
    [[ "$job" =~ ^[1-9][0-9]*$ ]]
    output="$shared/runs/rsd-author-stop"
    mkdir -p -- "$output"
    [[ ! -e "$output/$job.json" && ! -L "$output/$job.json" ]]
    ulimit -f 524288
    "$python" -IBu "$entry" sample "${args[@]}"
    "$python" -IBu "$entry" score "${args[@]}" --freeze "$freeze" --output "$output/$job.json"
fi
