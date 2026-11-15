#!/usr/bin/env bash
#SBATCH --job-name=rsd-oat-author
#SBATCH --partition=h200q
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=48G
#SBATCH --time=00:30:00
#SBATCH --output=/work/ChihChungHsu/cactus8603/icml/tmp/rsd-oat-author-%j.log
set -euo pipefail

freeze=${1:?Pass the full frozen commit}
mode=${2:?Pass --check-only, --check-runtime, smoke or run}
[[ "$freeze" =~ ^[0-9a-f]{40}$ ]]
case "$mode" in
    --check-only|--check-runtime|smoke) [[ "$#" == 2 ]] ;;
    run) [[ "$#" == 3 && "$3" =~ ^[1-9][0-9]*$ ]] ;;
    *) exit 2 ;;
esac
root=/work/ChihChungHsu/cactus8603
shared="$root/icml"
repo="$root/icml27-topic-search"
stage=codex/experiments/rl-pair-a-author
changes=$(git -C "$repo" status --porcelain --untracked-files=all -- "$stage")
[[ -z "$changes" ]]
git -C "$repo" diff --quiet "$freeze" HEAD -- "$stage"
git -C "$repo" cat-file -e "$freeze:$stage/evaluate.py"

export TZ=Asia/Taipei TMPDIR="$shared/tmp" HF_HOME="$shared/hf_home"
export HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 HF_HUB_DISABLE_IMPLICIT_TOKEN=1
export OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 TOKENIZERS_PARALLELISM=false
export PYTHONDONTWRITEBYTECODE=1 PYTHONUNBUFFERED=1 VLLM_NO_USAGE_STATS=1 DO_NOT_TRACK=1
unset OPENAI_API_KEY OPENAI_BASE_URL HF_TOKEN HUGGING_FACE_HUB_TOKEN HUGGINGFACE_HUB_TOKEN
unset VLLM_USE_V1 VLLM_USE_FLASHINFER_SAMPLER VLLM_ATTENTION_BACKEND
umask 077
bundle=$(mktemp -d "$TMPDIR/rsd-oat-author-${SLURM_JOB_ID:-check}.XXXXXXXX")
trap 'rm -r --one-file-system -- "$bundle"' EXIT
export XDG_CACHE_HOME="$bundle/xdg" TORCHINDUCTOR_CACHE_DIR="$bundle/torch"
git -C "$repo" archive "$freeze" "$stage" | tar -x -C "$bundle"
entry="$bundle/$stage/evaluate.py"
if [[ "$mode" == --check-only ]]; then
    CUDA_VISIBLE_DEVICES='' "$shared/envs/vllm033/bin/python" -IBu "$entry" preflight
    exit 0
fi
python="$shared/envs/oat-vllm084/bin/python"
[[ -x "$python" ]] || { printf '[codex] Missing standalone Python 3.10 / vLLM 0.8.4 runtime\n' >&2; exit 1; }
if [[ "$mode" == --check-runtime ]]; then
    CUDA_VISIBLE_DEVICES='' "$python" -IBu "$entry" runtime-preflight
    exit 0
fi
job=${SLURM_JOB_ID:?Submit the reviewed evaluation through Slurm}
[[ "$job" =~ ^[1-9][0-9]*$ ]]
args=("$mode" --freeze "$freeze")
receipt="$shared/runs/rsd-pair-a-author/smoke-$job.json"
if [[ "$mode" == run ]]; then
    args+=(--smoke-job "$3")
    receipt="$shared/runs/rsd-pair-a-author/$job.json"
fi
echo "[codex] $(date '+%F %R') Oat-Zero public author $mode freeze=$freeze"
if OAT_REPORT_STAGE=1 "$python" -IBu "$entry" "${args[@]}" 3>&2 >/dev/null 2>&1; then
    [[ -f "$receipt" && ! -L "$receipt" ]]
    sha256sum -- "$receipt"
else
    status=$?
    printf '[codex] Oat-Zero evaluation exited %s\n' "$status" >&2
    exit "$status"
fi
