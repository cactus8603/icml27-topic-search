#!/usr/bin/env bash
#SBATCH --job-name=rsd-oat-v1-engine
#SBATCH --partition=h200q
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=48G
#SBATCH --time=00:30:00
#SBATCH --output=/work/ChihChungHsu/cactus8603/icml/tmp/rsd-oat-v1-engine-%j.log
set -euo pipefail

freeze=${1:?Use the reviewed full commit}
mode=${2:?Use --check-assets, --check-runtime, smoke or run}
[[ "$freeze" =~ ^[0-9a-f]{40}$ ]]
case "$mode" in
    --check-assets) [[ "$#" == 2 ]] ;;
    --check-runtime|smoke) [[ "$#" == 3 && "$3" =~ ^[0-9a-f]{64}$ ]] ;;
    run) [[ "$#" == 4 && "$3" =~ ^[0-9a-f]{64}$ && "$4" =~ ^[1-9][0-9]*$ ]] ;;
    *) exit 2 ;;
esac

root=/work/ChihChungHsu/cactus8603
shared="$root/icml"
repo="$root/icml27-topic-search"
stage=codex/experiments/rl-oat-v1-engine
legacy=codex/experiments/rl-pair-a-author/evaluate.py
legacy_freeze=0ea7e896d68e238a360402607a1573185430c2d2
changes=$(git -C "$repo" status --porcelain --untracked-files=all -- "$stage")
[[ -z "$changes" ]]
git -C "$repo" diff --quiet "$freeze" HEAD -- "$stage"
git -C "$repo" cat-file -e "$freeze:$stage/engine.py"

export TZ=Asia/Taipei TMPDIR="$shared/tmp" HF_HOME="$shared/hf_home"
export HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 HF_HUB_DISABLE_IMPLICIT_TOKEN=1
export OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 TOKENIZERS_PARALLELISM=false
export PYTHONDONTWRITEBYTECODE=1 PYTHONUNBUFFERED=1 VLLM_NO_USAGE_STATS=1 DO_NOT_TRACK=1
unset OPENAI_API_KEY OPENAI_BASE_URL HF_TOKEN HUGGING_FACE_HUB_TOKEN HUGGINGFACE_HUB_TOKEN
unset VLLM_USE_V1 VLLM_USE_FLASHINFER_SAMPLER VLLM_ATTENTION_BACKEND
umask 077
bundle=$(mktemp -d "$TMPDIR/rsd-oat072-${SLURM_JOB_ID:-check}.XXXXXXXX")
trap 'rm -r --one-file-system -- "$bundle"' EXIT
export XDG_CACHE_HOME="$bundle/xdg" TORCHINDUCTOR_CACHE_DIR="$bundle/torch"
git -C "$repo" archive "$freeze" "$stage" | tar -x -C "$bundle"
git -C "$repo" archive "$legacy_freeze" "$legacy" | tar -x -C "$bundle"
entry="$bundle/$stage/engine.py"
if [[ "$mode" == --check-assets ]]; then
    CUDA_VISIBLE_DEVICES='' "$shared/envs/vllm033/bin/python" -IBu "$entry" asset-preflight
    exit 0
fi
python="$shared/envs/oat-vllm072/bin/python"
[[ -x "$python" ]] || { printf '[codex] Missing isolated vLLM 0.7.2 runtime\n' >&2; exit 1; }
if [[ "$mode" == --check-runtime ]]; then
    CUDA_VISIBLE_DEVICES='' "$python" -IBu "$entry" runtime-preflight --inventory-sha "$3"
    exit 0
fi

job=${SLURM_JOB_ID:?Use a reviewed Slurm job}
[[ "$job" =~ ^[1-9][0-9]*$ ]]
args=("$mode" --freeze "$freeze" --inventory-sha "$3")
receipt="$shared/runs/rsd-pair-a-v1-engine/smoke-$job.json"
if [[ "$mode" == run ]]; then
    args+=(--smoke-job "$4")
    receipt="$shared/runs/rsd-pair-a-v1-engine/$job.json"
fi
echo "[codex] $(date '+%F %R') Oat-Zero paper-v1 recipe $mode freeze=$freeze"
run_author() {
    if [[ "$mode" == smoke ]]; then
        "$python" -IBu -X faulthandler "$entry" "${args[@]}"
    else
        "$python" -IBu "$entry" "${args[@]}" >/dev/null 2>&1
    fi
}
if OAT_REPORT_STAGE=1 run_author 3>&2; then
    [[ -f "$receipt" && ! -L "$receipt" ]]
    sha256sum -- "$receipt"
else
    status=$?
    printf '[codex] Oat-Zero paper-v1 evaluation exited %s\n' "$status" >&2
    exit "$status"
fi
