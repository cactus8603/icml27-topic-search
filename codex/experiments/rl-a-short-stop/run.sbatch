#!/usr/bin/env bash
#SBATCH --job-name=rsd-a-short-stop
#SBATCH --partition=h200q
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=16
#SBATCH --mem=170G
#SBATCH --time=00:25:00
#SBATCH --output=/work/ChihChungHsu/cactus8603/icml/tmp/rsd-a-short-stop-%j.log
set -euo pipefail

freeze=${1:?Pass the full frozen commit}
mode=${2:?Use --check-only, smoke or diagnostic}
[[ "$freeze" =~ ^[0-9a-f]{40}$ ]]
case "$mode" in
    --check-only|smoke) [[ "$#" == 2 ]] ;;
    diagnostic) [[ "$#" == 3 && "$3" =~ ^[1-9][0-9]*$ ]] ;;
    *) exit 2 ;;
esac

root=/work/ChihChungHsu/cactus8603
shared="$root/icml"
repo="$root/icml27-topic-search"
stage=codex/experiments/rl-a-short-stop
source=killtests/rl-sparse-distill
upstream=1432b5dffa9527677f3cce697f138bbe437c40cb
python="$shared/envs/oat-vllm072/bin/python"
[[ -x "$python" ]]
[[ -z "$(git -C "$repo" status --porcelain --untracked-files=all -- "$stage")" ]]
git -C "$repo" diff --quiet "$freeze" HEAD -- "$stage"
git -C "$repo" cat-file -e "$freeze:$stage/pilot.py"

export TZ=Asia/Taipei TMPDIR="$shared/tmp" HF_HOME="$shared/hf_home"
export HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 HF_HUB_DISABLE_IMPLICIT_TOKEN=1
export OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 TOKENIZERS_PARALLELISM=false
export PYTHONDONTWRITEBYTECODE=1 PYTHONUNBUFFERED=1 VLLM_NO_USAGE_STATS=1 DO_NOT_TRACK=1
unset OPENAI_API_KEY OPENAI_BASE_URL HF_TOKEN HUGGING_FACE_HUB_TOKEN HUGGINGFACE_HUB_TOKEN
unset VLLM_USE_V1 VLLM_USE_FLASHINFER_SAMPLER VLLM_ATTENTION_BACKEND
umask 077
bundle=$(mktemp -d "$TMPDIR/rsd-a-short-stop-${SLURM_JOB_ID:-check}.XXXXXXXX")
trap 'rm -r --one-file-system -- "$bundle"' EXIT
export XDG_CACHE_HOME="$bundle/xdg" TORCHINDUCTOR_CACHE_DIR="$bundle/torch"
git -C "$repo" archive "$freeze" "$stage" | tar -x -C "$bundle"
git -C "$repo" archive "$upstream" "$source"/{configs/pairs.json,data/eval.jsonl} | tar -x -C "$bundle"
entry="$bundle/$stage/pilot.py"
inputs=(--source "$bundle/$source")

if [[ "$mode" == --check-only ]]; then
    CUDA_VISIBLE_DEVICES='' "$python" -IBu "$bundle/$stage/test_pilot.py"
    CUDA_VISIBLE_DEVICES='' "$python" -IBu "$entry" preflight "${inputs[@]}"
    exit 0
fi

job=${SLURM_JOB_ID:?Submit through Slurm}
[[ "$job" =~ ^[1-9][0-9]*$ ]]
outdir="$shared/runs/rsd-pair-a-short-stop"
smoke_args=()
if [[ "$mode" == diagnostic ]]; then
    smoke_file="$outdir/smoke-$3.json"
    "$python" -IBu "$entry" check-smoke --freeze "$freeze" --job "$3" --output "$smoke_file"
    smoke_args=(--smoke-file "$smoke_file" --smoke-job "$3")
    destination="$outdir/$job.json"
else
    destination="$outdir/smoke-$job.json"
fi
[[ ! -e "$destination" && ! -L "$destination" ]]
args=(--mode "$mode" --job "$job" --freeze "$freeze")
for role in base rl; do
    echo "[codex] $(date '+%F %R') short-budget stop $mode $role freeze=$freeze"
    generate=("$python" -IBu "$entry" generate "${inputs[@]}" "${args[@]}" --role "$role" --output "$bundle/$role.json")
    if [[ "$mode" == smoke ]]; then
        "${generate[@]}"
    else
        "${generate[@]}" >/dev/null 2>&1 || {
            status=$?
            echo "[codex] $(date '+%F %R') short-budget stop diagnostic $role exited $status" >&2
            exit "$status"
        }
    fi
done
CUDA_VISIBLE_DEVICES='' "$python" -IBu "$entry" summary "${args[@]}" --base "$bundle/base.json" --rl "$bundle/rl.json" \
    --output "$bundle/summary.json" "${smoke_args[@]}"
mkdir -p "$outdir"
ln "$bundle/summary.json" "$destination"
sha256sum -- "$destination"
