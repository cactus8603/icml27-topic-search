#!/usr/bin/env bash
#SBATCH --job-name=lambda-gcd-phm
#SBATCH --partition=h200q
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=48G
#SBATCH --time=00:30:00
#SBATCH --output=/work/ChihChungHsu/cactus8603/icml/tmp/lambda-gcd-phm-%j.log
set -euo pipefail
umask 077

root=/work/ChihChungHsu/cactus8603
repo="$root/icml27-topic-search"
shared="$root/icml"
stage=codex/experiments/lambda-gcd-family-design/phm
python="$shared/envs/inoc-phaseb-author-6964328/env/bin/python"
freeze=${1:?Pass the full frozen Git commit}
mode=${2:?Use --check-only, readiness, or score}
[[ "$freeze" =~ ^[0-9a-f]{40}$ ]]
[[ -z "$(git -C "$repo" status --porcelain --untracked-files=all -- "$stage")" ]]
git -C "$repo" diff --quiet "$freeze" HEAD -- "$stage"

case "$mode" in
    --check-only) [[ "$#" == 2 || "$#" == 6 ]] ;;
    readiness) [[ "$#" == 2 ]] ;;
    score) [[ "$#" == 6 ]] ;;
    *) exit 2 ;;
esac
panel=()
if [[ "$#" == 6 ]]; then
    [[ "$5" =~ ^[0-9a-f]{64}$ && "$6" =~ ^[0-9a-f]{64}$ ]]
    panel=(--design "$3" --calibration "$4" --design-sha "$5" --calibration-sha "$6")
fi

export TZ=Asia/Taipei TMPDIR="$shared/tmp" HF_HOME="$shared/hf_home" PYTHONDONTWRITEBYTECODE=1
export HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 HF_HUB_DISABLE_IMPLICIT_TOKEN=1
export OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 TOKENIZERS_PARALLELISM=false
unset OPENAI_API_KEY OPENAI_BASE_URL HF_TOKEN HUGGING_FACE_HUB_TOKEN HUGGINGFACE_HUB_TOKEN
bundle=$(mktemp -d "$TMPDIR/lambda-gcd-phm-${SLURM_JOB_ID:-local}.XXXXXXXX")
trap 'rm -r --one-file-system -- "$bundle"' EXIT
git -C "$repo" archive "$freeze" "$stage" | tar -xf - -C "$bundle"
entry="$bundle/$stage/phm.py"

if [[ "$mode" == --check-only ]]; then
    export CUDA_VISIBLE_DEVICES=''
    "$python" -IBu "$bundle/$stage/test_phm.py"
    "$python" -IBu "$entry" check "${panel[@]}"
else
    job=${SLURM_JOB_ID:?Submit GPU modes through Slurm}
    [[ "$job" =~ ^[1-9][0-9]*$ ]]
    "$python" -IBu "$entry" "$mode" "${panel[@]}" --freeze "$freeze" --job "$job" \
        --output "$shared/runs/lambda-gcd-family-phm/$job/$mode.json"
fi
