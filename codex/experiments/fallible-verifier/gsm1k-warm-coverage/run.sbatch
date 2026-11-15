#!/usr/bin/env bash
#SBATCH --job-name=fv-gsm1k-warm-coverage
#SBATCH --partition=h200q
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --time=00:30:00
#SBATCH --output=/work/ChihChungHsu/cactus8603/icml/tmp/fv-gsm1k-warm-coverage-%j.log
set -euo pipefail

freeze=${1:?Pass the full warm coverage freeze}
[[ "$freeze" =~ ^[0-9a-f]{40}$ ]]
[[ "$#" == 1 || "$#" == 2 && "$2" == --check-only ]]
root=/work/ChihChungHsu/cactus8603
repo="$root/icml27-topic-search"
shared="$root/icml"
topic=codex/experiments/fallible-verifier
stage="$topic/gsm1k-warm-coverage"
baseline="$topic/gsm1k-replication"
base_freeze=229cdcf22a60e3f030e1e1edf9347294e9584b02
scoring_freeze=ac3950820be8c7f4f1528ecc553b9cf51aad894b
solver_freeze=20c6a0426f7b75b5eb7efc079638047bbfb6d357
prepared="$shared/runs/fallible-verifier/gsm1k-replication/26669/prepared"
[[ -z "$(git -C "$repo" status --porcelain --untracked-files=all -- "$stage")" ]]
git -C "$repo" diff --quiet "$freeze" HEAD -- "$stage"

export TZ=Asia/Taipei TMPDIR="$shared/tmp" HF_HOME="$shared/hf_home"
export PYTHONUNBUFFERED=1 OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1 TOKENIZERS_PARALLELISM=false
export HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 HF_HUB_DISABLE_IMPLICIT_TOKEN=1 VLLM_NO_USAGE_STATS=1 DO_NOT_TRACK=1
unset OPENAI_API_KEY OPENAI_BASE_URL HF_TOKEN HUGGING_FACE_HUB_TOKEN HUGGINGFACE_HUB_TOKEN VLLM_USE_V1
umask 077
job=${SLURM_JOB_ID:-check}
bundle=$(mktemp -d "$TMPDIR/fv-gsm1k-warm-coverage-$job.XXXXXXXX")
trap 'rm -r --one-file-system -- "$bundle"' EXIT
export XDG_CACHE_HOME="$bundle/xdg" TORCHINDUCTOR_CACHE_DIR="$bundle/torch"

git -C "$repo" archive "$freeze" "$stage" | tar -x -C "$bundle"
git -C "$repo" archive "$base_freeze" "$baseline/"{panel,score,solvers}.py | tar -x -C "$bundle"
git -C "$repo" archive "$scoring_freeze" "$topic/holdout-evaluation" "$topic/holdout-scoring" \
    "$topic/real_holdout/"{prepare,dev_likelihood}.py "$topic/"{phase3_likelihood,phase3_feasibility,phase3_synthetic}.py \
    "$topic/round2/"{config.json,protocol.py,analysis.py} | tar -x -C "$bundle"
git -C "$repo" archive "$solver_freeze" codex/experiments/rl-a-short-stop/pilot.py \
    killtests/rl-sparse-distill/configs/pairs.json | tar -x -C "$bundle"

solver="$shared/envs/oat-vllm072/bin/python"
judge="$shared/envs/vllm033/bin/python"
entry="$bundle/$stage"
mkdir "$bundle/public"
cp -- "$prepared/questions.jsonl" "$prepared/prepared.json" "$bundle/public/"
CUDA_VISIBLE_DEVICES='' "$solver" -IB "$entry/generate.py" prepare --questions "$bundle/public/questions.jsonl" \
    --output-dir "$bundle/warm"
warm_sha=$(sha256sum "$bundle/warm/warm.json" | cut -d ' ' -f 1)
args=(--bundle "$bundle/warm" --warm-sha256 "$warm_sha")
if [[ "${2:-}" == --check-only ]]; then
    export CUDA_VISIBLE_DEVICES=''
    "$solver" -IB "$entry/test_generate.py"
    "$judge" -IB "$entry/test_coverage.py"
    for profile in cap1536 compact768; do
        "$solver" -IB "$entry/generate.py" generate --profile "$profile" "${args[@]}" --check-only
    done
    "$judge" -IB "$entry/measure.py" "${args[@]}" --check-only
    exit 0
fi

[[ "$job" =~ ^[1-9][0-9]*$ ]]
output="$shared/runs/fallible-verifier/gsm1k-warm-coverage/$job"
mkdir -p "$(dirname "$output")"
mkdir "$output"
for profile in cap1536 compact768; do
    printf '[codex] %s GSM1k warm generate %s\n' "$(date '+%F %R')" "$profile"
    "$solver" -IB "$entry/generate.py" generate --profile "$profile" "${args[@]}" --freeze "$freeze" \
        --output-dir "$output/$profile"
done
CUDA_VISIBLE_DEVICES='' "$judge" -IB "$entry/measure.py" "${args[@]}" --freeze "$freeze" --job "$job" --run-dir "$output"
