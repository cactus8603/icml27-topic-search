#!/usr/bin/env bash
#SBATCH --job-name=fv-gsm1k-replication
#SBATCH --partition=h200q
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=16
#SBATCH --mem=96G
#SBATCH --time=02:00:00
#SBATCH --output=/work/ChihChungHsu/cactus8603/icml/tmp/fv-gsm1k-replication-%j.log
set -euo pipefail

freeze=${1:?Pass the reviewed full GSM1k freeze commit}
[[ "$freeze" =~ ^[0-9a-f]{40}$ ]]
case ${2:-} in
    --check-only) [[ "$#" == 2 && -z "${SLURM_JOB_ID:-}" ]]; job=check ;;
    "") [[ "$#" == 1 && "${SLURM_JOB_ID:?Submit through Slurm}" =~ ^[1-9][0-9]*$ ]]; job=$SLURM_JOB_ID ;;
    *) exit 2 ;;
esac

root=/work/ChihChungHsu/cactus8603
repo="$root/icml27-topic-search"
shared="$root/icml"
topic=codex/experiments/fallible-verifier
stage="$topic/gsm1k-replication"
scoring_freeze=ac3950820be8c7f4f1528ecc553b9cf51aad894b
solver_freeze=20c6a0426f7b75b5eb7efc079638047bbfb6d357
release=bc09569d09a614b9b530edc7f076fb214ac10493
source="$shared/data/gsm1k-official/$release/test.arrow"
source_sha=bf7b848f2badbdcb138d6fbc1f246a1902337ce4c00cc079e4f35baf8ed10764
prepared_sha=2f81442c52e6b8c9a07ba0ee94b0798ed71225a7b7d5cab5f82b0b5b475c6daf

[[ -z "$(git -C "$repo" status --porcelain --untracked-files=all -- "$stage")" ]]
git -C "$repo" diff --quiet "$freeze" HEAD -- "$stage"
export TZ=Asia/Taipei TMPDIR="$shared/tmp" HF_HOME="$shared/hf_home"
export PYTHONDONTWRITEBYTECODE=1 PYTHONUNBUFFERED=1 TOKENIZERS_PARALLELISM=false
export HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 HF_HUB_DISABLE_IMPLICIT_TOKEN=1
export OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1 VLLM_NO_USAGE_STATS=1 DO_NOT_TRACK=1
unset OPENAI_API_KEY OPENAI_BASE_URL HF_TOKEN HUGGING_FACE_HUB_TOKEN HUGGINGFACE_HUB_TOKEN
unset VLLM_USE_V1 VLLM_USE_FLASHINFER_SAMPLER VLLM_ATTENTION_BACKEND
umask 077
bundle=$(mktemp -d "$TMPDIR/fv-gsm1k-replication-$job.XXXXXXXX")
trap 'rm -r --one-file-system -- "$bundle"' EXIT

inputs=("$topic/holdout-evaluation" "$topic/holdout-scoring" "$topic/real_holdout/prepare.py"
        "$topic/real_holdout/dev_likelihood.py" "$topic/"{phase3_likelihood,phase3_feasibility,phase3_synthetic}.py
        "$topic/round2/"{config.json,protocol.py,analysis.py})
git -C "$repo" archive "$freeze" "$stage" | tar -x -C "$bundle"
git -C "$repo" archive "$scoring_freeze" "${inputs[@]}" | tar -x -C "$bundle"
git -C "$repo" archive "$solver_freeze" codex/experiments/rl-a-short-stop/pilot.py \
    killtests/rl-sparse-distill/configs/pairs.json | tar -x -C "$bundle"
entry="$bundle/$stage"
gpu_score="$shared/envs/vllm033/bin/python"
gpu_solver="$shared/envs/oat-vllm072/bin/python"
cpu_eval="$shared/envs/t2irare/bin/python"
export XDG_CACHE_HOME="$bundle/xdg-solver" TORCHINDUCTOR_CACHE_DIR="$bundle/torch-solver"

if [[ "$job" == check ]]; then
    run="$bundle/check"
else
    run="$shared/runs/fallible-verifier/gsm1k-replication/$job"
fi
mkdir -p "$(dirname "$run")"
mkdir "$run"
printf '[codex] %s GSM1k prepare job=%s freeze=%s\n' "$(date '+%F %R')" "$job" "$freeze"
CUDA_VISIBLE_DEVICES='' "$gpu_score" -IB "$entry/panel.py" prepare --source "$source" \
    --source-sha256 "$source_sha" --output-dir "$run/prepared"
test "$(sha256sum "$run/prepared/prepared.json" | cut -d ' ' -f 1)" = "$prepared_sha"

# A solver receives the question file and its manifest, without the answer file.
solver_inputs="$bundle/solver-input"
mkdir "$solver_inputs"
cp -- "$run/prepared/questions.jsonl" "$run/prepared/prepared.json" "$solver_inputs/"
solver_args=(--questions "$solver_inputs/questions.jsonl" --prepared-sha256 "$prepared_sha")
if [[ "$job" == check ]]; then
    export CUDA_VISIBLE_DEVICES=''
    "$gpu_solver" -IB "$entry/test_solvers.py"
    "$gpu_score" -IB "$entry/test_panel.py"
    "$gpu_score" -IB "$entry/test_score.py"
    "$cpu_eval" -IB "$entry/test_evaluate.py"
    for role in base oat; do
        "$gpu_solver" -IB "$entry/solvers.py" --role "$role" --check-only "${solver_args[@]}"
    done
    for role in weak strong; do
        "$gpu_score" -IB "$entry/score.py" --role "$role" --check-only
    done
    exit 0
fi

for role in base oat; do
    printf '[codex] %s GSM1k generate %s\n' "$(date '+%F %R')" "$role"
    "$gpu_solver" -IB "$entry/solvers.py" --role "$role" "${solver_args[@]}" --freeze "$freeze" \
        --output-dir "$run/$role"
done
CUDA_VISIBLE_DEVICES='' "$gpu_score" -IB - "$run" "$job" "$freeze" <<'PY'
import json
from pathlib import Path
import sys

directory, job, freeze = Path(sys.argv[1]), sys.argv[2], sys.argv[3]
for role in ("base", "oat"):
    receipt = json.loads((directory / role / "receipt.json").read_bytes())
    if (receipt["job"], receipt["freeze"], receipt["role"]) != (job, freeze, role):
        raise ValueError("Generated solver receipt does not belong to the active run")
PY
base_sha=$(sha256sum "$run/base/receipt.json" | cut -d ' ' -f 1)
oat_sha=$(sha256sum "$run/oat/receipt.json" | cut -d ' ' -f 1)
CUDA_VISIBLE_DEVICES='' "$gpu_score" -IB "$entry/panel.py" assemble \
    --prepared-dir "$run/prepared" --prepared-sha256 "$prepared_sha" \
    --base-receipt "$run/base/receipt.json" --base-receipt-sha256 "$base_sha" \
    --oat-receipt "$run/oat/receipt.json" --oat-receipt-sha256 "$oat_sha" --output-dir "$run/panel"
coverage_sha=$(sha256sum "$run/panel/coverage.json" | cut -d ' ' -f 1)

export XDG_CACHE_HOME="$bundle/xdg-score" TORCHINDUCTOR_CACHE_DIR="$bundle/torch-score"
for role in weak strong; do
    printf '[codex] %s GSM1k score %s\n' "$(date '+%F %R')" "$role"
    "$gpu_score" -IB "$entry/score.py" --role "$role" --freeze "$freeze" --panel-dir "$run/panel" \
        --coverage-sha256 "$coverage_sha" --output-dir "$run/scoring"
done
CUDA_VISIBLE_DEVICES='' "$cpu_eval" -IB "$entry/evaluate.py" --job "$job" --freeze "$freeze" \
    --panel-dir "$run/panel" --coverage-sha256 "$coverage_sha" --score-dir "$run/scoring" --output "$run/summary.json"
