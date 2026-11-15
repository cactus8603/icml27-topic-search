#!/usr/bin/env bash
#SBATCH --job-name=fv-asdiv-replication
#SBATCH --partition=h200q
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=16
#SBATCH --mem=96G
#SBATCH --time=02:00:00
#SBATCH --output=/work/ChihChungHsu/cactus8603/icml/tmp/fv-asdiv-replication-%j.log
set -euo pipefail

freeze=${1:?Pass the reviewed ASDiv commit}
instruct_freeze=${2:?Pass the reviewed Instruct warm commit}
[[ "$freeze" =~ ^[0-9a-f]{40}$ && "$instruct_freeze" =~ ^[0-9a-f]{40}$ ]]
check=false
case ${3:-} in
    --check-only) [[ "$#" == 3 && -z "${SLURM_JOB_ID:-}" ]]; check=true; job=check ;;
    *) [[ "$#" == 4 && "${SLURM_JOB_ID:-}" =~ ^[1-9][0-9]*$ && "$3" =~ ^[1-9][0-9]*$ && "$4" =~ ^[0-9a-f]{64}$ ]]
       job=$SLURM_JOB_ID; warm_job=$3; warm_sha=$4 ;;
esac

root=/work/ChihChungHsu/cactus8603
repo="$root/icml27-topic-search"
shared="$root/icml"
topic=codex/experiments/fallible-verifier
stage="$topic/asdiv-replication"
instruct="$topic/instruct-candidate-coverage"
baseline="$topic/gsm1k-replication"
previous="$topic/gsm1k-warm-coverage"
base_freeze=229cdcf22a60e3f030e1e1edf9347294e9584b02
warm_freeze=bc762a190cff60a94bb285d47407516506389425
scoring_freeze=ac3950820be8c7f4f1528ecc553b9cf51aad894b
solver_freeze=20c6a0426f7b75b5eb7efc079638047bbfb6d357
prepared="$shared/data/asdiv-official/prepared-110881d5"
asdiv_sha=110881d5e9a7ba12d2776a30dbebe8ef919f76a81ef39b2a829790ffd2b2b8d8
[[ -z "$(git -C "$repo" status --porcelain --untracked-files=all -- "$stage")" ]]
git -C "$repo" diff --quiet "$freeze" HEAD -- "$stage"

export TZ=Asia/Taipei TMPDIR="$shared/tmp" HF_HOME="$shared/hf_home"
export PYTHONUNBUFFERED=1 OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1 TOKENIZERS_PARALLELISM=false
export HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 HF_HUB_DISABLE_IMPLICIT_TOKEN=1 VLLM_NO_USAGE_STATS=1 DO_NOT_TRACK=1
unset OPENAI_API_KEY OPENAI_BASE_URL HF_TOKEN HUGGING_FACE_HUB_TOKEN HUGGINGFACE_HUB_TOKEN
unset VLLM_USE_V1 VLLM_USE_FLASHINFER_SAMPLER VLLM_ATTENTION_BACKEND
umask 077
bundle=$(mktemp -d "$TMPDIR/fv-asdiv-replication-$job.XXXXXXXX")
trap 'rm -r --one-file-system -- "$bundle"' EXIT
export XDG_CACHE_HOME="$bundle/xdg" TORCHINDUCTOR_CACHE_DIR="$bundle/torch"

git -C "$repo" archive "$freeze" "$stage" | tar -x -C "$bundle"
git -C "$repo" archive "$instruct_freeze" "$instruct/"{generate_instruct,assess}.py | tar -x -C "$bundle"
git -C "$repo" archive "$base_freeze" "$baseline/"{panel,score,solvers,evaluate}.py | tar -x -C "$bundle"
git -C "$repo" archive "$warm_freeze" "$previous/"{generate,measure}.py | tar -x -C "$bundle"
git -C "$repo" archive "$scoring_freeze" "$topic/holdout-evaluation" "$topic/holdout-scoring" \
    "$topic/real_holdout/"{prepare,dev_likelihood}.py "$topic/"{phase3_likelihood,phase3_feasibility,phase3_synthetic}.py \
    "$topic/round2/"{config.json,protocol.py,analysis.py} | tar -x -C "$bundle"
git -C "$repo" archive "$solver_freeze" codex/experiments/rl-a-short-stop/pilot.py \
    killtests/rl-sparse-distill/configs/pairs.json | tar -x -C "$bundle"
solver="$shared/envs/oat-vllm072/bin/python"
judge="$shared/envs/vllm033/bin/python"
cpu="$shared/envs/t2irare/bin/python"
entry="$bundle/$stage"
mkdir "$bundle/public"
cp -- "$prepared/questions.jsonl" "$prepared/prepared.json" "$bundle/public/"
question_args=(--questions "$bundle/public/questions.jsonl" --manifest "$bundle/public/prepared.json")

if [[ "$check" == true ]]; then
    export CUDA_VISIBLE_DEVICES=''
    "$judge" -IB "$entry/test_prepare.py"
    "$cpu" -IB "$entry/test_assay.py"
    "$solver" -IB "$entry/test_generate_oat.py"
    "$solver" -IB "$bundle/$instruct/generate_instruct.py" --mode asdiv-full "${question_args[@]}" --sha256 "$asdiv_sha" --check-only
    "$solver" -IB "$entry/generate_oat.py" "${question_args[@]}" --check-only
    for role in weak strong; do
        "$judge" -IB "$entry/assay.py" score --role "$role" --check-only
    done
    exit 0
fi

test "$(sacct -X -n -P -j "$warm_job" -o JobID,State,ExitCode)" = "$warm_job|COMPLETED|0:0"
warm_path="$shared/runs/fallible-verifier/instruct-candidate-coverage/$warm_job/summary.json"
CUDA_VISIBLE_DEVICES='' "$judge" -IB "$bundle/$instruct/assess.py" verify-gate \
    --summary "$warm_path" --sha256 "$warm_sha" --job "$warm_job" --freeze "$instruct_freeze"
CUDA_VISIBLE_DEVICES='' "$judge" -IB - "$shared/tmp/fv-instruct-coverage-$warm_job.log" "$warm_sha" <<'PY'
from pathlib import Path
import re
import sys

matches = re.findall(r'"summary_sha256": "([0-9a-f]{64})"', Path(sys.argv[1]).read_text())
if matches != [sys.argv[2]]:
    raise ValueError("Warm qualification does not match the completed job log")
PY

output="$shared/runs/fallible-verifier/asdiv-replication/$job"
mkdir -p "$(dirname "$output")"
mkdir "$output"
printf '[codex] %s ASDiv Instruct generate job=%s freeze=%s warm_job=%s\n' "$(date '+%F %R')" "$job" "$freeze" "$warm_job"
"$solver" -IB "$bundle/$instruct/generate_instruct.py" --mode asdiv-full "${question_args[@]}" --sha256 "$asdiv_sha" \
    --freeze "$freeze" --warm-gate-sha256 "$warm_sha" --output-dir "$output/instruct"
printf '[codex] %s ASDiv Oat generate\n' "$(date '+%F %R')"
"$solver" -IB "$entry/generate_oat.py" "${question_args[@]}" --freeze "$freeze" \
    --warm-gate-sha256 "$warm_sha" --output-dir "$output/oat"
instruct_sha=$(sha256sum "$output/instruct/receipt.json" | cut -d ' ' -f 1)
oat_sha=$(sha256sum "$output/oat/receipt.json" | cut -d ' ' -f 1)
CUDA_VISIBLE_DEVICES='' "$judge" -IB "$entry/assay.py" assemble --freeze "$freeze" --prepared-dir "$prepared" \
    --base-receipt "$output/instruct/receipt.json" --base-receipt-sha256 "$instruct_sha" \
    --oat-receipt "$output/oat/receipt.json" --oat-receipt-sha256 "$oat_sha" \
    --warm-gate-sha256 "$warm_sha" --output-dir "$output/panel"
coverage_sha=$(sha256sum "$output/panel/coverage.json" | cut -d ' ' -f 1)
panel_args=(--freeze "$freeze" --panel-dir "$output/panel" --coverage-sha256 "$coverage_sha")
for role in weak strong; do
    printf '[codex] %s ASDiv score %s\n' "$(date '+%F %R')" "$role"
    "$judge" -IB "$entry/assay.py" score --role "$role" "${panel_args[@]}" --output-dir "$output/scoring"
done
CUDA_VISIBLE_DEVICES='' "$cpu" -IB "$entry/assay.py" evaluate "${panel_args[@]}" --score-dir "$output/scoring" \
    --output "$output/summary.json"
