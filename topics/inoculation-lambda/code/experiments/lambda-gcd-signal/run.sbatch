#!/usr/bin/env bash
#SBATCH --job-name=lambda-gcd-signal
#SBATCH --partition=h200q
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --time=00:30:00
#SBATCH --output=/work/ChihChungHsu/cactus8603/icml/tmp/lambda-gcd-signal-%j.log
set -euo pipefail
umask 077

shared=/work/ChihChungHsu/cactus8603/icml
repo=/work/ChihChungHsu/cactus8603/icml27-topic-search
stage=codex/experiments/lambda-gcd-signal
pilot=codex/experiments/lambda-offline-confirmation
gcd=codex/experiments/lambda-gcd
staged=codex/experiments/lambda-gcd-run
topic=killtests/inoculation-lambda
material=64eb9427f4bb14adfeff9bc25fb38b65ba728032
runner=b54a86fb05e845e2c11139822beac8846c63c77a
pilot_helpers=a2e2e7f716a9949025e3cf15ec100f184763f42d
gradient_core=b401effd9f560401e3ec9bb5ac0de70bb3275b4e
author=6964328411bf0ae49de47fa1bdfcfe8f94a75cce
freeze=${1:?Pass the full frozen Git commit}
mode=${2:-gradients}
[[ "$freeze" =~ ^[0-9a-f]{40}$ ]]
[[ -z "$(git -C "$repo" status --porcelain --untracked-files=all -- "$stage")" ]]
git -C "$repo" diff --quiet "$freeze" HEAD -- "$stage"

case "$mode" in
    --check-only) [[ "$#" == 2 ]]; mode=check ;;
    --join)
        [[ "$#" == 7 && "$3" =~ ^[1-9][0-9]*$ && "$5" =~ ^[1-9][0-9]*$ && "$6" =~ ^[0-9a-f]{40}$ ]]
        git -C "$repo" diff --quiet "$pilot_helpers" "$6" -- "$pilot/PROTOCOL.md"
        mode=join
        ;;
    gradients)
        [[ "$#" == 1 ]]
        job=${SLURM_JOB_ID:?Submit the fixed gradients through Slurm}
        [[ "$job" =~ ^[1-9][0-9]*$ ]]
        ;;
    *) exit 2 ;;
esac

export TZ=Asia/Taipei TMPDIR="$shared/tmp" HF_HOME="$shared/hf_home" PYTHONDONTWRITEBYTECODE=1
export HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 HF_HUB_DISABLE_IMPLICIT_TOKEN=1
export OMP_NUM_THREADS="${SLURM_CPUS_PER_TASK:-2}" TOKENIZERS_PARALLELISM=false
unset OPENAI_API_KEY OPENAI_BASE_URL HF_TOKEN HUGGING_FACE_HUB_TOKEN HUGGINGFACE_HUB_TOKEN
snapshot=$(mktemp -d "$TMPDIR/lambda-gcd-signal-${SLURM_JOB_ID:-local}.XXXXXXXX")
trap 'rm -r --one-file-system -- "$snapshot"' EXIT
git -C "$repo" archive "$freeze" "$stage" | tar -x -C "$snapshot"
git -C "$repo" archive "$material" "$gcd" | tar -x -C "$snapshot"
git -C "$repo" archive "$runner" "$staged/run_cell.py" | tar -x -C "$snapshot"
git -C "$repo" archive "$pilot_helpers" "$pilot/pilot.py" | tar -x -C "$snapshot"
git -C "$repo" archive "$gradient_core" "$topic/scripts/compute_lambda.py" | tar -x -C "$snapshot"
mkdir "$snapshot/author"
git -C "$shared/ext/shifting_the_gradient" archive "$author" grad_shift scripts/finetune_lora.py | tar -x -C "$snapshot/author"
python="$shared/envs/inoc-phaseb-author-6964328/env/bin/python"
entry="$snapshot/$stage/diagnostic.py"

case "$mode" in
    check) "$python" -IBu "$entry" check --source "$snapshot/author" ;;
    join)
        "$python" -IBu "$entry" join --freeze "$freeze" --job "$3" --gradient-sha "$4" \
            --behavior-job "$5" --behavior-freeze "$6" --behavior-sha "$7"
        ;;
    gradients) "$python" -IBu "$entry" gradients --source "$snapshot/author" --freeze "$freeze" --job "$job" ;;
esac
