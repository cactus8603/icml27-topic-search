#!/usr/bin/env bash
#SBATCH --job-name=cham-clean-panel
#SBATCH --partition=h200q
#SBATCH --qos=h200q_1g
#SBATCH --gres=gpu:1
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=48G
#SBATCH --time=00:20:00
#SBATCH --output=/work/ChihChungHsu/cactus8603/icml/tmp/cham-clean-panel-%j.log
set -euo pipefail

freeze=${1:?Pass a frozen commit}
phase=${2:?Pass preflight, smoke, or main}
[[ "$freeze" =~ ^[0-9a-f]{40}$ ]]
case "$phase" in
    preflight|smoke) [[ "$#" == 2 ]] ;;
    main) [[ "$#" == 3 && "$3" =~ ^[0-9]+$ ]] ;;
    *) exit 2 ;;
esac
root=/work/ChihChungHsu/cactus8603
shared="$root/icml"
repo="$root/icml27-topic-search"
stage=codex/experiments/chameleon-clean-panel
source_audit=codex/experiments/chameleon-source-audit/audit.py
prior=killtests/chameleon-defense
inputs=("$stage" "$source_audit" "$prior/scripts" "$prior/configs/config.json")
changes=$(git -C "$repo" status --porcelain --untracked-files=all -- "${inputs[@]}")
[[ -z "$changes" ]]
git -C "$repo" diff --quiet "$freeze" HEAD -- "${inputs[@]}"
export HF_HOME="$shared/hf_home" HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 TZ=Asia/Taipei
export TMPDIR="$shared/tmp" OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 TOKENIZERS_PARALLELISM=false
bundle=$(mktemp -d "$TMPDIR/cham-clean-${SLURM_JOB_ID:-check}.XXXXXXXX")
trap 'rm -r --one-file-system -- "$bundle"' EXIT
git -C "$repo" archive "$freeze" "${inputs[@]}" | tar -x -C "$bundle"
args=("$phase" --freeze "$freeze")
if [[ "$phase" == main ]]; then
    args+=(--smoke-job "$3")
fi
"$shared/envs/inoc/bin/python" -IB "$bundle/$stage/measure.py" "${args[@]}"
