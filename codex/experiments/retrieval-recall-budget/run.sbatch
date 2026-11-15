#!/usr/bin/env bash
#SBATCH --job-name=cr-recall-budget
#SBATCH --partition=h200q
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=48G
#SBATCH --time=00:40:00
#SBATCH --output=/work/ChihChungHsu/cactus8603/icml/tmp/cr-recall-budget-%j.log
set -euo pipefail

freeze=${1:?Pass the frozen commit}
[[ "$freeze" =~ ^[0-9a-f]{40}$ ]]
[[ "$#" == 1 || "$#" == 2 && "$2" == --check-only ]]
root=/work/ChihChungHsu/cactus8603
repo="$root/icml27-topic-search"
shared="$root/icml"
stage=codex/experiments/retrieval-recall-budget
prior=codex/experiments/retrieval-anchor-rerank
rerank=codex/experiments/retrieval-rerank
conjunction=codex/experiments/retrieval-conjunction
topic=killtests/compositional-retrieval
source=2daedf522c6d64b860277657d3067b61da3a0f2b
executables=("$stage/score.py" "$stage/run.sbatch" "$stage/test_score.py")
[[ -z "$(git -C "$repo" status --porcelain --untracked-files=all -- "${executables[@]}")" ]]
git -C "$repo" diff --quiet "$freeze" HEAD -- "${executables[@]}"
[[ "$(sacct -X -n -P -j 26601 --format=JobID,State,ExitCode,JobName%48)" == '26601|COMPLETED|0:0|cr-anchor-rerank' ]]

export TZ=Asia/Taipei TMPDIR="$shared/tmp" HF_HOME="$shared/hf_home" PYTHONUNBUFFERED=1
export HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1
umask 077
bundle=$(mktemp -d "$TMPDIR/cr-recall-budget-${SLURM_JOB_ID:-check}.XXXXXXXX")
trap 'rm -r --one-file-system -- "$bundle"' EXIT
git -C "$repo" archive "$source" "$topic/scripts" "$topic/configs/params_headroom.json" \
    "$prior"/{prepare.py,evaluate.py,panel.json} "$rerank"/{prepare.py,panel.json} \
    "$conjunction"/{prepare.py,panel.json} | tar -x -C "$bundle"
git -C "$repo" archive "$freeze" "$stage" | tar -x -C "$bundle"

export EXP="$bundle/$topic"
python="$shared/envs/vllm033/bin/python"
entry="$bundle/$stage"
if [[ "${2:-}" == --check-only ]]; then
    export CUDA_VISIBLE_DEVICES=''
    "$python" -IB "$entry/test_score.py"
    "$python" -IB "$entry/score.py" --preflight
else
    job=${SLURM_JOB_ID:?Submit through Slurm}
    [[ "$job" =~ ^[1-9][0-9]*$ ]]
    "$python" -IB "$entry/score.py" --score --freeze "$freeze"
fi
