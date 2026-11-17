#!/bin/bash -l
#SBATCH --job-name=rsd-pair-b
#SBATCH --partition=h200q
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=16
#SBATCH --mem=170G
#SBATCH --time=06:00:00
#SBATCH --output=killtests/rl-sparse-distill/slurm/logs/%x-%A_%a.out

set -euo pipefail
stage=${1:?請指定配對 B 階段}
id=${2:?請指定唯一 run ID}
commit=${3:?請指定完整凍結 commit}
[[ $# = 3 && $id =~ ^[a-zA-Z0-9][a-zA-Z0-9_-]{0,49}$ && $commit =~ ^[0-9a-f]{40}$ ]]
repo=$(cd -- "${SLURM_SUBMIT_DIR:?}" && pwd -P)
[[ $repo == "$(git -C "$repo" rev-parse --show-toplevel)" ]]
topic=killtests/rl-sparse-distill
live="$repo/$topic"
shared="$(dirname -- "$repo")/icml"
python="$shared/envs/vllm033/bin/python"
run="$live/results/pair_b/$id"
source="$run/source/$topic"
job=${SLURM_JOB_ID:?}
export HF_HOME="$shared/hf_home" TMPDIR="$shared/tmp" TRANSFORMERS_OFFLINE=1 HF_HUB_OFFLINE=1
export HF_DATASETS_OFFLINE=1 TOKENIZERS_PARALLELISM=false VLLM_NO_USAGE_STATS=1 DO_NOT_TRACK=1
export VLLM_USE_FLASHINFER_SAMPLER=0 XDG_CACHE_HOME="$shared/tmp/xdg"
export PAIR=B RSD_RUN_ID="$id" TZ=Asia/Taipei PYTHONDONTWRITEBYTECODE=1
export OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1
unset TOKENIZER_PATH
ulimit -f 1048576
printf 'stage=%s run=%s freeze=%s job=%s time=%s\n' "$stage" "$id" "$commit" "$job" "$(date -Is)"

if [[ $stage == prepare ]]; then
    quota -g -s || true
    mkdir -p -- "$live/results/pair_b"
    mkdir -- "$run"
    mkdir -- "$run/source"
    git -C "$repo" archive "$commit" "$topic/configs" "$topic/data/eval.jsonl" "$topic/scripts" \
        "$topic/tests/smoke_cpu.py" "$topic/slurm/pair_b.sbatch" "$topic/PREREG_pair_b.md" |
        tar -xf - -C "$run/source"
    ln -s -- "$live/pylib" "$source/pylib"
    cmp -- "$0" "$source/slurm/pair_b.sbatch"
    "$python" -BI "$source/scripts/pair_b.py" check "$shared"
    "$python" -BI "$source/scripts/pair_b.py" prepare "$shared" "$run" "$commit" "$job"
    printf '%s\n' "$commit" > "$run/prepared"
    exec /bin/true
fi

[[ $(<"$run/prepared") == "$commit" ]]
cmp -- "$0" "$source/slurm/pair_b.sbatch"

side=()
case "$stage" in
    long|short)
        case ${SLURM_ARRAY_TASK_ID:?請使用 --array=0-1%1} in
            0) side=(base) ;;
            1) side=(rl) ;;
            *) exit 2 ;;
        esac
        ;;
    correct|score-long|score-short) ;;
    *) exit 2 ;;
esac

exec "$python" -BI "$source/scripts/pair_b.py" "$stage" "$shared" "$run" "$commit" "$job" "${side[@]}"
