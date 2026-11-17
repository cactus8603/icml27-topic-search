#!/usr/bin/env bash
#SBATCH --job-name=t2ir-formal-blind-replay
#SBATCH --partition=h200q
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=16
#SBATCH --mem=96G
#SBATCH --time=00:30:00
#SBATCH --output=/work/ChihChungHsu/cactus8603/icml/tmp/t2irare/t2ir-blind-replay-%j.out
set -euo pipefail

extension=${1:?Pass the blind replay commit}
freeze=${2:?Pass the formal sampling commit}
shift 2
[[ $extension =~ ^[0-9a-f]{40}$ && $freeze =~ ^[0-9a-f]{40}$ ]]
if [[ ${1:-} != --plan && ${1:-} != --check-only ]]; then
    [[ ${SLURM_JOB_ID:-} =~ ^[0-9]+$ ]]
    quota -g -s
fi
repo=$(git -C "${SLURM_SUBMIT_DIR:-$PWD}" rev-parse --show-toplevel)
stage=codex/experiments/t2i-formal-replay
topic=killtests/t2i-rare-failure
candidate=codex/experiments/t2i-stage1-candidate
probe=codex/experiments/t2i-detector-probe
repair=codex/experiments/t2i-gate-repair
source=b54a86fb05e845e2c11139822beac8846c63c77a
candidate_source=a5d3fb86d489dbdc2b7209179879ba3b1e7ecedb
probe_source=0d86f961a09954cec06434999e63b136bf0a83b1
shared=/work/ChihChungHsu/cactus8603/icml
export TZ=Asia/Taipei HF_HOME="$shared/hf_home" HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 PYTHONUNBUFFERED=1
export TMPDIR="$shared/tmp/t2irare"
export OMP_NUM_THREADS=${SLURM_CPUS_PER_TASK:-4} OPENBLAS_NUM_THREADS=1
bundle=$(mktemp -d -- "$TMPDIR/t2ir-blind-replay.${SLURM_JOB_ID:-plan}.XXXXXXXX")
trap 'rm -r --one-file-system -- "$bundle"' EXIT
export XDG_CACHE_HOME="$bundle/xdg" TORCH_HOME="$bundle/torch"
export TRITON_CACHE_DIR="$bundle/triton" MPLCONFIGDIR="$bundle/mpl"

git -C "$repo" archive "$extension" "$stage/regen_exact.py" | tar -xf - -C "$bundle"
git -C "$repo" archive "$freeze" "$stage/pipeline.py" | tar -xf - -C "$bundle"
git -C "$repo" archive "$source" "$topic/configs" "$topic/scripts" "$topic/slurm/stage1.sbatch" "$repair/review_evidence.py" \
    | tar -xf - -C "$bundle"
git -C "$repo" archive "$candidate_source" "$candidate/run.py" | tar -xf - -C "$bundle"
git -C "$repo" archive "$probe_source" "$probe"/{probe,stage1_check}.py | tar -xf - -C "$bundle"
mv -- "$bundle/$candidate/run.py" "$bundle/$candidate/run_candidate.py"
export PYTHONPATH="$bundle/$candidate:$bundle/$probe"
"$shared/envs/t2irare/bin/python" -B "$bundle/$stage/regen_exact.py" \
    --run "$shared/runs/t2irare/formal-replay/$freeze" --upstream "$bundle/$topic" --source "$source" "$@"
