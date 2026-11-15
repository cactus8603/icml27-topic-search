#!/usr/bin/env bash
#SBATCH --job-name=t2ir-formal-replay
#SBATCH --partition=h200q
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=16
#SBATCH --mem=96G
#SBATCH --time=04:00:00
#SBATCH --output=/work/ChihChungHsu/cactus8603/icml/tmp/t2irare/t2ir-formal-%A_%a.out
set -euo pipefail

freeze=${1:?Pass the formal workflow commit}
shift
[[ $freeze =~ ^[0-9a-f]{40}$ ]]
repo=$(git -C "${SLURM_SUBMIT_DIR:-$PWD}" rev-parse --show-toplevel)
pipeline=codex/experiments/t2i-formal-replay
candidate=codex/experiments/t2i-stage1-candidate
probe=codex/experiments/t2i-detector-probe
repair=codex/experiments/t2i-gate-repair
topic=killtests/t2i-rare-failure
source=b54a86fb05e845e2c11139822beac8846c63c77a
candidate_source=a5d3fb86d489dbdc2b7209179879ba3b1e7ecedb
probe_source=0d86f961a09954cec06434999e63b136bf0a83b1
gate=codex/experiments/t2i-gate-b-original-replay
gate_precision=codex/experiments/t2i-gate-b-precision/run.py
gate_freeze=10d42bdf86534a44b47c03d56ea5e1dd749cdbac
gate_generator=f54cb310a921966440754d42f65f86d9f565dc62
gate_reference=37e6ae9a63cd3035822dccca2acf90ef12efec93
gate_precision_source=2b1c7f64b8b9013a9a32f8353024c1d6f82b6e53
git -C "$repo" diff --quiet "$freeze" HEAD -- "$pipeline/pipeline.py" "$pipeline/run.sbatch" "$pipeline/test_pipeline.py"

shared=/work/ChihChungHsu/cactus8603/icml
export TZ=Asia/Taipei HF_HOME="$shared/hf_home" HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 PYTHONUNBUFFERED=1
export TMPDIR="$shared/tmp/t2irare"
export OMP_NUM_THREADS=${SLURM_CPUS_PER_TASK:-4} OPENBLAS_NUM_THREADS=1
if [[ ${SLURM_JOB_ID:-} =~ ^[0-9]+$ ]]; then
    quota -g -s || true
fi
bundle=$(mktemp -d -- "$TMPDIR/t2ir-formal.${SLURM_JOB_ID:-check}.XXXXXXXX")
trap 'rm -r --one-file-system -- "$bundle"' EXIT
export XDG_CACHE_HOME="$bundle/xdg" TORCH_HOME="$bundle/torch"
export TRITON_CACHE_DIR="$bundle/triton" MPLCONFIGDIR="$bundle/mpl"
git -C "$repo" archive "$freeze" "$pipeline" | tar -xf - -C "$bundle"
git -C "$repo" archive "$source" "$topic/configs" "$topic/scripts" "$topic/slurm/stage1.sbatch" \
    "$repair/review_evidence.py" | tar -xf - -C "$bundle"
git -C "$repo" archive "$candidate_source" "$candidate/run.py" | tar -xf - -C "$bundle"
git -C "$repo" archive "$probe_source" "$probe"/{probe,stage1_check}.py | tar -xf - -C "$bundle"
mv -- "$bundle/$candidate/run.py" "$bundle/$candidate/run_candidate.py"
export PYTHONPATH="$bundle/$candidate:$bundle/$probe"

gate_archive="$bundle/gate-b"
mkdir -- "$gate_archive"
git -C "$repo" archive "$gate_freeze" "$gate/run.py" | tar -xf - -C "$gate_archive"
git -C "$repo" archive "$gate_generator" "$topic/scripts/generate.py" | tar -xf - -C "$gate_archive"
git -C "$repo" archive "$gate_precision_source" "$gate_precision" | tar -xf - -C "$gate_archive"
git -C "$repo" archive "$gate_reference" "$topic"/configs/{criteria,detectors,sampling,models,prompts}.json \
    "$topic"/scripts/{clip_precision_probe,common,detectors,evidence,verify_gate_b}.py | tar -xf - -C "$gate_archive"

python="$shared/envs/t2irare/bin/python"
"$python" -B "$bundle/$pipeline/pipeline.py" --repo "$repo" --upstream "$bundle/$topic" \
    --gate-archive "$gate_archive" --freeze "$freeze" "$@"
