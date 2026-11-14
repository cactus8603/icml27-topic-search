#!/usr/bin/env bash
#SBATCH --job-name=gate-validate
#SBATCH --partition=h200q
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=32G
#SBATCH --time=01:00:00
#SBATCH --output=/work/ChihChungHsu/cactus8603/icml27-topic-search/codex/experiments/gate-optimizer/logs/%x-%j.out
set -euo pipefail

work=/work/ChihChungHsu/cactus8603
repo="$work/icml27-topic-search"
icml="$work/icml"
topic=codex/experiments/gate-optimizer
stage="$topic/round2"
frozen=${1:?Pass the full frozen commit ID}
mode=${2:-}
calibration=9e4f3025621ad9c86ccd43cae25148603bb50954
receipt="$repo/$topic/results/round2-26446/calibration.json"
receipt_sha=28676645e5e7722c4f07ebb084bcb2cfab2492590310bde1b8b9c060f0d511df

[[ "$frozen" =~ ^[0-9a-f]{40}$ && $# -le 2 && ( -z "$mode" || "$mode" == --check-only ) ]]
[[ "$(git -C "$repo" rev-parse --verify "$frozen^{commit}")" == "$frozen" ]]
changes="$(git -C "$repo" status --porcelain --untracked-files=all -- "$topic")"
[[ -z "$changes" ]]
git -C "$repo" diff --exit-code "$frozen" HEAD -- "$topic"
unchanged=("$topic/config.json" "$topic/"{data,model,run,aggregate}.py
           "$stage/config.json" "$stage/"{experiment,judge}.py)
git -C "$repo" diff --exit-code "$calibration" "$frozen" -- "${unchanged[@]}"
files=("${unchanged[@]}" "$stage/PREREG_validation.md" "$stage/validation.py" "$stage/validate.sbatch")
for file in "${files[@]}"; do
    git -C "$repo" cat-file -e "$frozen:$file"
done
read -r actual_sha _ < <(sha256sum "$receipt")
[[ "$actual_sha" == "$receipt_sha" ]]
tree="$(git -C "$repo" rev-parse "$frozen:$topic")"
if [[ "$mode" == --check-only ]]; then
    echo "[codex] validation freeze and calibration receipt checks passed $frozen $tree"
    exit 0
fi

job=${SLURM_JOB_ID:?Submit via Slurm}
export TMPDIR="$icml/tmp" HF_HOME="$icml/hf_home"
export RAYON_NUM_THREADS="$SLURM_CPUS_PER_TASK" OMP_NUM_THREADS="$SLURM_CPUS_PER_TASK"
snapshot="$(mktemp -d "$TMPDIR/gate-validation-$job.XXXXXXXX")"
trap 'rm -r -- "$snapshot"' EXIT
git -C "$repo" archive "$frozen" "${files[@]}" | tar -x -f - -C "$snapshot"
code="$snapshot/$stage"
read -r prereg_sha _ < <(sha256sum "$code/PREREG_validation.md")
echo "[codex] $(TZ=Asia/Taipei date '+%Y-%m-%d %H:%M') gate validation $job commit=$frozen tree=$tree prereg=$prereg_sha"
"$icml/envs/vllm033/bin/python" -B "$code/validation.py" \
    --base-config "$snapshot/$topic/config.json" --config "$code/config.json" \
    --calibration-result "$receipt" --output "$repo/$topic/results/validation-$job" \
    --corpus "$icml/tmp/gate-optimizer-validation-owt-$job" --seconds 3500 \
    --frozen-commit "$frozen" --frozen-tree "$tree" --prereg-sha256 "$prereg_sha"
