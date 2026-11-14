#!/usr/bin/env bash
#SBATCH --job-name=gate-cal
#SBATCH --partition=h200q
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=32G
#SBATCH --time=02:00:00
#SBATCH --output=/work/ChihChungHsu/cactus8603/icml27-topic-search/codex/experiments/gate-optimizer/logs/%x-%j.out
set -euo pipefail

repo=/work/ChihChungHsu/cactus8603/icml27-topic-search
topic=codex/experiments/gate-optimizer
calibration="$topic/round2"
first=10792d355d258bd16b3f0ea5ec57a35bba7078fe
icml=/work/ChihChungHsu/cactus8603/icml
frozen="${1:-}"
mode="${2:-}"

fail() {
    echo "[codex] calibration freeze check failed: $*" >&2
    exit 2
}

if [[ ! "$frozen" =~ ^[a-f0-9]{40}$ || $# -gt 2 ||
    ( -n "$mode" && "$mode" != "--check-only" ) ]]; then
    fail "usage: calibrate.sbatch FULL_COMMIT_SHA [--check-only]"
fi
resolved="$(git -C "$repo" rev-parse --verify "$frozen^{commit}")" || fail "unknown commit"
[[ "$resolved" == "$frozen" ]] || fail "expected resolved commit ID"
base_files=("$topic/data.py" "$topic/model.py" "$topic/run.py" "$topic/config.json" "$topic/aggregate.py")
git -C "$repo" diff --quiet "$first" "$frozen" -- "${base_files[@]}" ||
    fail "round-one Python or config dependency differs from the frozen source"
round_files=("$calibration/PREREG.md" "$calibration/config.json" "$calibration/experiment.py"
             "$calibration/judge.py" "$calibration/calibrate.sbatch")
for file in "${base_files[@]}" "${round_files[@]}"; do
    git -C "$repo" cat-file -e "$frozen:$file" || fail "missing frozen file $file"
done
changes="$(git -C "$repo" status --porcelain=v1 --untracked-files=all -- "$topic")" || fail "git status failed"
[[ -z "$changes" ]] || fail "topic has staged, unstaged or untracked changes: $changes"
git -C "$repo" diff --quiet "$frozen" HEAD -- "$topic" || fail "HEAD topic differs from frozen commit"
git -C "$repo" diff --quiet "$frozen" -- "$topic" || fail "working topic differs from frozen commit"
tree="$(git -C "$repo" rev-parse --verify "$frozen:$topic")"
if [[ "$mode" == "--check-only" ]]; then
    echo "[codex] calibration freeze check passed $frozen $tree"
    exit 0
fi

python="$icml/envs/vllm033/bin/python"
export TMPDIR="$icml/tmp"
export HF_HOME="$icml/hf_home"
export PYTHONDONTWRITEBYTECODE=1
export RAYON_NUM_THREADS="${SLURM_CPUS_PER_TASK}"
export OMP_NUM_THREADS="${SLURM_CPUS_PER_TASK}"
output="$repo/$topic/results/round2-$SLURM_JOB_ID"
snapshot="$(mktemp -d "$TMPDIR/gate-optimizer-calibration-code-$SLURM_JOB_ID.XXXXXXXX")"
git -C "$repo" archive "$frozen" "${base_files[@]}" "${round_files[@]}" | tar -x -f - -C "$snapshot"
code="$snapshot/$calibration"
read -r prereg_sha256 _ < <(sha256sum "$code/PREREG.md")
read -r config_sha256 _ < <(sha256sum "$code/config.json")
chmod -R a-w "$snapshot"

echo "[codex] $(TZ=Asia/Taipei date +'%Y-%m-%d %H:%M') gate calibration $SLURM_JOB_ID commit=$frozen tree=$tree prereg=$prereg_sha256 config=$config_sha256"
status=0
"$python" -B "$code/experiment.py" --base-config "$snapshot/$topic/config.json" --config "$code/config.json" \
    --output "$output" --corpus "$icml/tmp/gate-optimizer-round2-owt-$SLURM_JOB_ID" --seconds 6500 \
    --frozen-commit "$frozen" --frozen-tree "$tree" --prereg-sha256 "$prereg_sha256" || status=$?
if [[ -f "$output/run.json" ]]; then
    "$python" -B "$code/judge.py" "$output"
fi
exit "$status"
