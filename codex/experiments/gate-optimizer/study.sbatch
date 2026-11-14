#!/usr/bin/env bash
#SBATCH --job-name=gate-optim
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
study="$repo/$topic"
icml=/work/ChihChungHsu/cactus8603/icml
frozen="${1:-}"
mode="${2:-}"

fail() {
    echo "[codex] gate-optimizer freeze check failed: $*" >&2
    exit 2
}

if [[ ! "$frozen" =~ ^([a-f0-9]{40}|[a-f0-9]{64})$ || $# -gt 2 ||
    ( -n "$mode" && "$mode" != "--check-only" ) ]]; then
    fail "usage: study.sbatch FULL_COMMIT_SHA [--check-only]"
fi
resolved="$(git -C "$repo" rev-parse --verify "$frozen^{commit}")" || fail "unknown commit"
[[ "$resolved" == "$frozen" ]] || fail "expected the resolved commit ID"
for file in PREREG.md config.json run.py aggregate.py data.py model.py study.sbatch; do
    git -C "$repo" cat-file -e "$frozen:$topic/$file" || fail "missing frozen file $file"
done
changes="$(git -C "$repo" status --porcelain=v1 --untracked-files=all -- "$topic")" || fail "git status failed"
[[ -z "$changes" ]] || fail "topic has staged, unstaged or untracked changes: $changes"
git -C "$repo" diff --quiet "$frozen" HEAD -- "$topic" || fail "HEAD topic differs from frozen commit"
git -C "$repo" diff --quiet "$frozen" -- "$topic" || fail "working topic differs from frozen commit"
tree="$(git -C "$repo" rev-parse --verify "$frozen:$topic")" || fail "missing frozen topic tree"
if [[ "$mode" == "--check-only" ]]; then
    echo "[codex] freeze check passed $frozen $tree"
    exit 0
fi

python="$icml/envs/vllm033/bin/python"
export TMPDIR="$icml/tmp"
export HF_HOME="$icml/hf_home"
export PYTHONDONTWRITEBYTECODE=1
export RAYON_NUM_THREADS="${SLURM_CPUS_PER_TASK}"
export OMP_NUM_THREADS="${SLURM_CPUS_PER_TASK}"
results="$study/results/$SLURM_JOB_ID"
snapshot="$(mktemp -d "$TMPDIR/gate-optimizer-code-$SLURM_JOB_ID.XXXXXXXX")"
git -C "$repo" archive "$frozen" "$topic" | tar -x -f - -C "$snapshot"
frozen_study="$snapshot/$topic"
read -r prereg_sha256 _ < <(sha256sum "$frozen_study/PREREG.md")
read -r config_sha256 _ < <(sha256sum "$frozen_study/config.json")
chmod -R a-w "$snapshot"

echo "[codex] $(TZ=Asia/Taipei date +'%Y-%m-%d %H:%M') gate-optimizer $SLURM_JOB_ID commit=$frozen tree=$tree prereg=$prereg_sha256 config=$config_sha256"
status=0
"$python" -B "$frozen_study/run.py" --config "$frozen_study/config.json" --output "$results" \
    --frozen-commit "$frozen" --frozen-tree "$tree" --prereg-sha256 "$prereg_sha256" \
    --corpus "$icml/tmp/gate-optimizer-owt-v1-$SLURM_JOB_ID" --seconds 6500 || status=$?
if [[ -f "$results/run.json" ]]; then
    "$python" -B "$frozen_study/aggregate.py" "$results"
fi
exit "$status"
