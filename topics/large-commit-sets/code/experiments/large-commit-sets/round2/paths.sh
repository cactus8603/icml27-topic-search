# Shared offline paths and verification of the frozen implementation.
repo=/work/ChihChungHsu/cactus8603/icml27-topic-search
shared=/work/ChihChungHsu/cactus8603/icml
topic=codex/experiments/large-commit-sets/round2
base=codex/experiments/large-commit-sets
python="$shared/envs/t2irare/bin/python"
parquet_python="$shared/envs/biasgen/bin/python"
train_data="$shared/data/gsm8k-hf/main/train-00000-of-00001.parquet"
test_data="$shared/data/gsm8k/main/test-00000-of-00001.parquet"
tracked=("$topic" "$base/pilot.py" "$base/judge.py" "$base/read_parquet.py" "$base/thresholds.json")

export TMPDIR="$shared/tmp" HF_HOME="$shared/hf_home"
export HF_MODULES_CACHE="$shared/tmp/large_commit_sets_r2_modules_${SLURM_JOB_ID:?Use sbatch}"
export HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 TOKENIZERS_PARALLELISM=false
export PYTHONUNBUFFERED=1 PYTHONDONTWRITEBYTECODE=1 OPENBLAS_NUM_THREADS=1 TZ=Asia/Taipei

freeze() {
  git -C "$repo" cat-file -e "$1:$topic/PREREG.md"
  [[ -z "$(git -C "$repo" status --porcelain -- "${tracked[@]}")" ]]
  git -C "$repo" diff --quiet "$1" HEAD -- "${tracked[@]}"
  export PILOT_FREEZE_COMMIT="$(git -C "$repo" rev-parse "$1^{commit}")"
  export PILOT_RESULTS_PATH="$repo/$topic/results"
  frozen_work="$(mktemp -d "$TMPDIR/large_commit_sets_r2_${SLURM_JOB_ID}.XXXXXXXX")"
  trap 'rm -r -- "$frozen_work"' EXIT
  git -C "$repo" archive "$PILOT_FREEZE_COMMIT" "${tracked[@]}" | tar -x -C "$frozen_work"
  printf '[codex] %s mode=%s job=%s freeze=%s\n' "$(date '+%F %R')" "$mode" "$SLURM_JOB_ID" "$PILOT_FREEZE_COMMIT"
}
