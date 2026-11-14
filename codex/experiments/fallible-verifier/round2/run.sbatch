#!/usr/bin/env bash
#SBATCH --job-name=fv2-cache
#SBATCH --partition=h200q
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=48G
#SBATCH --time=01:00:00
set -euo pipefail

root=/work/ChihChungHsu/cactus8603
shared="$root/icml"
topic="$root/icml27-topic-search/codex/experiments/fallible-verifier/round2"
freeze=${1:?Pass reviewed freeze commit}
prior_job=${2:?Pass the prerequisite terminal job ID}
[[ "$prior_job" =~ ^[1-9][0-9]*$ && "$prior_job" != "$SLURM_JOB_ID" ]]
run="$topic/results/$SLURM_JOB_ID"
python="$shared/envs/vllm033/bin/python"
weak="$shared/models/Qwen3-0.6B"
strong="$shared/hf_home/hub/models--Qwen--Qwen3-32B-AWQ/snapshots/0499c3ac83fdef8810b907a23894ba91e95eddd8"
export TZ=Asia/Taipei PYTHONDONTWRITEBYTECODE=1 PYTHONUNBUFFERED=1
export TMPDIR="$shared/tmp" HF_HOME="$shared/hf_home"
export XDG_CACHE_HOME="$shared/tmp/fv2-cache-$SLURM_JOB_ID"
export HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 TOKENIZERS_PARALLELISM=false
export OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 VLLM_NO_USAGE_STATS=1
export FV2_PREVIOUS_JOB_ID="$prior_job"

{
  for attempt in 1 2 3 4 5 6; do
    previous_state=$(sacct -X -n -P -j "$prior_job" --format=JobIDRaw,State |
      awk -F '|' -v job="$prior_job" '$1 == job {split($2, words, /[ +]/); print words[1]}') || previous_state=
    case "$previous_state" in
      COMPLETED|FAILED|CANCELLED|TIMEOUT|NODE_FAIL|OUT_OF_MEMORY|PREEMPTED|BOOT_FAIL|DEADLINE|REVOKED)
        break ;;
      *) if [[ "$attempt" == 6 ]]; then
           printf '[codex] Previous job %s is not verified terminal: %s\n' "$prior_job" "$previous_state" >&2
           exit 1
         fi
         sleep 5 ;;
    esac
  done
  export FV2_PREVIOUS_JOB_STATE="$previous_state"
  "$python" -B "$topic/protocol.py" --freeze "$freeze" --run-dir "$run" \
    --prior-dir "$shared/tmp/codex-fallible-verifier-08f42e1" \
    --solver-dir "$shared/tmp/codex-fallible-verifier-round2-solver-08f42e1"
  "$python" -B "$topic/judge.py" --freeze "$freeze" --run-dir "$run" --role weak --model-path "$weak"
  "$python" -B "$topic/judge.py" --freeze "$freeze" --run-dir "$run" --role strong --model-path "$strong"
} 2>&1 | awk '{sub(/^\[codex\] /, ""); print "[codex] " strftime("%Y-%m-%d %H:%M") " " $0; fflush()}'
