#!/usr/bin/env bash
#SBATCH --job-name=lam_phasea_v2
#SBATCH --time=06:00:00
#SBATCH --partition=h200q
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=96G
#SBATCH --output=/work/ChihChungHsu/cactus8603/icml27-topic-search/killtests/inoculation-lambda/logs/%x_%j.out
set -euo pipefail

if [[ $# != 4 || ! $1 =~ ^(gate|grant|pg)$ || ! $2 =~ ^[A-Za-z0-9][A-Za-z0-9_-]*$ || ! $3 =~ ^[a-f0-9]{40}$ || -z $4 || $4 == *$'\n'* ]]; then
    printf 'Usage: sbatch phasea_v2/run.sbatch {gate|grant|pg} RUN_ID FROZEN_COMMIT REASON\n' >&2
    exit 2
fi
stage=$1
run_id=$2
commit=$3
reason=$4
source_root=${SLURM_SUBMIT_DIR:-$PWD}
topic=killtests/inoculation-lambda
work=/work/ChihChungHsu/cactus8603/icml
export TMPDIR="$work/tmp" HF_HOME="$work/hf_home" HF_HUB_OFFLINE=1
py="$work/envs/inoc/bin/python"
results="$source_root/results/phasea_v2/$run_id"
job=${SLURM_JOB_ID:?Run this launcher through Slurm}

if [[ $stage == gate ]]; then
    repo=$(git -C "$source_root" rev-parse --show-toplevel)
    if [[ $source_root != "$repo/$topic" ]] || ! git -C "$repo" cat-file -e "$commit^{commit}"; then
        printf 'Submit from the clean, frozen inoculation-lambda directory.\n' >&2
        exit 2
    fi
    mkdir -p "$(dirname "$results")"
    mkdir "$results"
    git -C "$repo" archive "$commit" "$topic/PREREG.md" "$topic/scripts" "$topic/configs" \
        "$topic/phasea_v2/estimator.py" "$topic/phasea_v2/integration.patch" "$topic/phasea_v2/run.sbatch" | tar -xf - -C "$results"
    snapshot="$results/$topic"
    "$py" -BI - "$results/run_v2.json" "$run_id" "$commit" "$job" "$reason" "$snapshot/phasea_v2/run.sbatch" <<'PY'
import hashlib
import json
import sys

path, run_id, commit, job, reason, launcher = sys.argv[1:]
with open(launcher, "rb") as stream:
    launcher_sha256 = hashlib.file_digest(stream, "sha256").hexdigest()
with open(path, "x", encoding="utf-8") as stream:
    json.dump(dict(run_id=run_id, commit=commit, gate_job=job, reason=reason, launcher_sha256=launcher_sha256), stream, indent=2)
PY
else
    snapshot="$results/$topic"
    cd "$snapshot"
    "$py" -BI - "$results" "$commit" <<'PY'
import json
import sys
from pathlib import Path

root = Path(sys.argv[1])
report = json.loads((root / "aggregate_phaseA_v2_gate.json").read_text())
receipt = json.loads((root / "run_v2.json").read_text())
if (receipt["commit"] != sys.argv[2] or report.get("estimator") != "equal_prompt_token_fold_v2" or report.get("reference_group") != "gate"
        or report["gate1"]["status"] != "PASS" or report.get("origin_gate_job") != receipt["gate_job"]):
    sys.exit("Reference execution requires this run's Phase A v2 Gate 1 PASS.")
PY
    mkdir "$results/$stage"
fi

dest="$results"
if [[ $stage != gate ]]; then
    dest="$results/$stage"
fi
printf '[codex] %s Phase A v2 %s job %s commit %s output %s reason %s\n' \
    "$(TZ=Asia/Taipei date '+%Y-%m-%d %H:%M')" "$stage" "$job" "$commit" "$dest" "$reason" >> "$source_root/logs/RUNLOG.md"

if [[ $stage == gate ]]; then
    patch --batch --forward --fuzz=0 -d "$results" -p1 < "$snapshot/phasea_v2/integration.patch"
    cd "$snapshot"
    for script in compute_lambda aggregate; do
        help=$("$py" -BI "scripts/$script.py" --help)
        if [[ $help != *--phase-a-v2* ]]; then
            printf 'The frozen snapshot lacks Phase A v2 %s integration.\n' "$script" >&2
            exit 2
        fi
    done
    chmod -R a-w "$snapshot"
fi

compute() {
    local config=$1
    local output=$2
    shift 2
    "$py" -B scripts/compute_lambda.py --phase-a-v2 --config "configs/$config.yaml" --out "$output" "$@"
}

traits=(evil sycophantic hallucinating pirate spanish allcaps optimistic)
case "$stage" in
    gate)
        for t in "${traits[@]}"; do
            if ! compute "grant_${t}_qwen7b" "$results/grant_${t}_qwen7b_fd" --finite_diff; then
                # Missing or failed receipt makes the frozen gate UNTESTABLE.
                "$py" -BI scripts/aggregate.py --phase a --phase-a-v2 --results-dir "$results"
                exit 1
            fi
        done
        ;;
    grant)
        for t in "${traits[@]}"; do
            compute "grant_${t}_qwen7b" "$results/grant/grant_${t}_qwen7b"
        done
        ;;
    pg)
        for config in pg_qwen7b_idmedical pg_qwen7b_idsports; do
            compute "$config" "$results/pg/$config"
        done
        if "$py" -BI scripts/check_cached_model.py Qwen/Qwen2.5-14B-Instruct; then
            for config in pg_qwen14b_idmedical pg_qwen14b_idsports; do
                compute "$config" "$results/pg/$config"
            done
        else
            printf 'SKIP 14B: incomplete offline snapshot\n'
        fi
        if "$py" -BI scripts/check_cached_model.py allenai/Olmo-3-7B-Instruct; then
            compute pg_olmo7b_idmedical "$results/pg/pg_olmo7b_idmedical"
        else
            printf 'SKIP OLMo: incomplete offline snapshot\n'
        fi
        ;;
esac
"$py" -BI scripts/aggregate.py --phase a --phase-a-v2 --results-dir "$results" --reference-group "$stage"
