#!/usr/bin/env bash
# [codex] Explicit repaired MLLM family; central four-argument API only.
set -euo pipefail
if [[ $# -ne 4 ]]; then echo 'Usage: run.sh FULL_REF PLAN MODEL OUT' >&2; exit 2; fi
here=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)
repo=$(git -C "$here" rev-parse --show-toplevel)
ref=${1:?full new freeze} plan=${2:?genuinely qualified frozen P2 plan} model=${3:?exact model} out=${4:?new isolated output}
family=$(python3 - "$plan" <<'PY'
import json,sys
print(json.load(open(sys.argv[1]))['family_adapter'])
PY
)
case "$family" in
  core-qwen3-9943159)
    python="$repo/dl/retrieval3090/envs/retrieval3090-torch214/bin/python"
    export PYTHONPATH="$repo/dl/retrieval-matrix/deps/core-transformers457:$repo/dl/retrieval-matrix/deps/core-p0:$repo";;
  qwen2-einops-9943159)
    python="$repo/dl/retrieval3090/envs/retrieval3090/bin/python"
    export PYTHONPATH="$repo/dl/retrieval-matrix-qwen2-runtime-fix/deps:$repo/dl/retrieval-matrix/deps/qwen2-transformers451:$repo/dl/retrieval-matrix/deps/qwen2-flash:$repo/dl/retrieval-matrix/deps/core-p0:$repo";;
  *) echo 'Unsupported fixed repaired MLLM family' >&2; exit 2;;
esac
export HF_HOME="$repo/hf_home" TMPDIR="$repo/hf_home/tmp" TZ=Asia/Taipei HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1
export PYTHONDONTWRITEBYTECODE=1 PYTHONUNBUFFERED=1 TOKENIZERS_PARALLELISM=false OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1
"$python" -B "$here/runner.py" --stage smoke --plan "$plan" --model "$model" --dataset conjunction-dev --out "$out/smoke" --freeze-commit "$ref"
for dataset in conjunction-dev negation-dev cola-gqa cola-paco; do
  "$python" -B "$here/runner.py" --stage index --plan "$plan" --model "$model" --dataset "$dataset" --out "$out/$dataset" --smoke-out "$out/smoke" --freeze-commit "$ref"
done
