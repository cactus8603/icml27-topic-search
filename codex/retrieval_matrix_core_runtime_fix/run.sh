#!/usr/bin/env bash
# [codex] Same inference protocol, separately committed API compatibility repair.
set -euo pipefail
here=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)
repo=$(git -C "$here" rev-parse --show-toplevel)
ref=${1:?full repair freeze}
plan=${2:?frozen repaired plan}
model=${3:?original model key}
out=${4:?new isolated output directory}
python="$repo/dl/retrieval3090/envs/retrieval3090-torch214/bin/python"
export PYTHONPATH="$repo/dl/retrieval-matrix/deps/core-transformers457:$repo/dl/retrieval-matrix/deps/core-p0"
export TZ=Asia/Taipei TMPDIR="$repo/hf_home/tmp" HF_HOME="$repo/hf_home"
export HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 PYTHONDONTWRITEBYTECODE=1 PYTHONUNBUFFERED=1
export TOKENIZERS_PARALLELISM=false OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1
"$python" -B "$repo/codex/retrieval_matrix/check_dependencies.py" --plan "$plan"
phase=$("$python" -c 'import json,sys; print(json.load(open(sys.argv[1]))["phase"])' "$plan")
if [[ "$phase" == P0 ]]; then
  "$python" -B "$here/runner.py" --mode embedding --stage smoke --plan "$plan" --model "$model" --dataset coco --out "$out/smoke" --freeze-commit "$ref"
  for dataset in coco negbench flickr; do
    "$python" -B "$here/runner.py" --mode embedding --stage index --plan "$plan" --model "$model" --dataset "$dataset" --out "$out/$dataset" --smoke-out "$out/smoke" --freeze-commit "$ref"
  done
elif [[ "$phase" == MCMR-native-MMEB-v2 ]]; then
  "$python" -B "$here/runner.py" --mode mcmr --stage smoke --plan "$plan" --model "$model" --out "$out/smoke" --freeze-commit "$ref"
  "$python" -B "$here/runner.py" --mode mcmr --stage index --plan "$plan" --model "$model" --out "$out/mcmr" --smoke-out "$out/smoke" --freeze-commit "$ref"
else
  echo "Unsupported repaired launch phase" >&2
  exit 2
fi
