#!/usr/bin/env bash
# [codex] Install a separate development environment; never change the system driver.
set -euo pipefail
here=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)
repo=$(git -C "$here" rev-parse --show-toplevel)
runtime=${RETRIEVAL_ROOT:-$repo/dl/retrieval3090}
export TMPDIR="$repo/hf_home/tmp" PIP_CACHE_DIR="$repo/hf_home/pip-cache"
mkdir -p "$TMPDIR" "$PIP_CACHE_DIR" "$runtime/envs"
python="$runtime/envs/retrieval3090/bin/python"
if [[ ! -x "$python" ]]; then
    base=${PYTHON3_12:-/ssd7/cactus8603/roco_spring/.conda-stereo/bin/python}
    "$base" -c 'import sys; assert sys.version_info[:2] == (3, 12), "Use Python 3.12"'
    "$base" -m venv "$runtime/envs/retrieval3090"
fi
"$python" -m pip install 'torch==2.6.0' 'torchvision==0.21.0' --index-url https://download.pytorch.org/whl/cu124
requirements="$here/requirements.txt"
[[ ! -f "$here/requirements-lock.txt" ]] || requirements="$here/requirements-lock.txt"
"$python" -m pip install -r "$requirements"
"$python" -m pip check
