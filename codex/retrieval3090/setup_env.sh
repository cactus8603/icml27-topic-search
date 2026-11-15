#!/usr/bin/env bash
# [codex] Install a separate development environment; never change the system driver.
set -euo pipefail
here=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)
repo=$(git -C "$here" rev-parse --show-toplevel)
runtime=${RETRIEVAL_ROOT:-$repo/dl/retrieval3090}
export TMPDIR="$repo/hf_home/tmp" PIP_CACHE_DIR="$repo/hf_home/pip-cache"
mkdir -p "$TMPDIR" "$PIP_CACHE_DIR" "$runtime/envs"
profile=${RETRIEVAL_ENV_PROFILE:-compat26}
case "$profile" in
    compat26)
        env_name=retrieval3090
        torch_version=2.6.0
        vision_version=0.21.0
        wheel_cuda=cu124
        requirements="$here/requirements-lock.txt"
        ;;
    torch214)
        env_name=retrieval3090-torch214
        torch_version=2.14.0
        vision_version=0.29.0
        wheel_cuda=cu126
        requirements="$here/requirements-torch214-lock.txt"
        ;;
    *) echo 'RETRIEVAL_ENV_PROFILE must be compat26 or torch214' >&2; exit 2 ;;
esac
python="$runtime/envs/$env_name/bin/python"
if [[ ! -x "$python" ]]; then
    base=${PYTHON3_12:-/ssd7/cactus8603/roco_spring/.conda-stereo/bin/python}
    "$base" -c 'import sys; assert sys.version_info[:2] == (3, 12), "Use Python 3.12"'
    "$base" -m venv "$runtime/envs/$env_name"
fi
"$python" -m pip install "torch==$torch_version" "torchvision==$vision_version" --index-url "https://download.pytorch.org/whl/$wheel_cuda"
[[ -f "$requirements" ]] || { echo "Missing environment lock: $requirements" >&2; exit 2; }
"$python" -m pip install -r "$requirements"
"$python" -m pip check
