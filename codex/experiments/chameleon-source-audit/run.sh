#!/usr/bin/env bash
set -euo pipefail

freeze=${1:?Pass a frozen commit}
[[ "$#" == 1 && "$freeze" =~ ^[0-9a-f]{40}$ ]]
root=/work/ChihChungHsu/cactus8603
repo="$root/icml27-topic-search"
stage=codex/experiments/chameleon-source-audit
changes=$(git -C "$repo" status --porcelain --untracked-files=all -- "$stage")
[[ -z "$changes" ]]
git -C "$repo" diff --quiet "$freeze" HEAD -- "$stage"
export HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 TZ=Asia/Taipei
export TMPDIR="$root/icml/tmp"
git -C "$repo" show "$freeze:$stage/audit.py" | "$root/icml/envs/inoc/bin/python" -IB - --freeze "$freeze"
