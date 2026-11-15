#!/usr/bin/env bash
set -euo pipefail

repo=/work/ChihChungHsu/cactus8603/icml27-topic-search
shared=/work/ChihChungHsu/cactus8603/icml
stage=codex/experiments/lambda-language-local
panel=codex/experiments/lambda-language-panel
panel_freeze=ac12f76c30adb0b491a598af7b20272bb8439113
importer=codex/experiments/lambda-author-import/import_scores.py
importer_freeze=bcbcc012d6f7f55b58e2bacafc9e1906c3741521
author_freeze=6964328411bf0ae49de47fa1bdfcfe8f94a75cce
questions=data/trait_questions/extract/spanish.json
python="$shared/envs/lambda-language-fasttext093/bin/python"

if [[ "$#" != 2 || ! "$1" =~ ^[0-9a-f]{40}$ || ! "$2" =~ ^--(check-only|run)$ ]]; then
    printf 'Usage: run.sh FULL_LOCAL_FREEZE --check-only|--run\n' >&2
    exit 2
fi
freeze=$1
phase=$2
git -C "$repo" diff --quiet "$freeze" HEAD -- "$stage"
[[ -z "$(git -C "$repo" status --porcelain --untracked-files=all -- "$stage")" ]]

export TZ=Asia/Taipei TMPDIR="$shared/tmp" CUDA_VISIBLE_DEVICES=''
export OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 MKL_NUM_THREADS=1
export HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 HF_HUB_DISABLE_IMPLICIT_TOKEN=1
unset OPENAI_API_KEY OPENAI_BASE_URL HF_TOKEN HUGGING_FACE_HUB_TOKEN HUGGINGFACE_HUB_TOKEN
umask 077
snapshot=$(mktemp -d "$TMPDIR/lambda-language-local.XXXXXXXX")
trap 'rm -r --one-file-system -- "$snapshot"' EXIT
git -C "$repo" archive "$freeze" "$stage" | tar -x -C "$snapshot"
git -C "$repo" archive "$panel_freeze" "$panel" | tar -x -C "$snapshot"
git -C "$repo" archive "$importer_freeze" "$importer" | tar -x -C "$snapshot"
git -C "$shared/ext/shifting_the_gradient" archive "$author_freeze" "$questions" | tar -x -C "$snapshot"

cpu=(timeout --kill-after=1s 29s "$python" -IBu)
if [[ "$phase" == --check-only ]]; then
    "${cpu[@]}" "$snapshot/$panel/test_diagnostic.py"
    "${cpu[@]}" "$snapshot/$stage/test_replay.py"
    "${cpu[@]}" "$snapshot/$panel/panel.py" --questions "$snapshot/$questions" --freeze "$panel_freeze" --check-only
else
    [[ -z "${SLURM_JOB_ID:-}" ]]
    "${cpu[@]}" "$snapshot/$stage/replay.py" --questions "$snapshot/$questions" --freeze "$freeze" --panel-freeze "$panel_freeze"
fi
