#!/usr/bin/env bash
# Check that a host has every file dev-3090 reads (see DATA.md). Read-only; prints one line per item.
#   bash check_host.sh            # quick: existence, sizes, image count, imports
#   bash check_host.sh --hash     # also SHA-256 of the pinned files (~250 MB read)
set -uo pipefail

ICML=/work/ChihChungHsu/cactus8603/icml
HUB=$ICML/hf_home/hub
fail=0
ok()   { printf 'ok    %s\n' "$*"; }
bad()  { printf 'MISS  %s\n' "$*"; fail=1; }

declare -A PINNED=(
    [data/coco2017-anchor-rerank/manifest.json]=8f43eedbcef31fe5057c26705e97bf9f213f162612d9a15b53da11bb9bacf78d
    [data/coco2017-rerank/manifest.json]=df51d103caa1ec7968d46254b5757789e7f834e1f3a11dae1105b2f8c63ccd3f
    [data/coco2017/annotations_trainval2017.zip]=113a836d90195ee1f884e704da6304dfaaecff1f023f49b6ca93c4aaae470268
    [runs/retrieval-anchor-rerank/26601/summary.json]=817c6ab72bc4fa2d106debf4dbfe85066859c4767dd9605c1016c5aa98de5dfa
)
for rel in "${!PINNED[@]}"; do
    path=$ICML/$rel
    if [[ ! -f $path ]]; then bad "$rel"; continue; fi
    if [[ ${1:-} == --hash && $(sha256sum "$path" | cut -c1-64) != "${PINNED[$rel]}" ]]; then
        bad "$rel (SHA-256 differs)"
    else
        ok "$rel"
    fi
done

images=$(find "$ICML/data/coco2017-anchor-rerank/images" -maxdepth 1 -name '*.jpg' 2>/dev/null | wc -l)
[[ $images == 8192 ]] && ok "8192 pool JPEGs" || bad "pool JPEGs: found $images, need 8192"

check_model() {  # repo revision files...
    local dir=$HUB/models--${1//\//--}/snapshots/$2 f
    shift 2
    for f in "$@"; do
        [[ -s $dir/$f ]] || { bad "$dir/$f (missing or dangling symlink)"; return; }
    done
    ok "$(basename "$(dirname "$(dirname "$dir")")") $(du -shL "$dir" | cut -f1)"
}
check_model kbora/CS-CLIP afd6e90f39fe49137fe15d6a462d994eae4811f5 open_clip_pytorch_model.bin
check_model Alibaba-NLP/core-emb-2b cc9870ea92d64ebd5f8f9fc368d1120453564158 config.json chat_template.jinja \
    pytorch_model.bin.index.json pytorch_model-00001-of-00002.bin pytorch_model-00002-of-00002.bin tokenizer.json \
    tokenizer_config.json preprocessor_config.json video_preprocessor_config.json
check_model Alibaba-NLP/core-reranker-2b 64e1a36cd6ecace459a4c7cd8f7d0ef5d6fee30b config.json chat_template.jinja \
    model.safetensors tokenizer.json tokenizer_config.json preprocessor_config.json video_preprocessor_config.json

py=$ICML/envs/vllm033/bin/python
# Like the frozen headroom.py: the extra site (open_clip) goes last so vllm033's torch/torchvision win.
if [[ -x $py ]] && out=$("$py" -c \
    'import sys; sys.path.append("'"$ICML"'/envs/t2irare_extra_cscomp"); import numpy, torch, transformers, ijson, PIL, open_clip; print(torch.__version__, torch.version.cuda, torch.cuda.is_available(), torch.cuda.get_device_name(0) if torch.cuda.is_available() else "-")' 2>&1); then
    ok "python env: $out"
else
    bad "python env vllm033 + t2irare_extra_cscomp: ${out:-missing}"
fi
command -v nvidia-smi >/dev/null && ok "driver $(nvidia-smi --query-gpu=driver_version,name,memory.total --format=csv,noheader 2>/dev/null | head -1)"
free -g | awk '/Mem:/ {print ($2 >= 32 ? "ok   " : "WARN ") "  host RAM " $2 " GiB (want >= 32)"}'
df -h "$ICML" | awk 'NR==2 {print "info  free disk under icml: " $4}'
exit $fail
