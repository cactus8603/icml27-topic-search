# [codex] P0 Qwen2 fixed-baseline reproduction

This independent topic completes GME2 and VLM2Vec-V2.0 cells. Freeze this file,
the execution plan, adapter loader, and all source/dependency identities before
GPU work. The complete public COCO/NegBench/Flickr input protocols and published
comparison targets are inherited from the referenced P0 plan. No custom method
or sealed conjunction test data is used.

GME uses CORE's released Qwen2 wrapper, processor, i2t embedding function,
last-token pooling, native BF16 normalization, FlashAttention2, original CSV
order, native BF16 similarities, per-row torch.topk and released pytrec scoring.
The only fixed memory adaptation is image batch4/text batch32, with the same
synthetic invariance gate as P0. Failures remain explicit; no batch, precision,
attention, prompt, or processor fallback is allowed after observing outcomes.

VLM2Vec means the V2.0 rank16 DoRA checkpoint and its pinned Qwen2-VL-2B base.
CORE's generic release omits adapter application. The disclosed correction
attaches the adapter to the complete base with PEFT, verifies every adapter
key/shape/value, and uses the upstream inference merge operation. The resulting
full BF16 model is evaluated through CORE's unchanged Qwen2 embedding wrapper
and processor. CPU meta checks must prove all420 tensors, including140 DoRA
magnitude vectors, match both CORE's evaluation class and VLM2Vec's native
class. A bare base or V1 checkpoint cannot substitute for V2.

The V2 saved image-processor `size` uses obsolete min_pixels/max_pixels keys.
Initialization passes explicit shortest_edge/longest_edge keys with the same
CORE min/max pixel values, following VLM2Vec's own processor loading pattern.
This is a configuration-schema compatibility repair; the effective resize
limits are unchanged from the released CORE protocol.

The isolated environment is torch2.6.0+cu124, transformers4.51.3,
tokenizers0.21.4, PEFT0.15.2, qwen-vl-utils0.0.8 and official FlashAttention
2.7.4.post1 cp312/cu12/torch2.6/cxx11abiFALSE. This follows CORE's compatible
Qwen2/GME path. The newer VLM2Vec repository lists transformers4.52.3;
the execution plan discloses this difference. CPU preflight and synthetic GPU
smoke establish compatibility, without changing evaluation settings.

All six model/dataset cells are retained. The existing P0 gate applies:
at least two distinct published benchmark columns must match within one
percentage point before later method work. Unresolved cells cannot pass.
The implementation emits raw receipts before independent CPU analysis. No
research results have been read for this independent topic before its freeze.
