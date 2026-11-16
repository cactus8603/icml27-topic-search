# [codex] Qwen2 missing-einops runtime repair

This separate batch restores a required import without editing the frozen Qwen2 producer, model spec or dependency directories. Only einops0.8.2 and its metadata are copied into a new isolated overlay. `overlay.json` records all payload hashes and actual import identities.

The plan builder preserves the original scientific fields exactly and emits REVIEW_REQUIRED. Root must independently review, freeze and commit the new plan and executing source, then schedule the new launcher with a new output directory. CPU preflight is technical evidence only; there is no GPU run or model-effect claim here.

Use the existing Qwen2 Python, prepending only `dl/retrieval-matrix-qwen2-runtime-fix/deps` to the original three isolated dependency roots. Do not put the entire rerank environment on PYTHONPATH. The overlay may also help future separate Qwen2-based batches, but they require their own explicit launcher/plan and review.

Both `embedding` and `mcmr` modes preserve their respective original plan in full. The MCMR path keeps its `MCMR-native-MMEB-v2` phase, full product metadata, model batch1/storage64/query32 and original native top100 scoring. It remains ineligible for strict CORE P0 reproduction.

The central dispatcher interface is `bash central_task.sh NEW_COMMIT NEW_PLAN MODEL NEW_OUT`; this committed wrapper derives the mode from the unchanged plan phase and calls `bash run.sh MODE NEW_COMMIT NEW_PLAN MODEL NEW_OUT`. `NEW_OUT` must be a distinct directory under `dl/retrieval-matrix-qwen2-runtime-fix/runs/`; old failed runs are rejected by the entry point. Analysis remains a separate CPU stage through the same mode entry. Source/dependency checks and all scientific field comparisons run again before production.
