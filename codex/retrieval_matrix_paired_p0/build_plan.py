"""[codex] Full published SugarCrepe/SugarCrepe++ qualification for three CLIC variants."""
import argparse
import copy
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
from codex.retrieval_matrix_native_p0.evidence import digest, entry, read, sha, write
from codex.retrieval_matrix_paired_p0 import acquire, metrics

HERE = Path(__file__).resolve().parent
TARGETS = {
    "clic_b32": {"sugarcrepe": {"add_ITT": .845, "replace_ITT": .840, "swap_ITT": .737},
        "sugarcrepe_pp": {"replace_ITT": .756, "replace_TOT": .601, "swap_ITT": .611, "swap_TOT": .279}, "table": "Table1, CLIC-LAION ViT-B/32"},
    "clic_l14": {"sugarcrepe": {"add_ITT": .877, "replace_ITT": .856, "swap_ITT": .713},
        "sugarcrepe_pp": {"replace_ITT": .790, "replace_TOT": .591, "swap_ITT": .598, "swap_TOT": .263}, "table": "Table3 first CLIC-LAION group, OpenAI CLIP ViT-L/14"},
    "clic_clips_l14": {"sugarcrepe": {"add_ITT": .967, "replace_ITT": .924, "swap_ITT": .857},
        "sugarcrepe_pp": {"replace_ITT": .843, "replace_TOT": .719, "swap_ITT": .730, "swap_TOT": .494}, "table": "Table3 final CLIC-LAION group, CLIPS ViT-L/14"}}
OWN = ("acquire.py", "data.py", "models.py", "metrics.py", "qualification.py", "runner.py",
       "build_plan.py", "test_paired.py", "run.sh", "central_task.sh", "PREREG.md", "SOURCES.md")


def build(key):
    registry = read(ROOT/"codex/retrieval_matrix_extra/registry.json")
    baseline = registry["models"][key]
    model = copy.deepcopy(baseline)
    model.update(family="CLIP", baseline_model_spec_sha256=digest(baseline),
        batches={"image": 1, "text": 1, "cross": 1},
        native_protocol={"name": "CLIC official SugarCrepe paired native protocol", "text_cleaner": "raw-caption-to-author-tokenizer",
            "precision": "FP32, unchanged source load_encode_img/load_encode_text F.normalize",
            "model_factory": "OpenCLIP2.31.0 official HF configuration and exact fixed checkpoint bytes",
            "tokenizer": "CLIC clips.CLIPS_Tokenizer CustomTokenizer (bos1,eos2,trailing101,pad0,length80)"
                if key == "clic_clips_l14" else "official OpenCLIP tokenizer from the fixed HF configuration, length77",
            "baseline_retrieval_protocol_preserved_by_digest": digest(baseline["native_protocol"]),
            "batch_override": "Original paired functions call each image and caption separately; earlier generic acquisition profile4/32 is not used",
            "disclosed_adaptation": "Pinned local paths replace mutable HF references only; frozen Torch2.6cu124/OpenCLIP2.31 rather than author Torch2.2.2cu121; TF32 explicitly disabled."})
    audit = read(acquire.BASE/"data-audit.json")
    paper = entry(ROOT/"dl/retrieval-matrix-extra/sources/clic-paper.html")
    benchmarks = {}
    for name in ("sugarcrepe", "sugarcrepe_pp"):
        pp = name == "sugarcrepe_pp"
        categories = acquire.CATEGORIES[2:] if pp else acquire.CATEGORIES
        total = sum(acquire.EXPECTED[category] for category in categories)
        canonical = "SugarCrepe++" if pp else "SugarCrepe"
        ds = {"benchmark_id": canonical, "format": "sugarcrepe-pp-official" if pp else "sugarcrepe-official",
            "subsets": audit["subsets"][name], "category_order": list(categories),
            "expected_subsets": {category: acquire.EXPECTED[category] for category in categories},
            "image_receipt": audit["image_receipt"], "expected_images": total,
            "expected_texts": total*(3 if pp else 2), "image_count_semantics": "one image embedding per original paired row, without deduplication; unique physical images reported separately",
            "official_revision": audit[name+"_revision"], "raw_caption_policy": "no pre_caption30, rewriting, lowercasing or truncation before the author's tokenizer"}
        descriptors = {}
        for metric, target in TARGETS[key][name].items():
            group, direction = metric.split("_")
            parts = [category for category in categories if category.startswith(group+"_")]
            descriptors[metric] = {"direction": direction, "metric": "macro mean of complete-subset strict-comparison accuracies",
                "denominator": sum(acquire.EXPECTED[category] for category in parts), "subsets": parts,
                "subset_denominators": {category: acquire.EXPECTED[category] for category in parts},
                "aggregation": "unweighted arithmetic mean across listed subsets, not pooled accuracy",
                "target": target, "absolute_tolerance": .01,
                "published": {"variant": key, "table": TARGETS[key]["table"],
                    "url": "https://arxiv.org/html/2505.24424v2", "source": paper}}
        benchmarks[name] = {"canonical_benchmark": canonical, "benchmark_id": canonical,
            "task": "paired-caption-comparison", "data": ds, "score_profile": copy.deepcopy(metrics.PROFILE),
            "primary_metrics": descriptors, "protocol_status": "OFFICIAL_RELEASE_PAIRED_REQUIRES_INDEPENDENT_REVIEW",
            "disclosed_differences": [model["native_protocol"]["disclosed_adaptation"],
                "Encode each original image and caption at batch1, cache normalized FP32 vectors, then replay the exact scalar source formulas on GPU. Encoder calls are grouped by modality rather than interleaved per source row; no deduplication or candidate selection.",
                "Source main enables cudnn.benchmark; it remains True. Original exact GPU package/kernel versions and table-run training seed are not fully documented.",
                "SugarCrepe++ has official edits to original positive/negative strings and some identical-caption cases; preserve all raw rows. It is not created here by joining SugarCrepe with caption2.",
                "CLIPS uses the author's CustomTokenizer branch and OpenCLIP model, correcting the earlier generic acquisition alias route; that earlier technical receipt is not native qualification."]}
    model["published_targets"] = [{"benchmark": row["benchmark_id"], "metrics": TARGETS[key][name],
        "table": TARGETS[key]["table"], "source_file": paper["path"], "source_sha256": paper["sha256"],
        "reported_only": True, "checkpoint_identity_verified": True} for name, row in benchmarks.items()]
    native = read(ROOT/"codex/retrieval_matrix_native_p0/p0-siglip2.json")
    sources = dict(native["source_files"])
    sources.update({"codex/retrieval_matrix_paired_p0/"+name: sha(HERE/name) for name in OWN})
    for relative in ("codex/retrieval_matrix/models.py", "codex/retrieval_matrix_extra/adapters.py",
                     "codex/retrieval_matrix_extra/preflight.py", "codex/retrieval_matrix_extra/registry.json"):
        sources[relative] = sha(ROOT/relative)
    root = Path(model["sources"]["clic"]["root"])
    return {"schema": 2, "owner": "[codex]", "phase": "P0_NATIVE", "state": "NOT_RUN",
        "purpose": "P0_OFFICIAL_QUALIFICATION", "model_key": key, "model_spec": model, "benchmarks": benchmarks,
        "qualification": {"required_distinct_benchmarks": 2, "counting_unit": "SugarCrepe and SugarCrepe++, never their category subsets or ITT/TOT directions",
            "requires_separate_reviewer": True, "requires_independent_array_recomputation": True,
            "all_declared_benchmarks_retained": True},
        "settings": {"image_batch": 1, "text_batch": 1, "seed": 1, "precision": "float32", "allow_tf32": False,
            "quantization": False, "offload": False, "smoke_cosine_min": .9995, "smoke_max_abs": .01},
        "dependencies": native["dependencies"], "source_files": sources,
        "external_sources": {"clic_paired_evaluator": entry(root/"eval/general_eval.py"),
            "clic_encode_helpers": entry(root/"eval/eval_utils.py"), "clic_model_factory": entry(root/"models.py"),
            "clic_custom_tokenizer": entry(root/"clips/tokenizer.py"), "clic_custom_vocabulary": entry(root/"clips/vocab.txt"),
            "sugarcrepe_source_receipt": audit["source"], "dataset_audit": entry(acquire.BASE/"data-audit.json"),
            "sugarcrepe_pp_source_api": entry(ROOT/"dl/retrieval-matrix/data/sugarcrepe_pp/source_api.json"),
            "paper": paper}}


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", required=True, choices=tuple(TARGETS))
    parser.add_argument("--out", type=Path)
    args = parser.parse_args()
    out = args.out or HERE/("p0-"+args.model+".json")
    write(out, build(args.model))
    print(str(out.resolve()))
