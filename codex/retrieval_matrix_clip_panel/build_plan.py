"""[codex] Build CLIP P2 from real native P0 evidence and audited COCO dev."""
import argparse
import copy
import json
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[2]
HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT))
from codex.retrieval_matrix_native_p0 import qualification as Q
from codex.retrieval_matrix_panel import data_audit
from codex.retrieval_matrix_clip_panel import contracts


def datasets(development):
    manifest = Q.read(development/"manifest.json")
    if manifest["conjunction"]["queries"] < 1000:
        raise ValueError("Formal conjunction development requires at least 1000 queries")
    result = {}
    for name, folder, meta in (("conjunction-dev", "dev", "conjunction"), ("negation-dev", "negation", "negation")):
        path = development/folder
        for file, expected in manifest[meta]["outputs"].items():
            if Q.sha(path/file) != expected:
                raise ValueError("Development metadata changed")
        count = manifest[meta]["queries"]
        result[name] = {"format": "conjunction", "split": "dev", "gallery": Q.entry(path/"gallery.json"),
            "queries": Q.entry(path/"queries.jsonl"), "images_receipt": Q.entry(development/"dev/images-receipt.jsonl"),
            "images_dir": str(development/"dev/images"), "expected_gallery": 20000,
            "expected_queries": count, "expected_eligible": count,
            "scope": "formal conjunction development" if folder == "dev" else "negation diagnostic only; insufficient 1000-query power"}
    cola = ROOT/"dl/retrieval-matrix/data/cola"
    for subset, gallery, queries in (("gqa", 2589, 320), ("paco", 7921, 400)):
        result["cola-"+subset] = {"format": "cola_single", "split": "public", "subset": subset,
            "parquet": Q.entry(cola/"data"/f"singleobjects_{subset}.parquet"),
            "vocabulary": Q.entry(cola/"data"/f"singleobjects_{subset}_labels.json"),
            "images_receipt": Q.entry(cola/"file_receipt.jsonl"), "image_root": str(cola),
            "allowed_shared_image_roots": [str(ROOT/"dl/retrieval3090/data/coco2017/val2017"),
                                           str(ROOT/"dl/claude-panel/conj-panel-v1/dev/images")],
            "expected_gallery": gallery, "expected_queries": queries, "expected_eligible": queries,
            "use": "fixed baseline test only; no fitting or method selection"}
    return result


def build(base_path, receipts, audit_path):
    base = Q.read(base_path)
    Q.validate_plan(base)
    if base.get("state") != "NOT_RUN" or base.get("phase") != "P0_NATIVE":
        raise ValueError("CLIP P2 must extend its frozen native caption-retrieval P0 plan")
    if not receipts or set(receipts) != {base["model_key"]}:
        raise ValueError("The exact native P0 model set must be independently qualified")
    qualified = {key: Q.entry(path) for key, path in receipts.items()}
    # Admit before opening development metadata; no fabricated PASS placeholders.
    candidate = {"phase": contracts.PHASE, "retriever_family": "CLIP",
        "models": {base["model_key"]: base["model_spec"]}, "qualification": qualified,
        "dependencies": copy.deepcopy(base["dependencies"]),
        "parent_p0_plan": Q.entry(base_path), "parent_native_settings": copy.deepcopy(base["settings"]),
        "settings": {"main_k": 100, "top_ks": [50, 100, 200], "block_queries": 64,
            "batch_invariance_cosine_min": base["settings"]["smoke_cosine_min"],
            "batch_invariance_max_abs": base["settings"]["smoke_max_abs"],
            "seed": base["settings"]["seed"], "allow_tf32": False}}
    contracts.admit_models(candidate)
    development = ROOT/"dl/retrieval-matrix/data/coco-development-v2"
    panel = ROOT/"codex/retrieval_matrix_panel"
    audit_args = (audit_path, development/"manifest.json", panel/"development-sources.json",
                  development/"images-summary.json", development/"dev/images-receipt.jsonl")
    data_audit.verify(*audit_args)
    sources = {}
    for relative, expected in base["source_files"].items():
        actual = Q.sha(ROOT/relative)
        if actual != expected and not relative.endswith("/PREREG.md"):
            raise ValueError("Qualified native P0 implementation changed: "+relative)
        sources[relative] = actual
    for folder in (HERE, panel, ROOT/"codex/retrieval_matrix_extra", ROOT/"codex/retrieval_matrix_native_p0"):
        for path in sorted(folder.iterdir()):
            if path.suffix in {".py", ".md", ".sh", ".json"}:
                sources[str(path.relative_to(ROOT))] = Q.sha(path)
    for name in ("runner.py", "models.py", "metrics.py"):
        relative = "codex/retrieval_matrix/"+name
        sources[relative] = Q.sha(ROOT/relative)
    result = {**copy.deepcopy(base), **candidate, "schema": 1, "state": "FROZEN",
        "purpose": "P2_QUALIFIED_FIXED_CLIP_BASELINES", "scope": "native-qualified CLIP on isolated development and COLA test",
        "datasets": datasets(development), "source_files": sources,
        "prereg": "codex/retrieval_matrix_clip_panel/PREREG.md", "parent_p0_plan": Q.entry(base_path),
        "qualification_adapter": Q.entry(ROOT/"codex/retrieval_matrix_native_p0/qualification.py"),
        "fixed_baseline_test": False, "development_audit": Q.entry(audit_path),
        "development_manifest": Q.entry(development/"manifest.json"),
        "development_sources": Q.entry(panel/"development-sources.json"),
        "development_images_summary": Q.entry(development/"images-summary.json"),
        "development_images_receipt": Q.entry(development/"dev/images-receipt.jsonl"),
        "not_covered": ["MCMR fused metadata", "custom methods", "P3 rerank"],
        "disclosure": "Same qualified P0 encoder and processor; panel ranking uses FP32 CPU dot with stable gallery-order ties. COLA uses fixed disclosed caption adaptation. No claim of native COLA paper reproduction."}
    data_audit.bind_datasets(result, Q.read(development/"manifest.json"))
    return result


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--base-plan", type=Path, required=True)
    p.add_argument("--qualification", action="append", required=True, help="MODEL=actual qualified receipt")
    p.add_argument("--development-audit", type=Path, required=True)
    p.add_argument("--out", type=Path, required=True)
    args = p.parse_args()
    if args.out.exists():
        raise ValueError("Never overwrite a CLIP P2 plan")
    refs = dict(row.split("=", 1) for row in args.qualification)
    if len(refs) != len(args.qualification):
        raise ValueError("Each qualified model must be declared once")
    result = build(args.base_plan, refs, args.development_audit)
    args.out.write_text(json.dumps(result, sort_keys=True, indent=2, allow_nan=False)+"\n")
    print(json.dumps({"state": "PREPARED_REQUIRES_COMMIT", "path": str(args.out), "sha256": Q.sha(args.out)}))


if __name__ == "__main__":
    main()
