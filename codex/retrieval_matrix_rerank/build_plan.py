"""[codex] Bind the complete CORE MCQ batch for independent review; no inference."""
from pathlib import Path
import argparse
import json
from build_specs import ROOT, RUNTIME, sha, source


def build(spec_path):
    here = Path(__file__).parent
    draft = json.loads(spec_path.read_text())
    preparation = RUNTIME / "data/mcq-prepared/preparation.json"
    prepared = json.loads(preparation.read_text())
    if prepared["state"] != "COMPLETE" or prepared["cola_rule"] != "text_choice":
        raise ValueError("Complete CORE text-choice MCQ preparation required")
    protocols = json.loads((here / "protocols.json").read_text())
    external = [source(ROOT / protocols["official_CORE"]["cache"] / row["path"])
                for row in protocols["official_CORE"]["files"]]
    for name in ("cpu-probe.json", "sources/ms_swift-4.0.0-py3-none-any.whl",
                 "acquisition/core-args-receipt.json", "data/voc/archive-receipt.json"):
        external.append(source(RUNTIME / name))
    files = [path for path in sorted(here.iterdir()) if path.suffix in {".py", ".md", ".sh"}]
    files.extend([here / "protocols.json", spec_path, ROOT / "codex/retrieval_scaling/runner.py"])
    models = draft["models"]
    if set(models) != {"core2", "core8"} or any(s["verified_protocol"] for s in models.values()):
        raise ValueError("First plan must start with independently unreviewed CORE-only specs")
    return {"owner": "[codex]", "schema": 1, "state": "REVIEW_REQUIRED", "reviewed": False,
            "phase": "P0 CORE released MCQ complete reproduction", "models": models,
            "source_files": {str(path.resolve().relative_to(ROOT)): sha(path) for path in files},
            "external_sources": external, "environment": draft["environment"],
            "preparation": source(preparation), "counts": prepared["counts"],
            "examples": source(preparation.parent / "examples.jsonl"),
            "pairs": source(preparation.parent / "pairs.jsonl"),
            "images_receipt": source(preparation.parent / "images-receipt.jsonl"),
            "source_data": [{"path": path, "sha256": digest} for path, digest in prepared["sources"].items()],
            "pair_batch": 4, "smoke_absolute_tolerance": .01, "synthetic_image_size": [1280, 960],
            "output_root": str(RUNTIME / "runs"),
            "acceptance": {"absolute_tolerance": .01,
                "core2": {"cola_text_choice": .724, "scpp": .839, "negbench_mcq": .646},
                "core8": {"cola_text_choice": .843, "scpp": .875, "negbench_mcq": .698}},
            "disclosure": "Checkpoint args record swift4.0.0.dev0; publicly available stable4.0.0 is pinned. Official request, model type, template and binary projection checked on CPU. Full-GPU fp16, no quantization/offload/OOM fallback.",
            "review_action": "Independent reviewer sets state=FROZEN, reviewed=true and both models' verified_protocol=true, then commits this plan and all declared source files. The full commit is required at execution."}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--spec", type=Path, default=Path(__file__).parent / "backend_specs.core.draft.json")
    parser.add_argument("--out", type=Path, default=Path(__file__).parent / "plan.core-mcq.v1.json")
    args = parser.parse_args()
    if args.out.exists():
        raise ValueError("Refusing to overwrite an existing execution plan")
    value = build(args.spec)
    args.out.write_text(json.dumps(value, indent=2, sort_keys=True) + "\n")
    print(json.dumps({"owner": "[codex]", "out": str(args.out), "sha256": sha(args.out), "state": value["state"]}))


if __name__ == "__main__":
    main()
