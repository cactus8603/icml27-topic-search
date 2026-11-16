"""[codex] Build a separate GME / VLM2Vec-V2 frozen execution plan."""
from __future__ import annotations

import argparse
from copy import deepcopy
import hashlib
from importlib.metadata import version
import json
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[2]
HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT))
from codex.retrieval_matrix import build_plan as B


def sha(path):
    with Path(path).open("rb") as stream:
        return hashlib.file_digest(stream, "sha256").hexdigest()


def payload(source, receipt):
    snapshot = ROOT/source["snapshot"]
    files = {}
    for path in sorted(snapshot.iterdir()):
        if not path.is_file() or path.name.endswith((".lock", ".pending", ".download-partial")):
            continue
        digest = sha(path)
        row = receipt.get(str(path))
        if (not row or row.get("sha256") != digest or row.get("revision") != source["revision"]
                or row.get("repo") != source["repo"]):
            raise ValueError(f"Pinned model file lacks its official checksum-verified acquisition receipt: {path}")
        files[path.name] = digest
    return {"repo": source["repo"], "revision": source["revision"], "snapshot": str(snapshot), "files": files}


def vlm_source(protocol):
    root = ROOT/protocol["cache"]
    tree = {row["path"]: row for row in json.loads((root/"tree.json").read_text())["tree"]}
    files = {}
    for path in sorted(root.rglob("*")):
        if not path.is_file() or path.suffix not in {".py", ".md", ".txt"}:
            continue
        relative = path.relative_to(root).as_posix()
        body = path.read_bytes()
        git_blob = hashlib.sha1(f"blob {len(body)}\0".encode()+body).hexdigest()
        if tree[relative]["sha"] != git_blob:
            raise ValueError("Official VLM2Vec source differs from the pinned Git tree")
        files[relative] = hashlib.sha256(body).hexdigest()
    return {"repo": "TIGER-AI-Lab/VLM2Vec", "revision": protocol["official_vlm2vec_code"],
            "root": str(root), "files": files}


def dependencies():
    roots = [ROOT/"dl/retrieval-matrix/deps/qwen2-transformers451", ROOT/"dl/retrieval-matrix/deps/qwen2-flash",
             ROOT/"dl/retrieval-matrix/deps/core-p0"]
    files = {str(path.relative_to(ROOT)): sha(path) for root in roots for path in sorted(root.rglob("*"))
             if path.is_file() and not path.name.endswith(".pyc")}
    if not files or any(not root.is_dir() for root in roots):
        raise ValueError("The isolated Qwen2 dependency targets are incomplete")
    names = ("torch", "torchvision", "transformers", "tokenizers", "huggingface-hub", "qwen-vl-utils", "av",
             "pytrec-eval-terrier", "numpy", "Pillow", "accelerate", "peft", "flash-attn")
    return {"pythonpath": list(map(str, roots)), "packages": {name: version(name) for name in names}, "files": files,
            "install_reports": [B.entry(ROOT/"dl/retrieval-matrix/deps"/name) for name in
                                ("qwen2-transformers451-install.json", "qwen2-flash-install.json", "core-p0-install.json")]}


def build(reference_path, output):
    reference = json.loads(Path(reference_path).read_text())
    protocols = json.loads((ROOT/"codex/retrieval_matrix/official_protocols.json").read_text())
    receipt_path = ROOT/"dl/retrieval-matrix/acquisition/public-download-receipt.json"
    receipt = json.loads(receipt_path.read_text())["files"]
    official = protocols["official_code"]
    table = {row["path"]: row["sha256"] for row in official["files"]}
    proto = protocols["protocols"]["core_qwen2"]
    specs = {}
    names = {"gme2": "gme_2b", "vlm2vec2": "vlm2vec_v2"}
    for key, model_key in names.items():
        source = protocols["models"][model_key]
        spec = payload(source, receipt)
        spec.update(kind="official_embedding", variant="gme-qwen2" if key == "gme2" else "vlm2vec-v2-dora",
            embedding_dim=source["embedding_dim"], published_targets=source["targets"],
            official={"verified": True, "root": str(ROOT/official["cache"]), "revision": official["revision"], "files": table,
                      "wrapper": "models/qwen2_vl/qwen2_vl_wrapper.py", "model_class": proto["model_class"],
                      "processor_class": proto["processor_class"], "embedding": "embedding.py",
                      "model_name": "gme" if key == "gme2" else "vlm2vec"},
            loader={"torch_dtype": "bfloat16", "attn_implementation": "flash_attention_2"},
            processor=deepcopy(proto["processor_init"]), batches={"image": 4, "text": 32, "cross": 4},
            environment={name: version(name) for name in ("torch", "transformers", "numpy", "qwen-vl-utils", "Pillow", "tokenizers", "accelerate", "peft", "flash-attn")})
        if key == "vlm2vec2":
            v2 = protocols["protocols"]["vlm2vec_dora_unresolved"]
            base = {"repo": v2["base"], "revision": v2["base_revision"],
                    "snapshot": "hf_home/hub/models--Qwen--Qwen2-VL-2B-Instruct/snapshots/"+v2["base_revision"]}
            spec["base"] = payload(base, receipt)
            spec["vlm2vec_source"] = vlm_source(v2)
            spec["processor"]["size"] = {"shortest_edge": proto["processor_init"]["min_pixels"],
                                          "longest_edge": proto["processor_init"]["max_pixels"]}
            spec["disclosed_adaptation"] = "CORE release omitted V2 DoRA. Apply official PEFT full-model attachment and inference merge while preserving CORE's released evaluation wrapper; do not use a naked base or V1 checkpoint."
        specs[key] = spec
    sources = {name: sha(ROOT/name) for name in reference["source_files"]}
    for name in ("adapters.py", "runner.py", "build_plan.py", "preflight.py", "run.sh", "PREREG.md", "test_adapters.py"):
        sources[str((HERE/name).relative_to(ROOT))] = sha(HERE/name)
    plan = {"schema": 1, "state": "FROZEN", "owner": "[codex]", "phase": "P0-Qwen2",
            "prereg": "codex/retrieval_matrix_qwen2/PREREG.md", "source_files": sources,
            "reference_plan": B.entry(reference_path), "models": specs,
            "datasets": deepcopy(reference["datasets"]), "settings": deepcopy(reference["settings"]),
            "dependencies": dependencies(), "fixed_baseline_test": False, "gate": protocols["gate"],
            "acquisition_receipt": B.entry(receipt_path),
            "gpu_queue": {"min_free_mib": 20480, "max_utilization": 20},
            "known_differences": ["torch2.6.0/cu124 instead of original hardware runtime", "image4/text32 batches fixed before metrics",
                "transformers4.51.3 follows CORE's compatible Qwen2/GME path; newer VLM2Vec repository requirements list4.52.3",
                "V2 saved image size has obsolete min_pixels/max_pixels keys. Pass explicit shortest_edge/longest_edge equal to CORE's unchanged min/max settings, as upstream VLM2Vec processor loading does.",
                "V2 PEFT adapter application repairs an omission in the CORE release; this is disclosed rather than claimed byte-identical released loader behavior"]}
    if output.exists():
        raise ValueError("Refusing to overwrite an existing frozen plan")
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(plan, ensure_ascii=False, indent=2, sort_keys=True)+"\n")
    print(json.dumps({"owner": "[codex]", "plan": str(output), "sha256": sha(output), "models": list(specs)}))


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--reference-plan", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()
    build(args.reference_plan, args.out)
