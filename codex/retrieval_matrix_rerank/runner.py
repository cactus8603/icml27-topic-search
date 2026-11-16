"""[codex] Frozen CORE MCQ producer; CPU analysis is a separate explicit stage."""
from __future__ import annotations
import argparse
from contextlib import contextmanager
import fcntl
import hashlib
import importlib.util
import json
import os
from pathlib import Path
import subprocess
from collections import Counter
import numpy as np
import backends
import metrics

ROOT = Path(__file__).resolve().parents[2]
S = importlib.util.spec_from_file_location("rerank_checkpoint", ROOT / "codex/retrieval_scaling/runner.py")
C = importlib.util.module_from_spec(S)
S.loader.exec_module(C)
COUNTS = {"cola": 210, "replace_att": 788, "replace_obj": 1652, "replace_rel": 1406,
          "swap_att": 666, "swap_obj": 245, "coco_mcq": 5914, "voc_mcq": 5031}
REQUIRED_SOURCES = {"codex/retrieval_matrix_rerank/" + name for name in
                    ("runner.py", "backends.py", "metrics.py", "PREREG.md", "protocols.json", "run_core.sh")}
REQUIRED_SOURCES.add("codex/retrieval_scaling/runner.py")


def sha(path):
    return backends.sha(path)


def artifact(entry):
    path = Path(entry["path"]).resolve()
    if not path.is_relative_to(ROOT) or "retrieval-independent-panel" in str(path) or sha(path) != entry["sha256"]:
        raise ValueError("Frozen artifact path/checksum differs")
    return path


def source_gate(plan_path, plan, commit):
    if len(commit or "") != 40 or any(x not in "0123456789abcdef" for x in commit):
        raise ValueError("An immutable full --freeze-commit is required")
    if plan.get("state") != "FROZEN" or plan.get("reviewed") is not True:
        raise ValueError("Independent review and frozen plan required")
    declared = plan["source_files"]
    if not REQUIRED_SOURCES <= declared.keys():
        raise ValueError("Plan must bind all executed source, protocol and preregistration files")
    paths = set(declared) | {str(plan_path.resolve().relative_to(ROOT))}
    for relative in paths:
        path = (ROOT / relative).resolve()
        if not path.is_relative_to(ROOT) or subprocess.check_output(["git", "-C", str(ROOT), "show", f"{commit}:{relative}"]) != path.read_bytes():
            raise ValueError("Execution differs from committed source/plan")
        if relative in declared and sha(path) != declared[relative]:
            raise ValueError("Declared execution SHA differs")
    if subprocess.check_output(["git", "-C", str(ROOT), "status", "--porcelain", "--", "codex/retrieval_matrix_rerank"], text=True).strip():
        raise ValueError("CORE MCQ production requires a clean module worktree")


def validate_plan(plan):
    if (plan.get("schema") != 1 or plan.get("pair_batch") != 4
            or plan.get("smoke_absolute_tolerance") != .01 or plan.get("counts") != COUNTS
            or plan.get("synthetic_image_size") != [1280, 960]):
        raise ValueError("Fixed CORE MCQ schema, batching, smoke or complete data counts differ")
    revisions = {"core2": "64e1a36cd6ecace459a4c7cd8f7d0ef5d6fee30b",
                 "core8": "d95d36a34300e34b87cc1793827ac798642c4488"}
    if set(plan["models"]) != set(revisions):
        raise ValueError("CORE first batch requires both pinned released checkpoints")
    for key, revision in revisions.items():
        spec = plan["models"][key]
        if (spec.get("revision") != revision or spec.get("backend") != "swift_released"
                or spec.get("max_batch_size") != 32 or spec.get("verified_protocol") is not True):
            raise ValueError("Released CORE model/protocol requires independent review")


def output_path(path, plan):
    path = C.safe(path)
    root = C.safe(plan["output_root"])
    if (root != ROOT / "dl/retrieval-matrix-rerank/runs" or path == root
            or not path.is_relative_to(root)):
        raise ValueError("Run outputs must use a distinct directory under the frozen runtime runs root")
    return path


def load_examples(plan, *, verify_images=False):
    rows = [json.loads(line) for line in artifact(plan["examples"]).read_text().splitlines()]
    pairs = [pair for row in rows for pair in row["pairs"]]
    if (len(rows) != 15912 or len(pairs) != 58891 or len({p["id"] for p in pairs}) != len(pairs)
            or len({row["id"] for row in rows}) != len(rows)
            or Counter(row["subset"] for row in rows) != COUNTS):
        raise ValueError("Complete original MCQ row/pair count differs")
    for row in rows:
        expected = ("cola_text_choice", 4) if row["subset"] == "cola" else (
            ("negbench_mcq", 4) if row["subset"].endswith("_mcq") else ("scpp", 3))
        instruction = "Determine if the following image is relevant to the query." + ("" if row["subset"] == "cola" else " ")
        if (row["task"], len(row["pairs"])) != expected or row["instruction"] != instruction:
            raise ValueError("Released MCQ rule, comparison axes or instruction differs")
    if [json.loads(line) for line in artifact(plan["pairs"]).read_text().splitlines()] != pairs:
        raise ValueError("Prepared pair order differs from original examples")
    for source in plan["source_data"]:
        artifact(source)
    receipt = [json.loads(line) for line in artifact(plan["images_receipt"]).read_text().splitlines()]
    images = {row["path"]: row for row in receipt}
    if len(images) != len(receipt) or len(images) != 9425 or {p["image"] for p in pairs} != set(images):
        raise ValueError("Original complete image receipt differs")
    if verify_images:
        for path, row in images.items():
            path = C.safe(path)
            if (not path.is_relative_to(ROOT / "dl") or sha(path) != row["sha256"]
                    or path.stat().st_size != row["bytes"]):
                raise ValueError("Original image payload differs")
    return rows


def environment(plan):
    import torch
    import transformers
    import flash_attn
    from importlib.metadata import version
    for entry in (plan["environment"]["install_report"], plan["environment"]["frozen_pip"]):
        artifact(entry)
    for name, expected in plan["environment"]["packages"].items():
        if version(name) != expected:
            raise ValueError(f"Frozen package differs: {name}")
    if (torch.__version__ != plan["environment"]["packages"]["torch"]
            or transformers.__version__ != plan["environment"]["packages"]["transformers"]
            or not torch.cuda.is_available()):
        raise ValueError("Actual imported modules/CUDA differ")
    return {"torch": torch.__version__, "transformers": transformers.__version__, "flash_attn": flash_attn.__version__,
            "cuda": torch.version.cuda, "gpu": torch.cuda.get_device_name(0)}


def require_smoke(proof, identity, plan):
    difference = proof.get("batch1_vs_fixed_max_abs", np.nan)
    if (proof.get("state") != "PASS" or proof.get("technical_only") is not True
            or proof.get("identity") != identity or not isinstance(difference, (int, float))
            or not np.isfinite(difference) or not 0 <= difference <= plan["smoke_absolute_tolerance"]):
        raise ValueError("Matching committed synthetic smoke required")


def smoke(out, spec, identity, plan):
    if (out / "smoke.json").exists():
        require_smoke(C.read_json(out / "smoke.json"), identity, plan)
        return
    from PIL import Image
    import torch
    engine = backends.SwiftReleased(spec)
    image = out / "synthetic.png"
    Image.new("RGB", tuple(plan["synthetic_image_size"]), (64, 128, 32)).save(image)
    rows = [{"id": f"synthetic:{i}", "query": "A synthetic color image.", "image": str(image)}
            for i in range(plan["pair_batch"])]
    instruction = "Determine if the following image is relevant to the query. "
    multiple = engine.score_pairs(rows, instruction=instruction)
    single = engine.score_pairs(rows[:1], instruction=instruction)
    difference = abs(multiple[0]["score"] - single[0]["score"])
    passed = difference <= plan["smoke_absolute_tolerance"]
    C.fixed_json(out / "smoke.json", {"owner": "[codex]", "state": "PASS" if passed else "FAILED",
        "technical_only": True, "identity": identity, "batch1_vs_fixed_max_abs": difference,
        "signature": engine.signature, "raw": multiple, "peak_allocated_bytes": torch.cuda.max_memory_allocated()})
    if not passed:
        raise ValueError("Frozen synthetic score batch-invariance tolerance failed")


def produce(out, spec, identity, plan, rows, smoke_out):
    import torch
    proof = json.loads((smoke_out / "smoke.json").read_text())
    require_smoke(proof, identity, plan)
    engine = backends.SwiftReleased(spec)
    C.fixed_json(out / "backend.json", engine.signature)
    by_task = {}
    for row in rows:
        by_task.setdefault(row["subset"], []).append(row)
    complete = {}
    for subset, examples in by_task.items():
        pairs = [pair for row in examples for pair in row["pairs"]]
        instruction = examples[0]["instruction"]
        if any(row["instruction"] != instruction for row in examples):
            raise ValueError("Official subset mixes incompatible instructions")
        def compute(batch):
            original = engine.score_pairs(batch, instruction=instruction)
            # ms-swift native content is a one-element float32 scalar list.
            if any(row["official_raw"] != [row["score"]] for row in original):
                raise ValueError("Released score shape changed")
            return np.array([[row["score"]] for row in original], dtype=np.float32)
        path = out / (subset + ".npy")
        C.checkpoint(path, pairs, plan["pair_batch"], compute,
                     {**identity, "subset": subset, "instruction": instruction}, progress=out / "progress.json")
        state = json.loads(path.with_suffix(".state.json").read_text())
        if state["sha256"] != sha(path) or state["completed"] != len(pairs):
            raise ValueError("Incomplete official subset checkpoint")
        complete[subset] = {"pairs": len(pairs), "array_sha256": sha(path), "state_sha256": sha(path.with_suffix(".state.json"))}
    C.fixed_json(out / "complete.json", {"owner": "[codex]", "state": "COMPLETE", "identity": identity,
        "subsets": complete, "raw_semantics": "Original content[0] activated float32 scalar, no rescaling/clipping; list wrapper reconstructed exactly.",
        "peak_allocated_bytes": torch.cuda.max_memory_allocated(), "peak_reserved_bytes": torch.cuda.max_memory_reserved()})


def analyze(out, identity, rows, plan):
    receipt = json.loads((out / "complete.json").read_text())
    if receipt["identity"] != identity or receipt["state"] != "COMPLETE":
        raise ValueError("Incomplete or mismatched producing provenance")
    by_subset = {}
    for row in rows:
        by_subset.setdefault(row["subset"], []).extend(row["pairs"])
    scores = []
    for subset, pairs in by_subset.items():
        path = out / (subset + ".npy")
        expected = receipt["subsets"][subset]
        if sha(path) != expected["array_sha256"] or sha(path.with_suffix(".state.json")) != expected["state_sha256"]:
            raise ValueError("Official score array/state differs")
        values = np.load(path, allow_pickle=False)
        if values.shape != (len(pairs), 1):
            raise ValueError("Official score coverage differs")
        scores.extend({"id": pair["id"], "score": float(value[0])} for pair, value in zip(pairs, values, strict=True))
    report, detail = metrics.evaluate(rows, scores)
    target = plan["acceptance"][identity["model"]]
    report["reproduction"] = {task: {"published": expected, "observed": report["benchmarks"][task]["macro_accuracy"],
        "absolute_tolerance": plan["acceptance"]["absolute_tolerance"],
        "passed": abs(report["benchmarks"][task]["macro_accuracy"] - expected) <= plan["acceptance"]["absolute_tolerance"]}
        for task, expected in target.items()}
    C.atomic_json(out / "results/report.json", report)
    C.atomic_json(out / "results/per-example.json", detail)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--stage", choices=["smoke", "score", "analyze"], required=True)
    parser.add_argument("--plan", type=Path, required=True)
    parser.add_argument("--model", choices=["core2", "core8"], required=True)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--smoke-out", type=Path)
    parser.add_argument("--freeze-commit", required=True)
    args = parser.parse_args()
    plan = json.loads(args.plan.read_text())
    source_gate(args.plan, plan, args.freeze_commit)
    validate_plan(plan)
    for entry in plan["external_sources"] + [plan["preparation"]]:
        artifact(entry)
    spec = plan["models"][args.model]
    identity = {"plan_sha256": sha(args.plan), "freeze_commit": args.freeze_commit, "model": args.model,
                "model_spec": spec, "source_files": plan["source_files"]}
    args.out = output_path(args.out, plan)
    args.out.mkdir(parents=True, exist_ok=True)
    with (args.out / "writer.lock").open("a") as lock:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        if args.stage == "analyze":
            identity["environment"] = json.loads((args.out / "identity.json").read_text())["environment"]
            analyze(args.out, identity, load_examples(plan), plan)
            return
        identity["environment"] = environment(plan)
        C.fixed_json(args.out / "identity.json", identity)
        if args.stage == "smoke":
            smoke(args.out, spec, identity, plan)
        else:
            if args.smoke_out is None:
                raise ValueError("Official score production requires --smoke-out")
            produce(args.out, spec, identity, plan, load_examples(plan, verify_images=True), output_path(args.smoke_out, plan))


if __name__ == "__main__":
    main()
