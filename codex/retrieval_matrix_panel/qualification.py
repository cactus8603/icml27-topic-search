"""[codex] Require native P0 metrics and independent evidence before P2."""
from __future__ import annotations

import argparse
import hashlib
import json
import math
from pathlib import Path

DATASETS = {"coco": ("coco_t2i_recall_5", ("t2i", "recall_5")),
            "flickr": ("flickr_t2i_recall_5", ("t2i", "recall_5")),
            "negbench": ("negbench_coco_recall_5", ("R@5",))}


def sha(path):
    d = hashlib.sha256()
    with Path(path).open("rb") as f:
        for block in iter(lambda: f.read(1 << 20), b""):
            d.update(block)
    return d.hexdigest()


def read(path):
    return json.loads(Path(path).read_text())


def digest(value):
    return hashlib.sha256(json.dumps(value, ensure_ascii=False, sort_keys=True,
        separators=(",", ":"), allow_nan=False).encode()).hexdigest()


def entry(path):
    path = Path(path).resolve()
    return {"path": str(path), "sha256": sha(path)}


def checked(item):
    path = Path(item["path"])
    if sha(path) != item["sha256"]:
        raise ValueError("Qualification evidence changed: "+str(path))
    return path


def assess(plan, plan_sha, key, reports, review):
    if set(reports) != set(DATASETS):
        raise ValueError("All three fixed P0 dataset reports must be retained")
    model = plan["models"][key]
    if (review.get("state") != "PASS" or review.get("schema") != 1
            or review.get("scope") != "official-native-p0"
            or not review.get("reviewer") or not review.get("producer")
            or review["reviewer"] == review["producer"]
            or review.get("independent_recomputation") is not True
            or review.get("audited_arrays") is not True
            or review.get("plan_sha256") != plan_sha or review.get("model") != key):
        raise ValueError("A separate reviewer must independently recompute original native arrays")
    detail = {}
    for name, (target_key, fields) in DATASETS.items():
        row = reports[name]
        identity = row.get("provenance", {}).get("index", {}).get("identity", {})
        if (row.get("model") != key or row.get("dataset") != name
                or identity.get("model") != key or identity.get("dataset") != name
                or identity.get("plan_sha256") != plan_sha or identity.get("model_spec") != model
                or identity.get("settings") != plan["settings"]):
            raise ValueError("Report belongs to a different frozen model, protocol, or dataset")
        ds = plan["datasets"][name]
        if (row.get("queries") != ds["expected_eligible"]
                or row.get("raw_queries") != ds["expected_queries"]
                or row.get("gallery") != ds["expected_gallery"]
                or row.get("excluded") != []):
            raise ValueError("Report denominator differs from its complete frozen dataset")
        value = row.get("official")
        try:
            for field in fields:
                value = value[field]
            observed = float(value)
            target = float(model["published_targets"][target_key])
        except (TypeError, KeyError, ValueError) as error:
            raise ValueError("Official native metric missing; secondary scores cannot qualify") from error
        if not math.isfinite(observed) or not 0 <= observed <= 1 or not 0 <= target <= 1:
            raise ValueError("P0 scores must be finite fractions")
        delta = observed-target
        detail[name] = {"metric": target_key, "observed": observed, "target": target,
                        "delta_pp": 100*delta, "within_1pp": abs(delta) <= .01+1e-12}
    passed = sum(row["within_1pp"] for row in detail.values())
    return {"state": "PASS" if passed >= 2 else "FAILED", "required_distinct_datasets": 2,
            "matched_distinct_datasets": passed, "datasets": detail,
            "model": key, "model_spec_sha256": digest(model), "p0_plan_sha256": plan_sha}


def make(plan_path, key, report_paths, review_path):
    plan = read(plan_path)
    references = {name: entry(path) for name,path in report_paths.items()}
    review = read(review_path)
    if review.get("reports") != {name: item["sha256"] for name,item in references.items()}:
        raise ValueError("Independent review is not bound to all exact report bytes")
    result = assess(plan, sha(plan_path), key, {name: read(path) for name,path in report_paths.items()}, review)
    result.update(schema=1, owner="[codex]", p0_plan=entry(plan_path), reports=references,
                  independent_review=entry(review_path))
    return result


def admit(receipt_entry, model_spec):
    receipt = read(checked(receipt_entry))
    actual = make(checked(receipt["p0_plan"]), receipt["model"],
                  {name: checked(item) for name,item in receipt["reports"].items()},
                  checked(receipt["independent_review"]))
    if receipt != actual or actual["state"] != "PASS" or actual["model_spec_sha256"] != digest(model_spec):
        raise ValueError("P2 requires unchanged independently qualified P0 model evidence")
    return actual


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--plan", type=Path, required=True)
    parser.add_argument("--model", required=True)
    parser.add_argument("--reports-root", type=Path, required=True)
    parser.add_argument("--independent-review", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()
    if args.out.exists():
        raise ValueError("Never overwrite an existing P0 qualification verdict")
    result = make(args.plan, args.model, {name: args.reports_root/name/"results/report.json" for name in DATASETS},
                  args.independent_review)
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(result, sort_keys=True, indent=2, allow_nan=False)+"\n")
    print(json.dumps({key:result[key] for key in ("state", "model", "matched_distinct_datasets")}, sort_keys=True))


if __name__ == "__main__":
    main()
