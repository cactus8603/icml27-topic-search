"""Verify the complete registered CLIP-only Gate (b) corpus and prerequisites."""

import argparse
import sys
from pathlib import Path

from common import ROOT, load_json
from evidence import atomic_json, atomic_text, bench_evidence, config_sha, file_sha, gate_a_ok, load_scores, read_json


def prerequisites():
    ga = ROOT / "results/gate_a/gate_a.json"
    if not gate_a_ok(ga):
        raise ValueError("registered Gate (a) evidence is missing or did not pass")
    bench = bench_evidence(ROOT / "results_bench", ROOT / "results_bench2")
    if not bench["pass"]:
        raise ValueError("registered bench reproducibility or throughput gate did not pass")
    return {"gate_a_sha256": file_sha(ga), "bench": bench}


def evaluate(root, job=None):
    G = load_json("criteria.json")["gate_b"]
    cells = {(G["model"], p) for p in G["prompts"]}
    df, source = load_scores(root, selected=(G["detector"],), expected_cells=cells, expected_job=job)
    rows = []
    for prompt in G["prompts"]:
        g = df[df.prompt == prompt]
        n = G["n_per_prompt"]
        if len(g) != n or g.seed.min() != 0 or g.seed.max() != n - 1:
            raise ValueError(f"{prompt}: expected all {n:,} seeds in [0,{n})")
        clip = g[G["detector"]]
        distinct = int(clip.nunique())
        hits = {str(t): int((clip >= t).sum()) for t in G["thresholds"]}
        good = [t for t in G["thresholds"] if hits[str(t)] >= G["min_hits"] and hits[str(t)] / n <= G["max_rate"]]
        rows.append(dict(prompt=prompt, n=n, distinct=distinct, hits=hits,
                         qualified=good if distinct >= G["min_distinct_scores"] else []))
    return {"status": "PASS" if any(row["qualified"] for row in rows) else "FAIL",
            "rows": rows, "source_sha256": source, "slurm_job_id": job, "config_sha256": config_sha()}


def verified_status(path):
    path = Path(path)
    try:
        stored = read_json(path)
        if stored.get("status") not in ("PASS", "FAIL"):
            return None
        observed = evaluate(path.parent, stored.get("slurm_job_id"))
        observed["prerequisites"] = prerequisites()
        if observed == stored:
            return observed["status"] == "PASS"
    except (OSError, KeyError, TypeError, ValueError):
        pass
    return None


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--results", type=Path, default=ROOT / "results_gateb")
    ap.add_argument("--job", help="Expected generator job owner recorded in its score sidecars")
    ap.add_argument("--prerequisites-only", action="store_true")
    ap.add_argument("--require-pass", action="store_true", help="Check an existing verified result against current evidence")
    args = ap.parse_args()
    output = args.results / "gate_b_verified.json"
    if args.require_pass:
        if verified_status(output) is True:
            print("Gate (b), Gate (a), and bench verified.")
            return 0
        print("Gate (b) has no current verified passing artifact.", file=sys.stderr)
        return 2
    if not args.prerequisites_only:
        atomic_json(output, {"status": "UNTESTABLE", "reason": "validation started"})
        atomic_text(args.results / "gate_b_verified.md", "# Gate (b)\n\nValidation has started.\n")
    try:
        pre = prerequisites()
        if args.prerequisites_only:
            print("Gate (a) and bench verified.")
            return 0
        result = evaluate(args.results, args.job)
        result["prerequisites"] = pre
    except (OSError, KeyError, TypeError, ValueError) as exc:
        print(f"Gate (b): UNTESTABLE ({exc})", file=sys.stderr)
        if not args.prerequisites_only:
            atomic_json(output, {"status": "UNTESTABLE", "reason": str(exc)})
            atomic_text(args.results / "gate_b_verified.md", f"# Gate (b): UNTESTABLE\n\n{exc}\n")
        return 2
    lines = ["# 閘門 (b) 完整資料驗證", "", "| 提示 | 張數 | 相異 CLIP 分數 | 命中 | 合格門檻 |", "|---|---:|---:|---|---|"]
    for row in result["rows"]:
        lines.append(f"| {row['prompt']} | {row['n']} | {row['distinct']} | {row['hits']} | {row['qualified']} |")
    lines.append(f"\n**Gate (b): {result['status']}**")
    atomic_text(args.results / "gate_b_verified.md", "\n".join(lines) + "\n")
    atomic_json(output, result)
    print("\n".join(lines))
    return 0 if result["status"] == "PASS" else 1


if __name__ == "__main__":
    sys.exit(main())
