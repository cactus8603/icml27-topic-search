"""正式結果驗證全分片後判分；保留 grade.py DIR 作既有試驗用入口。"""
import argparse
from contextlib import contextmanager
import fcntl
from pathlib import Path
import os
import sys
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from common import cfg, grade_raw, load_data, pairs_cfg
from artifacts import (atomic_json, check_gate_minus1, check_run, grader_receipt, input_receipts,
                       load_raw_bundle, load_verified, require, sha256, validate_records)


@contextmanager
def grading_lock(directory):
    with (directory / ".grade.lock").open("a", encoding="utf-8") as lock:
        fcntl.flock(lock, fcntl.LOCK_EX)
        yield


def verified_grade(directory, pair, cond, workers, n_shards):
    pc, ecfg, data = pairs_cfg(), cfg("criteria.json")["eval"], load_data("all")
    gate = check_gate_minus1(directory.parent.parent, pair, pc)
    if n_shards is not None:
        expected = {f"meta_{i}of{n_shards}.json" for i in range(n_shards)}
        seen = {p.name for p in directory.glob("meta_*.json")}
        require(seen <= expected, "發現不相容或舊分片憑證")
        if seen != expected:
            print("等待全部分片完成憑證")
            return
    bundle, n = check_run(directory, pair, cond, pc, ecfg, data, gate)
    inputs = input_receipts(bundle)
    if (directory / "graded_receipt.json").exists():
        load_verified(directory, pair, cond, pc, ecfg, data, gate)
        print("已存在有效的正式判分憑證")
        return
    raw = load_raw_bundle(bundle, n, pc["max_new_tokens"], cond)
    stage = directory / ".graded_verify_stage.json"
    try:
        grader = grader_receipt()
        graded = grade_raw([str(path) for path, _, _ in bundle], stage, workers)
        validate_records(graded, raw.keys(), n, pc["max_new_tokens"], cond in ("rl_hf", "rl_replace", "geo_mix"))
        for i, rec in raw.items():
            fields = ["finished", "ntok"] + (["n_repl"] if "n_repl" in rec else [])
            require(all(graded[i][key] == rec[key] for key in fields), f"判分與取樣資料不對齊：{i}")
        require(input_receipts(bundle) == inputs, "判分中取樣分片或憑證受到變更")
        target = directory / "graded_verified.json"
        stage.replace(target)
        atomic_json(directory / "graded_receipt.json", {"pair": pair, "condition": cond, "grader": grader,
                                                        "inputs": inputs, "grade_sha256": sha256(target)})
        print(f"正式判分完成：{target}")
    finally:
        stage.unlink(missing_ok=True)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("dir", type=Path)
    ap.add_argument("--workers", type=int, default=8)
    ap.add_argument("--pair")
    ap.add_argument("--condition")
    ap.add_argument("--n-shards", type=int)
    args = ap.parse_args()
    if bool(args.pair) != bool(args.condition):
        ap.error("--pair 與 --condition 必須一併指定")
    with grading_lock(args.dir):
        if args.pair:
            return verified_grade(args.dir, args.pair, args.condition, args.workers, args.n_shards)
        raws = sorted(args.dir.glob("raw_*.jsonl.gz"))
        require(raws, f"no raw files in {args.dir}")
        out = grade_raw(raws, args.dir / "graded.json", args.workers)
        n = sum(len(v["correct"]) for v in out.values())
        print(args.dir, "questions", len(out), "samples", n, "pass@1", sum(sum(v["correct"]) for v in out.values()) / n)


if __name__ == "__main__":
    main()
