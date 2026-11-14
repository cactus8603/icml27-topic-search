"""把某 cond 的 raw_*.jsonl.gz 判分並合併成 graded.json。用法：grade.py DIR [--workers 8]"""
import argparse, glob, os, sys
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from common import grade_raw

ap = argparse.ArgumentParser()
ap.add_argument("dir")
ap.add_argument("--workers", type=int, default=8)
a = ap.parse_args()
raws = sorted(glob.glob(os.path.join(a.dir, "raw_*.jsonl.gz")))
assert raws, "no raw files in " + a.dir
out = grade_raw(raws, os.path.join(a.dir, "graded.json"), a.workers)
n = sum(len(v["correct"]) for v in out.values())
print(a.dir, "questions", len(out), "samples", n, "pass@1", sum(sum(v["correct"]) for v in out.values()) / n)
