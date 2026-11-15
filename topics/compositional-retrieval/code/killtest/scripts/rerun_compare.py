"""閘門 G3：重跑一致性。比較 extract_ht.py 兩次（不同 tag）輸出的逐列 cosine 之最大差。
用法：python scripts/rerun_compare.py --a results/ht --tag-a "" --tag-b _rerun [--batch-note ...]
輸出 results/gates/g3_rerun.json（max_abs_diff 為所有模型／子集、所有 cosine 的最大值）。
第二次建議用不同 batch size（見 slurm/e0_ht.sbatch 的 rerun 段），同時檢驗 batch 不變性。
"""
import argparse
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
from common import EXP, SUBSETS, jdump


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dir", default=str(EXP / "results" / "ht"))
    ap.add_argument("--tag-a", default="")
    ap.add_argument("--tag-b", default="_rerun")
    ap.add_argument("--out", default=str(EXP / "results" / "gates" / "g3_rerun.json"))
    args = ap.parse_args()
    d = Path(args.dir)
    worst, per = 0.0, {}
    for m in ("clip_b32", "cs_clip"):
        for s in SUBSETS:
            a, b = d / f"{m}_{s}{args.tag_a}.npz", d / f"{m}_{s}{args.tag_b}.npz"
            if not (a.exists() and b.exists()):
                continue
            za, zb = np.load(a), np.load(b)
            diff = 0.0
            for t in ("anchor", "full", "half", "dplus", "dminus"):
                diff = max(diff, float(np.abs(np.einsum("nd,nd->n", za["img"], za[t]) -
                                              np.einsum("nd,nd->n", zb["img"], zb[t])).max()))
            per[f"{m}/{s}"] = diff
            worst = max(worst, diff)
    jdump({"max_abs_diff": worst, "per": per}, args.out)
    print({"max_abs_diff": worst, "per": per})


if __name__ == "__main__":
    main()
