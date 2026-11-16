"""[codex] Deliberately closed central entry until independent protocol review."""
import argparse
from pathlib import Path
import sys
ROOT=Path(__file__).resolve().parents[2];sys.path.insert(0,str(ROOT))
from codex.retrieval_matrix_triplet_native_p0 import contracts as C,build_plans


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    for name in ('freeze-commit','plan','model','out'):parser.add_argument('--'+name,required=True)
    args=parser.parse_args();plan=build_plans.validate(C.read(args.plan))
    if args.model!=plan['model_key']:raise ValueError('Central model differs from this exact draft variant')
    # Before imports of Torch, a GPU probe, model construction, output mkdir or
    # any producer work. Deliberately no implementation fallback or retry.
    C.require_executable(plan)


if __name__=='__main__':main()
