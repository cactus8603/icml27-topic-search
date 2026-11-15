"""Prepare disjoint class-0 blocks from the frozen Week 2 grammar."""

import argparse
from hashlib import sha256
from io import BytesIO
import json
import os
from pathlib import Path
import subprocess
from tempfile import NamedTemporaryFile
import time

import numpy as np


SHARED = Path("/work/ChihChungHsu/cactus8603/icml")
DATA = SHARED / "data/biasgen-hier"
ORIGINAL = DATA / "hier_q6_l4_s1.0_qeff4_seed0_n40000.npy"
OUTPUT = DATA / "two-clocks-count-v2.npy"
ORIGINAL_SHA = "38cf10f797a0dfead18d8036f97f39efc58aa178663de68bd6d2525fb8615cee"
GRAMMAR_SHA = "5b1d6b3f12a2cfad9db0645648b32eeba8f9e382a5d8920290616741283faeac"
EXTERNAL = SHARED / "ext/biased-generalization"
GENERATOR_PATH = "modules/gen_filtered_hierarchical_data_wforbidden.py"
GENERATOR_REV = "df08bc634274f78ba1b99e83b07228b4252608ce"
GENERATOR_SHA = "4b06159dad53fb18009bbebb669b24d50c108064355d114ac767e75bf005793c"
SEED = 2026111519
SHUFFLE_SEED = 2026111521
BLOCKS = 7
BLOCK_ROWS = 5000
SMALL_ROWS = 833
EVAL_ROWS = 700
OUTPUT_CAP = 2 * 1024**2
INDEX_CAP = 1_000_000


def sha(payload):
    return sha256(payload).hexdigest()


def original_inputs():
    with ORIGINAL.open("rb") as stream:
        payload = stream.read(OUTPUT_CAP + 1)
    if sha(payload) != ORIGINAL_SHA:
        raise ValueError("The frozen Week 2 data hash differs")
    source = np.load(BytesIO(payload), allow_pickle=True)
    if source.dtype != object or source.shape != (7,):
        raise ValueError("Unexpected frozen data record layout")
    q, depth, sigma, roots, leaves, grammar, restrictions = source
    if ((q, depth, sigma) != (6, 4, 1.0) or roots.shape != (40_000,) or roots.dtype != np.float64
            or leaves.shape != (2**depth, 40_000) or leaves.dtype != np.int8
            or not np.isin(roots, np.arange(q)).all() or leaves.min() < 0 or leaves.max() >= q
            or grammar.shape != (q, q, q) or grammar.dtype != np.float64
            or sha(grammar.tobytes()) != GRAMMAR_SHA or not np.isfinite(grammar).all()
            or (grammar < 0).any() or not np.allclose(grammar.sum(axis=(1, 2)), 1, rtol=0, atol=1e-12)
            or not isinstance(restrictions, list) or len(restrictions) != 12):
        raise ValueError("The frozen grammar, roots or leaf layout differs")
    return q, depth, sigma, leaves, grammar, restrictions


def generator():
    source = subprocess.check_output(["git", "-C", str(EXTERNAL), "show", f"{GENERATOR_REV}:{GENERATOR_PATH}"])
    if sha(source) != GENERATOR_SHA:
        raise ValueError("The public generator source hash differs")
    local = {"__name__": "frozen_hier_generator", "__file__": str(EXTERNAL / GENERATOR_PATH)}
    exec(compile(source, GENERATOR_PATH, "exec"), local)
    return local["get_P_xlevel_root"], local["get_leaves"]


def build():
    q, depth, sigma, original_leaves, grammar, restrictions = original_inputs()
    probabilities, get_leaves = generator()
    probs = probabilities(grammar, 0)
    original = {row.tobytes() for row in original_leaves.T}
    seen = set()
    rows = BLOCKS * BLOCK_ROWS
    leaves = np.empty((2**depth, rows), dtype=np.int8)
    duplicates_original = duplicates_new = 0
    for index in range(INDEX_CAP):
        rng = np.random.default_rng(np.random.SeedSequence([SEED, index]))
        leaf = get_leaves(0, grammar, depth, 0, probs, rng=rng)
        key = leaf.tobytes()
        if key in original:
            duplicates_original += 1
        elif key in seen:
            duplicates_new += 1
        else:
            if leaf.dtype != np.int8 or leaf.shape != (2**depth,) or leaf.min() < 0 or leaf.max() >= q:
                raise ValueError("The author sampler returned an invalid leaf sequence")
            leaves[:, len(seen)] = leaf
            seen.add(key)
            if len(seen) == rows:
                break
    else:
        raise ValueError("Could not build all seven disjoint blocks within the index limit")

    # Randomize the accepted corpus before taking nested prefixes and assigning blocks.
    accepted_sha = sha(leaves.tobytes())
    permutation = np.random.default_rng(SHUFFLE_SEED).permutation(rows)
    leaves = np.ascontiguousarray(leaves[:, permutation])
    # Validate the actual saved columns, including cross-block uniqueness and the original corpus exclusion.
    saved = {column.tobytes() for column in leaves.T}
    if saved != seen or saved & original or sha(grammar.tobytes()) != GRAMMAR_SHA:
        raise ValueError("Generated data violate the disjointness or literal grammar requirement")
    roots = np.zeros(rows, dtype=np.float64)
    result = np.array([q, depth, sigma, roots, leaves, grammar, restrictions], dtype=object)
    buffer = BytesIO()
    np.save(buffer, result, allow_pickle=True)
    payload = buffer.getvalue()
    if len(payload) >= OUTPUT_CAP:
        raise ValueError("Prepared data exceed the storage budget")
    return payload, {
        "source_file": str(ORIGINAL), "source_sha256": ORIGINAL_SHA, "grammar_sha256": GRAMMAR_SHA,
        "generator_source": GENERATOR_PATH, "generator_commit": GENERATOR_REV, "generator_sha256": GENERATOR_SHA,
        "seed": SEED, "shuffle_seed": SHUFFLE_SEED, "index_end_exclusive": index + 1,
        "accepted_order_sha256": accepted_sha, "permutation_sha256": sha(permutation.tobytes()),
        "rejected_original": duplicates_original, "rejected_duplicate": duplicates_new,
        "original_distinct_leaves": len(original), "distinct_leaves": len(saved),
        "block_rows": BLOCK_ROWS, "train_blocks": list(range(BLOCKS - 1)), "test_block": BLOCKS - 1,
        "train_small_rows": SMALL_ROWS, "common_train_eval_rows": EVAL_ROWS, "numpy": np.__version__,
    }


def save_new(payload):
    with NamedTemporaryFile(dir=DATA, prefix=".two-clocks-count-v2.", suffix=".pending") as temp:
        temp.write(payload)
        temp.flush()
        os.link(temp.name, OUTPUT)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--check-only", action="store_true", help="Regenerate in memory and verify the published bytes")
    args = parser.parse_args()
    start = time.monotonic()
    payload, receipt = build()
    if args.check_only or OUTPUT.exists():
        if OUTPUT.read_bytes() != payload:
            raise ValueError("Shared output differs from the deterministic regeneration")
        status = "identical"
    else:
        save_new(payload)
        status = "created"
    print(json.dumps(receipt | {"status": status, "path": str(OUTPUT), "sha256": sha(payload),
                                "bytes": len(payload), "elapsed_seconds": round(time.monotonic() - start, 3)},
                     sort_keys=True))


if __name__ == "__main__":
    main()
