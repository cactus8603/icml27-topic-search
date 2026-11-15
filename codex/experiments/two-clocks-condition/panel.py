"""Build overlapping-arm and nonoverlapping-pick tree panels and exposure schedules."""

from hashlib import file_digest, sha256
from importlib.util import module_from_spec, spec_from_file_location
from pathlib import Path

import numpy as np


HERE = Path(__file__).resolve().parent
COUNT_PREP_SHA = "5b135f302ee6d1650d2c86e1a87c5c131ecee674a2fa65530da4ef81fee7c46b"
COUNT_DATA_SHA = "7bd02fff2196a894218738028d0202614f87c886ae420ea8b92f5b8389c98499"
Q, WIDTH, BLOCKS = 6, 16, 7
STRATUM_ROWS, SLOTS = 128, 2
STRATA = Q * Q
TRAIN_N = STRATA * STRATUM_ROWS
BATCH = STRATA * SLOTS
UPDATES_PER_EPOCH, HORIZON = 10, 20000
GRID = (1, 2, 4, 8, 12, 18, 24, 30, 36, 44, 54, 66, 82, 101, 125, 155, 192)
EVAL_N = 720
DATA_SEED = 2026111730
ASSIGN_SEED = 2026111731
TRAIN_SEED = 58000
EVAL_SEED = 2026111732


def hash_file(path):
    with path.open("rb") as stream:
        return file_digest(stream, "sha256").hexdigest()


def reference():
    path = HERE.parent / "two-clocks-count/prepare.py"
    if hash_file(path) != COUNT_PREP_SHA:
        raise ValueError("Pinned grammar sampling helper differs")
    spec = spec_from_file_location("frozen_count_prepare", path)
    module = module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def first_leaf_tv(grammar):
    # Exact marginal of the leftmost leaf given the real root.
    distribution = np.linalg.matrix_power(grammar.sum(axis=2), 4)
    pairs = [float(np.abs(distribution[i] - distribution[j]).sum() / 2)
             for i in range(Q) for j in range(i + 1, Q)]
    if not np.allclose(distribution.sum(axis=1), 1) or min(pairs) <= 1e-6:
        raise ValueError("The six true root conditions do not have distinct leaf laws")
    return min(pairs), max(pairs)


def build_data(path):
    parent = reference()
    q, depth, sigma, original, grammar, restrictions = parent.original_inputs()
    count_path = parent.DATA / "two-clocks-count-v2.npy"
    if hash_file(count_path) != COUNT_DATA_SHA:
        raise ValueError("The previous sample-count panel differs")
    previous = np.load(count_path, allow_pickle=True)[4]
    excluded = {row.tobytes() for data in (original, previous) for row in data.T}
    seen = set(excluded)
    root_probabilities, get_leaves = parent.generator()
    root_prob = root_probabilities(grammar, 0)
    by_root = np.empty((Q, BLOCKS * Q * STRATUM_ROWS, WIDTH), dtype=np.int8)
    for root, output in enumerate(by_root):
        count = 0
        for index in range(200_000):
            rng = np.random.default_rng(np.random.SeedSequence([DATA_SEED, root, index]))
            leaf = get_leaves(root, grammar, depth, 0, root_prob, rng=rng)
            key = leaf.tobytes()
            if key in seen:
                continue
            if leaf.shape != (WIDTH,) or leaf.dtype != np.int8 or not ((0 <= leaf) & (leaf < Q)).all():
                raise ValueError("Pinned tree sampler returned an invalid leaf")
            output[count] = leaf
            seen.add(key)
            count += 1
            if count == len(output):
                break
        else:
            raise ValueError("The fresh panel is not supportable at the fixed draw cap")
        np.random.default_rng(np.random.SeedSequence([ASSIGN_SEED, root])).shuffle(output)
    leaves = by_root.reshape(Q, BLOCKS, Q, STRATUM_ROWS, WIDTH).transpose(1, 3, 0, 2, 4)
    leaves = np.ascontiguousarray(leaves.reshape(-1, WIDTH).T)
    roots = np.tile(np.repeat(np.arange(Q), Q), BLOCKS * STRATUM_ROWS).astype(float)
    if len(seen) - len(excluded) != BLOCKS * TRAIN_N or len({row.tobytes() for row in leaves.T}) != len(roots):
        raise ValueError("The pooled tree cohort did not preserve exact uniqueness")
    np.save(path, np.array([q, depth, sigma, roots, leaves, grammar, restrictions], dtype=object), allow_pickle=True)
    lo, hi = first_leaf_tv(grammar)
    return {"data_sha256": hash_file(path), "literal_grammar_sha256": sha256(grammar.tobytes()).hexdigest(),
            "excluded_prior_distinct": len(excluded), "rows": len(roots), "min_true_root_first_leaf_tv": lo,
            "max_true_root_first_leaf_tv": hi}


def schedule(epochs=GRID[-1]):
    groups = np.arange(STRATA)
    slots = STRATUM_ROWS // SLOTS
    steps = np.arange(epochs * UPDATES_PER_EPOCH)
    cycle, phase = divmod(steps, slots)
    indices = ((phase[:, None, None] * SLOTS + np.arange(SLOTS)[None, :, None]) * STRATA
               + groups[None, None, :]).reshape(len(steps), BATCH)
    drop_by_group = (cycle[:, None] + phase[:, None] + groups // Q + groups % Q) % 10 == 0
    drop = np.tile(drop_by_group, (1, SLOTS))
    return indices.reshape(epochs, UPDATES_PER_EPOCH, BATCH), drop.reshape(epochs, UPDATES_PER_EPOCH, BATCH)


def verify_schedule(indices, drop, complete=True):
    index, dropped = indices.reshape(-1, BATCH), drop.reshape(-1, BATCH)
    truth, shuffled = index % STRATA // Q, index % Q
    for label in range(Q):
        if (not np.all((truth == label).sum(axis=1) == BATCH // Q)
                or not np.all((shuffled == label).sum(axis=1) == BATCH // Q)
                or not np.array_equal((dropped & (truth == label)).sum(axis=1),
                                      (dropped & (shuffled == label)).sum(axis=1))):
            raise ValueError("Effective per-label sample exposures differ between arms")
    counts = np.bincount(index.ravel(), minlength=TRAIN_N)
    null_counts = np.bincount(index[dropped], minlength=TRAIN_N)
    if complete and (not np.all(counts == 30) or not np.all(null_counts == 3)):
        raise ValueError("Each row must occur exactly 30 times and exactly three times on the null branch")
    return {"pooled_unique_rows_per_pick": TRAIN_N, "unique_rows_per_label": TRAIN_N // Q,
            "rows_per_root_by_shuffled_label_cell": STRATUM_ROWS, "pre_dropout_step_rows": BATCH,
            "updates": len(index), "examples": int(counts.sum()), "null_examples": int(null_counts.sum()),
            "schedule_sha256": sha256(indices.tobytes() + drop.tobytes()).hexdigest()}
