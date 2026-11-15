"""Read-only verification of registered Stage 0 exact/retrain prerequisites."""

import argparse
import hashlib
import json
from pathlib import Path

import numpy as np
from scipy.stats import rankdata


def collect(path, shape):
    arr = np.load(path, allow_pickle=False)
    if arr.shape != shape or not np.isfinite(arr).all():
        raise ValueError(f"{path} has invalid shape or nonfinite contents")
    return arr


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--cfg", required=True, type=Path)
    p.add_argument("--criteria", required=True, type=Path)
    p.add_argument("--out", type=Path)
    a = p.parse_args()
    cfg = json.loads(a.cfg.read_text())
    criteria = json.loads(a.criteria.read_text())
    root = Path(cfg["run_dir"])
    k, n, total = cfg["k_stage0"], cfg["n_train"] or 50_000, cfg["k_total"]
    paths = []

    def get(path, shape):
        paths.append(path)
        return collect(path, shape)

    chunks = []
    lengths = []
    for begin in range(0, k, cfg["qchunk"]):
        end = min(k, begin + cfg["qchunk"])
        lengths.append([begin, end])
        chunks.append(get(root / "exact" / f"Y_{begin}_{end}.npy", (n, end - begin)))
        get(root / "exact" / f"ql_{begin}_{end}.npy", (end - begin,))
    Y = np.concatenate(chunks, axis=1)
    L = np.stack([get(root / "retrain" / f"loss_{j}.npy", (total + 1,)) for j in range(cfg["M"])])
    C = np.stack([get(root / "retrain" / f"noise_{j}_task0.npy", (total + 1,)) for j in range(1, cfg["noise_repeats"] + 1)])
    if not ((L[:, -1] >= 0) & (L[:, -1] <= 1)).all() or not ((C[:, -1] >= 0) & (C[:, -1] <= 1)).all():
        raise ValueError("invalid retrain accuracy values")
    L, C = L[:, :k], C[:, :k]
    masks = np.stack([np.random.RandomState(cfg["mask_seed"] * 100003 + j).random_sample(n) < cfg["p_drop"] for j in range(len(L))])
    pred_ranks = rankdata(-(masks.astype(np.float32) @ Y), axis=0)
    true_ranks = rankdata(L, axis=0)
    pred_ranks -= pred_ranks.mean(axis=0)
    true_ranks -= true_ranks.mean(axis=0)
    denom = np.linalg.norm(pred_ranks, axis=0) * np.linalg.norm(true_ranks, axis=0)
    defined = denom > 0
    lds = ((pred_ranks[:, defined] * true_ranks[:, defined]).sum(axis=0) / denom[defined]).mean() if defined.all() else None
    signal = L.std(axis=0, ddof=1)
    noise = C.std(axis=0, ddof=1)
    noise_ratio = (noise / signal).mean() if (signal > 0).all() else None
    G = criteria["gates"]
    checksum = hashlib.sha256()
    for path in paths:
        checksum.update(path.relative_to(root).as_posix().encode())
        with path.open("rb") as data:
            checksum.update(hashlib.file_digest(data, "sha256").digest())
    data = {"exact_ranges": lengths, "retrain_count": len(L), "noise_count": len(C), "source_sha256": checksum.hexdigest(),
            "exact_lds": None if lds is None else float(lds),
            "exact_pass": None if lds is None else bool(lds >= G["cifar_magic_lds_min"]),
            "undefined_lds_queries": np.flatnonzero(~defined).tolist(),
            "noise_over_signal": None if noise_ratio is None else float(noise_ratio),
            "noise_pass": None if noise_ratio is None else bool(noise_ratio <= G["noise_floor"]["max_noise_std_vs_signal_std"]),
            "undefined_noise_queries": np.flatnonzero(signal <= 0).tolist()}
    output = json.dumps(data, indent=2, allow_nan=False) + "\n"
    if a.out:
        a.out.write_text(output)
    print(output, end="")


if __name__ == "__main__":
    main()
