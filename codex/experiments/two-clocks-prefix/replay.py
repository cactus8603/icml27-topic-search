"""Replay the first 36 epochs of the frozen u833 control for an exploratory boundary check."""

import argparse
import ast
from collections import Counter
import hashlib
import json
import os
from pathlib import Path
import subprocess
import sys

import numpy as np


FORMAL_COMMIT = "9e92804a7bfded3c5cafd4fccd4315df2034eca5"
PREFIX_EPOCHS = (1, 2, 4, 8, 12, 18, 24, 30, 33, 36)
ANCHORS = (30, 33, 36)
FORMAL_FIRST = {(0, 3), (0, 5), (1, 3), (2, 3), (2, 5), (3, 4), (3, 5)}


def read_json(path):
    return json.loads(path.read_text(encoding="utf-8"))


def sha256(path):
    with path.open("rb") as stream:
        return hashlib.file_digest(stream, "sha256").hexdigest()


def prefix_trainer(source):
    """Preserve the training loop, including its original scheduler horizon."""
    replacements = {
        "ck_epochs = ckpt_schedule(args.first_ckpt_epoch, args.epochs, args.max_ratio)":
            f"ck_epochs = {list(PREFIX_EPOCHS)!r}",
        "for ep in range(1, args.epochs + 1):":
            f"for ep in range(1, {PREFIX_EPOCHS[-1]} + 1):",
    }
    for old, new in replacements.items():
        if source.count(old) != 1:
            raise ValueError("pinned trainer instrumentation point differs")
        source = source.replace(old, new)
    compile(source, "<prefix trainer>", "exec")
    return source


def load_curve_class(source, normalization_indices=None):
    """Load only the original curve processing, without reading its formal result inputs."""
    from schedule import WEEK2_T_GRID, week2_epochs

    script = source / "scripts/aggregate_week2.py"
    code = script.read_text(encoding="utf-8")
    if normalization_indices is not None:
        old = "m = y.mean(axis=1, keepdims=True)"
        if code.count(old) != 1:
            raise ValueError("pinned normalization expression differs")
        code = code.replace(old, "m = y[:, normalization_indices, :].mean(axis=1, keepdims=True)")
    tree = ast.parse(code)
    names = ("smooth", "tstar", "interior_info", "Run")
    parts = [node for node in tree.body if isinstance(node, (ast.FunctionDef, ast.ClassDef)) and node.name in names]
    if tuple(node.name for node in parts) != names:
        raise ValueError("pinned curve definitions differ")
    criteria = read_json(source / "configs/criteria_week2.json")
    mask = read_json(source / "configs/t_mask_hier.json")
    scope = {
        "np": np, "json": json, "week2_epochs": week2_epochs, "WEEK2_T_GRID": WEEK2_T_GRID,
        "K": criteria["smooth_window"], "RISE": criteria["interior_min_rise"], "KEEP": mask["keep_t"],
        "normalization_indices": normalization_indices,
    }
    exec(compile(ast.Module(parts, type_ignores=[]), str(script), "exec"), scope)
    return scope["Run"]


def preflight(source, original, data):
    from schedule import week2_epochs
    from week2_identity import receipt_error, training_error

    data_sha = sha256(data)
    if data_sha != read_json(source / "configs/t_mask_hier.json")["data_sha256"]:
        raise ValueError("hierarchical data checksum differs from the formal run")
    for pick in range(6):
        run = original / f"pick{pick}"
        meta_path = run / "meta.json"
        meta = read_json(meta_path)
        if error := training_error(meta, "uncond_only", pick):
            raise ValueError(error)
        if meta["n_c"] < 700 or Path(meta["data"]).resolve() != data.resolve() or Path(meta["out"]).resolve() != run.resolve():
            raise ValueError(f"pick {pick}: registered training paths or sample count differ")
        receipt = read_json(run / "completion.json")
        if error := receipt_error(receipt, sha256(meta_path), data_sha):
            raise ValueError(f"pick {pick}: {error}")
        for epoch in week2_epochs():
            weight = run / f"ep{epoch}.pt"
            if not weight.is_file() or weight.stat().st_size == 0:
                raise ValueError(f"pick {pick}: formal checkpoint {epoch} is missing")
    return data_sha


def compare_anchor(original, replayed):
    import torch

    old_sha, new_sha = sha256(original), sha256(replayed)
    if old_sha == new_sha:
        return {"match": True, "comparison": "sha256", "original_sha256": old_sha, "replay_sha256": new_sha}
    old = torch.load(original, map_location="cpu", weights_only=True)
    new = torch.load(replayed, map_location="cpu", weights_only=True)
    same = old.keys() == new.keys() and all(
        new[name].dtype == value.dtype and torch.equal(new[name], value) for name, value in old.items()
    )
    return {"match": same, "comparison": "tensor_equal", "original_sha256": old_sha, "replay_sha256": new_sha}


def run_prefixes(source, work, data, original, trainer_source):
    from schedule import week2_epochs

    script = source / "scripts/replay_prefix_trainer.py"
    script.write_text(trainer_source, encoding="utf-8")
    anchors, views = {}, []
    for pick in range(6):
        formal_run = original / f"pick{pick}"
        prefix_run = work / f"prefix-{pick}"
        subprocess.run([
            sys.executable, "-IB", str(script), "--data", str(data), "--out", str(prefix_run),
            "--pick", str(pick), "--seed", str(pick), "--n", "5000", "--epochs", "20000",
            "--pdrop", "1.0", "--batch", "85", "--class-only", "0", "--device", "cuda",
        ], check=True, stdout=subprocess.DEVNULL)
        anchors[str(pick)] = {
            str(epoch): compare_anchor(formal_run / f"ep{epoch}.pt", prefix_run / f"ep{epoch}.pt")
            for epoch in ANCHORS
        }
        if not all(item["match"] for item in anchors[str(pick)].values()):
            return anchors, None
        view = work / f"view-{pick}"
        view.mkdir()
        (view / "meta.json").symlink_to(formal_run / "meta.json")
        for epoch in week2_epochs():
            (view / f"ep{epoch}.pt").symlink_to(formal_run / f"ep{epoch}.pt")
        for epoch in PREFIX_EPOCHS:
            if epoch < ANCHORS[0]:
                (view / f"ep{epoch}.pt").symlink_to(prefix_run / f"ep{epoch}.pt")
        views.append(view)
        print(f"[codex] pick {pick}: replayed 36 epochs and matched three anchors", flush=True)
    return anchors, views


def summarize_curves(extended, formal, fixed):
    branch = "uncond"
    if any(run.br != [branch] or not np.array_equal(run.pairs, formal.pairs) for run in (extended, formal, fixed)):
        raise ValueError("unexpected diagnostic curve layout")
    old = [formal.inter(branch, i) for i in range(formal.P)]
    first_indices = [i for i, (_, reason) in enumerate(old) if reason == "first"]
    matching_baseline = {tuple(formal.pairs[i]) for i in first_indices} == FORMAL_FIRST
    if not matching_baseline:
        return {"status": "formal_boundary_recompute_mismatch", "n_pairs": formal.P,
                "recomputed_formal_first_count": len(first_indices)}
    def count(states):
        return dict(Counter("interior" if valid else reason for valid, reason in states))

    formal_indices = [fixed.ep.tolist().index(epoch) for epoch in formal.ep]
    if not np.allclose(fixed.curve[branch][:, formal_indices], formal.curve[branch], rtol=1e-12, atol=1e-12):
        raise ValueError("fixed-scale curves changed the original-grid normalized values")

    def boundary_report(run):
        states = [run.inter(branch, i) for i in range(run.P)]
        return {
            "pair_counts": count(states),
            "former_first_grid_pairs": count([states[i] for i in first_indices]),
            "former_first_now_interior_before_epoch_30": sum(
                states[i][0] and run.ts(branch, i) < 30 for i in first_indices
            ),
        }

    return {
        "status": "exploratory_only",
        "n_pairs": formal.P,
        "original_grid_recomputed": count(old),
        "extended_grid": boundary_report(extended),
        "extended_grid_original_normalizers": boundary_report(fixed),
    }


def evaluate(source, work, data, views, curve_class):
    from schedule import week2_epochs

    grid = sorted(set(PREFIX_EPOCHS).union(week2_epochs()))
    result = work / "extended.npz"
    subprocess.run([
        sys.executable, "-IB", str(source / "scripts/eval_hier_oracle.py"), "--data", str(data),
        "--runs", *(str(view) for view in views), "--device", "cuda", "--epochs", ",".join(map(str, grid)),
        "--class-filter", "0", "--n-eval", "700", "--guidance", "0", "--out", str(result),
    ], check=True, stdout=subprocess.DEVNULL)
    extended = curve_class(result)
    old_grid = work / "original_grid.npz"
    indices = [grid.index(epoch) for epoch in week2_epochs()]
    with np.load(result, allow_pickle=False) as rows:
        fields = {k: rows[k] for k in ("branches", "pairs", "picks", "t_grid")}
        if extended.ep.tolist() != grid or extended.picks != list(range(6)):
            raise ValueError("evaluation returned unexpected epoch grid or training picks")
        np.savez(old_grid, epochs=week2_epochs(), pair_cos_eps=rows["pair_cos_eps"][:, indices], **fields)
    fixed = load_curve_class(source, normalization_indices=indices)(result)
    return summarize_curves(extended, curve_class(old_grid), fixed)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source", required=True, type=Path)
    parser.add_argument("--data", required=True, type=Path)
    parser.add_argument("--original", required=True, type=Path)
    parser.add_argument("--work", required=True, type=Path)
    parser.add_argument("--output", type=Path)
    parser.add_argument("--commit", required=True)
    args = parser.parse_args()
    source = args.source.resolve()
    sys.path.insert(0, str(source / "scripts"))
    trainer = prefix_trainer((source / "scripts/train_hier_cond_w2.py").read_text(encoding="utf-8"))
    curve_class = load_curve_class(source)
    data_sha = preflight(source, args.original, args.data)
    if args.output is None:
        print("[codex] inputs, frozen code and six control receipts verified")
        return

    import torch

    if not torch.cuda.is_available() or "H200" not in torch.cuda.get_device_name(0):
        raise SystemExit("replay requires one H200 to match the formal training backend")
    args.output.mkdir(parents=True, exist_ok=False)
    anchors, views = run_prefixes(source, args.work, args.data, args.original, trainer)
    result = {
        "commit": args.commit,
        "formal_training_commit": FORMAL_COMMIT,
        "gpu": torch.cuda.get_device_name(0),
        "torch_version": torch.__version__,
        "data_sha256": data_sha,
        "prefix_epochs": list(PREFIX_EPOCHS),
        "anchors": anchors,
    }
    if views is None:
        result["diagnostic"] = {"status": "checkpoint_anchor_mismatch"}
    else:
        result["diagnostic"] = evaluate(source, args.work, args.data, views, curve_class)
    report = args.output / "results.json"
    pending = args.output / ".results.json.pending"
    pending.write_text(json.dumps(result, indent=2, allow_nan=False) + "\n", encoding="utf-8")
    os.replace(pending, report)
    print(f"[codex] {result['diagnostic']['status']} {report}", flush=True)
    if result["diagnostic"]["status"] != "exploratory_only":
        raise SystemExit(2)


if __name__ == "__main__":
    main()
