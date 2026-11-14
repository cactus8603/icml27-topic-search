"""第 2 週正式輸入的角色規格與訓練來源檢查。"""
import hashlib
import json
import math
from pathlib import Path

from schedule import week2_epochs


ROLES = {
    "k6": ("normal", None, 1000, (1., 3.), True),
    "k1": ("const", None, 1000, (1., 3.), False),
    "k6c0": ("normal", 0, 700, (3.,), False),
    "u833c0": ("uncond_only", 0, 700, (), False),
    "k1c0": ("const", 0, 700, (3.,), False),
}
TRAINING_BASE = {
    "n": 5000, "epochs": 20000, "lr": 3e-4, "t_final": 500,
    "hidden": 512, "nhead": 4, "layers": 8, "first_ckpt_epoch": 30, "max_ratio": 1.10,
    "save_dtype": "fp16", "amp": "bf16", "compile": False, "q": 6, "depth": 4,
}


def sha256(path):
    with Path(path).open("rb") as fh:
        return hashlib.file_digest(fh, "sha256").hexdigest()


def expected_eval(role):
    kind, class_filter, n_eval, guidance, gamma = ROLES[role]
    return {"role": role, "run_type": kind, "class_filter": class_filter, "n_eval": n_eval, "guidance": list(guidance),
            "gamma": gamma, "block_size": 5000, "test_block": 7, "eval_seed": 123}


def actual_eval(role, run_type, class_filter, n_eval, guidance, gamma, block_size, test_block, seed):
    # unconditional-only evaluation suppresses all guidance branches.
    return {"role": role, "run_type": run_type, "class_filter": class_filter, "n_eval": n_eval,
            "guidance": list(guidance) if run_type != "uncond_only" else [], "gamma": gamma,
            "block_size": block_size, "test_block": test_block, "eval_seed": seed}


def expected_training(kind, pick):
    unconditional = kind == "uncond_only"
    return {**TRAINING_BASE, "pick": pick, "seed": pick, "batch": 85 if unconditional else 512,
            "pdrop": 1.0 if unconditional else 0.1, "const_label": 0 if kind == "const" else None,
            "class_only": 0 if unconditional else None, "ckpt_epochs": week2_epochs()}


def training_error(meta, kind, pick):
    expected = expected_training(kind, pick)
    for key, value in expected.items():
        if (value is not None and key not in meta) or meta.get(key) != value:
            return f"pick {pick} {key}={meta.get(key)!r}，規格為 {value!r}"
    if kind in ("const", "uncond_only"):
        size = meta.get("n_c")
        if (meta.get("block_size") != expected["n"] or not isinstance(size, int) or not 0 < size <= expected["n"]
                or meta.get("steps_per_epoch") != math.ceil(size / expected["batch"])
                or (kind == "const" and size != expected["n"])):
            return f"pick {pick} 控制組訓練樣本數或 steps_per_epoch 不符"
    return None


def is_legacy(kind, pick):
    return kind == "normal" and pick in (0, 1, 2)


def receipt_error(receipt, meta_sha, data_sha):
    if (not isinstance(receipt, dict) or receipt.get("version") != 1 or receipt.get("meta_sha256") != meta_sha
            or receipt.get("data_sha256") != data_sha or receipt.get("epochs") != week2_epochs()):
        return "completion.json 的版本、資料、訓練 metadata 或 checkpoint 排程不符"
    return None


def read_runs(run_paths, metas, kind, data_file, data_sha):
    runs = []
    for path, meta in zip(run_paths, metas):
        pick = meta.get("pick")
        error = training_error(meta, kind, pick)
        if error:
            raise ValueError(error)
        run_dir = Path(path)
        if Path(meta["data"]).resolve() != Path(data_file).resolve() or Path(meta["out"]).resolve() != run_dir.resolve():
            raise ValueError(f"{run_dir} 的訓練資料或 metadata 路徑不符")
        meta_sha = sha256(run_dir / "meta.json")
        receipt_file = run_dir / "completion.json"
        receipt = json.loads(receipt_file.read_text()) if receipt_file.is_file() else None
        if receipt is not None or not is_legacy(kind, pick):
            error = receipt_error(receipt, meta_sha, data_sha)
            if error:
                raise ValueError(f"{run_dir}：{error}")
        expected = expected_training(kind, pick)
        training = {key: meta.get(key) for key in expected}
        if kind in ("const", "uncond_only"):
            training.update({key: meta[key] for key in ("block_size", "n_c", "steps_per_epoch")})
        runs.append({"training": training, "meta_sha256": meta_sha, "completion": receipt})
    return runs


def manifest_error(manifest, role, picks, data_sha):
    if not isinstance(manifest, dict) or manifest.get("version") != 1:
        return "缺正式評估 manifest"
    if manifest.get("evaluation") != expected_eval(role) or manifest.get("data_sha256") != data_sha:
        return "評估角色、評估設定或資料識別碼不符"
    runs = manifest.get("runs")
    if not isinstance(runs, list) or len(runs) != len(picks):
        return "manifest 缺完整訓練來源"
    kind = ROLES[role][0]
    for pick, run in zip(picks, runs):
        if not isinstance(run, dict) or not isinstance(run.get("training"), dict):
            return f"pick {pick} 缺訓練來源"
        error = training_error(run["training"], kind, pick)
        if error:
            return error
        meta_sha = run.get("meta_sha256")
        if not isinstance(meta_sha, str) or len(meta_sha) != 64 or any(c not in "0123456789abcdef" for c in meta_sha):
            return f"pick {pick} 缺訓練 metadata SHA-256"
        receipt = run.get("completion")
        if receipt is not None or not is_legacy(kind, pick):
            error = receipt_error(receipt, meta_sha, data_sha)
            if error:
                return f"pick {pick} {error}"
    return None
