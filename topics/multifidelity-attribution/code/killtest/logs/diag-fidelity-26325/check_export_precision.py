import argparse
import hashlib
import json
from pathlib import Path

import numpy as np


def jax_logits(config, dest):
    import jax
    import jax.numpy as jnp
    from mf_core import ResNet9, load_cifar, query_indices

    devices = jax.devices()
    if devices[0].platform != "gpu":
        raise RuntimeError(f"expected a GPU, saw {devices}")
    print(f"JAX {jax.__version__}, devices={devices}", flush=True)
    path = Path(config["run_dir"]) / "models" / f"model_s{config['train_seed']}.npz"
    params, buffers = {}, {}
    with np.load(path) as checkpoint:
        ref = checkpoint["ref_logits"]
        for key in checkpoint.files:
            if key.startswith(("p.", "b.")):
                kind, layer, attr = key.split(".")
                target = params if kind == "p" else buffers
                target.setdefault(layer, {})[attr] = jnp.asarray(checkpoint[key])
    _, _, images, _ = load_cifar(1)
    selected = query_indices(config)[:64]
    x = jnp.asarray(images[selected])
    model = ResNet9(config["width"])
    default = np.asarray(model.apply(params, buffers, x, False)[0])
    with jax.default_matmul_precision("highest"):
        highest = np.asarray(model.apply(params, buffers, x, False)[0])
    with path.open("rb") as stream:
        sha256 = hashlib.file_digest(stream, "sha256").hexdigest()
    np.savez(dest / "jax_logits.npz", saved=ref, default=default, highest=highest,
             query_indices=selected, checkpoint_sha256=sha256)


def torch_logits(config, dest):
    import torch
    from cheap_torch import load_model, to_t
    from common import load_cifar, query_indices

    if not torch.cuda.is_available():
        raise RuntimeError("expected a CUDA GPU")
    print(f"PyTorch {torch.__version__}, device={torch.cuda.get_device_name(0)}", flush=True)
    path = Path(config["run_dir"]) / "models" / f"model_s{config['train_seed']}.npz"
    model, _ = load_model(config["width"], path)
    _, _, images, _ = load_cifar(1)
    x = to_t(images[query_indices(config)[:64]]).to("cuda")
    with torch.no_grad():
        full = model(x).cpu().numpy()
        torch.backends.cudnn.allow_tf32 = True
        torch.backends.cuda.matmul.allow_tf32 = True
        tf32 = model(x).cpu().numpy()
    np.savez(dest / "torch_logits.npz", full=full, tf32=tf32)


def compare(config, dest):
    import itertools

    groups = {}
    for label in ("jax", "torch"):
        with np.load(dest / f"{label}_logits.npz") as data:
            for key in data.files:
                if data[key].shape == (64, 10):
                    groups[f"{label}_{key}"] = data[key]
    path = Path(config["run_dir"]) / "models" / f"model_s{config['train_seed']}.npz"
    with path.open("rb") as stream:
        sha256 = hashlib.file_digest(stream, "sha256").hexdigest()
    errors = {}
    for (name1, arr1), (name2, arr2) in itertools.combinations(groups.items(), 2):
        assert arr1.shape == arr2.shape == (64, 10)
        assert np.isfinite(arr1).all() and np.isfinite(arr2).all()
        err = np.abs(arr1 - arr2)
        errors[f"{name1}__{name2}"] = {"max": float(err.max()), "median": float(np.median(err))}
    report = {"checkpoint_sha256": sha256, "same_saved_inputs": 64, "errors": errors}
    (dest / "paired_error.json").write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--cfg", required=True, type=Path)
    parser.add_argument("--out", required=True, type=Path)
    parser.add_argument("backend", choices=("jax", "torch", "compare"))
    opts = parser.parse_args()
    opts.out.mkdir(exist_ok=True, parents=True)
    cfg = json.loads(opts.cfg.read_text())
    {"jax": jax_logits, "torch": torch_logits, "compare": compare}[opts.backend](cfg, opts.out)
