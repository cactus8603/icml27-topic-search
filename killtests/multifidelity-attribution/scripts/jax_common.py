import os, sys
import numpy as np
import jax
sys.path.insert(0, os.path.dirname(__file__))
import mf_core as mc
from common import cfg_paths


def setup(cfg_path, n_train_override=None):
    # 審查修正 2026-11-14：所有 JAX 工作（訓練、重訓、精確 Y）啟動時都印裝置，REQUIRE_GPU=1 時無 GPU 即中止
    print("jax.devices():", jax.devices(), flush=True)
    if os.environ.get("REQUIRE_GPU") == "1" and jax.devices()[0].platform == "cpu":
        sys.exit("REQUIRE_GPU=1 但 JAX 只看到 CPU：中止（檢查 jax[cuda13] 與驅動）")
    cfg = mc.load_cfg(cfg_path)
    if cfg.get("matmul_precision", "default") != "default":
        jax.config.update("jax_default_matmul_precision", cfg["matmul_precision"])
    xtr, ytr, xte, yte = mc.load_cifar(cfg["n_train"])
    qi = mc.query_indices(cfg)
    xq, yq = xte[qi], yte[qi]
    tr = mc.Trainer(mc.ResNet9(cfg["width"]), xtr, ytr, cfg["hp"])
    return cfg, tr, jax.numpy.asarray(xq), jax.numpy.asarray(yq), (xte, yte), cfg_paths(cfg)


def weights_for(cfg, j, n):
    return jax.numpy.asarray(1.0 - mc.removal_mask(cfg, j, n), jax.numpy.float32)
