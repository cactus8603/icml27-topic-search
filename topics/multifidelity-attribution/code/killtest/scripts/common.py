import os, sys, json
import numpy as np
sys.path.insert(0, os.path.dirname(__file__))
HERE = os.path.dirname(os.path.abspath(__file__))
RES = os.path.join(HERE, "..", "results")


def cfg_paths(cfg):
    rd = cfg["run_dir"]
    d = dict(models=f"{rd}/models", exact=f"{rd}/exact", retrain=f"{rd}/retrain", cheap=f"{rd}/cheap",
             res=os.path.join(RES, cfg["tag"]))
    for v in d.values():
        os.makedirs(v, exist_ok=True)
    return d


CIFAR_DIR = os.environ.get("CIFAR_DIR", "/work/ChihChungHsu/cactus8603/icml/data/cifar10/cifar-10-batches-py")


def load_cifar(n_train=None):
    import pickle

    def rd(f):
        with open(os.path.join(CIFAR_DIR, f), "rb") as fh:
            d = pickle.load(fh, encoding="bytes")
        return d[b"data"].reshape(-1, 3, 32, 32).transpose(0, 2, 3, 1), np.array(d[b"labels"])

    xs, ys = zip(*[rd(f"data_batch_{i}") for i in range(1, 6)])
    xtr, ytr = np.concatenate(xs), np.concatenate(ys)
    xte, yte = rd("test_batch")
    if n_train:
        xtr, ytr = xtr[:n_train], ytr[:n_train]
    f = lambda x: (x.astype(np.float32) / 255.0)  # NHWC，與 simple-magic 一樣只除以 255
    return f(xtr), ytr.astype(np.int32), f(xte), yte.astype(np.int32)


def query_indices(cfg):
    """固定的查詢集：測試集以固定 seed 排列後取前 k_total 個。stage0 取前 k_stage0 個。"""
    rs = np.random.RandomState(cfg["query_seed"])
    return rs.permutation(10000)[: cfg["k_total"]]




def removal_mask_np(cfg, j, n):
    """與 mf_core.removal_mask 相同（numpy 版，供無 JAX 的環境使用）。"""
    if j < 0:
        return np.zeros(n, bool)
    rs = np.random.RandomState(cfg["mask_seed"] * 100003 + j)
    return rs.random_sample(n) < cfg["p_drop"]
