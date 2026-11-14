"""閘門 G1：精確 metagradient 在小 MLP 上對照 (a) 中央有限差分 (b) 實際留一重訓。float64、CPU 可跑。
通過條件寫在 configs/criteria.json 的 gates.mlp。輸出 results/gates/mlp_magic.json。"""
import json, os, sys
import numpy as np
import jax
jax.config.update("jax_enable_x64", True)
import jax.numpy as jnp
sys.path.insert(0, os.path.dirname(__file__))
import mf_core as mc
from common import RES

crit = json.load(open(os.path.join(os.path.dirname(__file__), "..", "configs", "criteria.json")))["gates"]["mlp"]
rs = np.random.RandomState(0)
n, din = 48, 5
X = rs.randn(n, din); Y = (X[:, 0] * X[:, 1] + 0.3 * rs.randn(n) > 0).astype(np.int32)
xq = jnp.asarray(rs.randn(4, din)); yq = jnp.asarray((np.asarray(xq)[:, 0] * np.asarray(xq)[:, 1] > 0).astype(np.int32))
hp = dict(bs=8, epochs=6, momentum=0.85, peak_lr_base=0.5, wd_base=1e-3)
tr = mc.Trainer(mc.MLP(din, 16), X, Y, hp, use_aug=False)
w = jnp.ones(n)
qloss = lambda w_: np.asarray(tr.query_losses(tr.train(w_, 0), xq, yq))
dY, base = tr.metagrad(w, 0, xq, yq, ckpt_every=4)
assert np.allclose(base, qloss(w), atol=1e-10)
# (a) 中央有限差分
eps, errs = 1e-5, []
for i in rs.choice(n, 12, replace=False):
    e = np.zeros(n); e[i] = eps
    fd = (qloss(w + e) - qloss(w - e)) / (2 * eps)
    errs.append(np.abs(fd - dY[:, i]).max() / (np.abs(fd).max() + 1e-12))
# (b) 實際留一：Δloss_i = loss(w_i=0) - loss(w)；預測 -Y_i
loo = np.stack([qloss(np.where(np.arange(n) == i, 0.0, 1.0)) - base for i in range(n)])  # n×Q
pear = [float(np.corrcoef(loo[:, q], -dY[q])[0, 1]) for q in range(dY.shape[0])]
res = dict(fd_rel_err_max=float(max(errs)), loo_pearson=pear, loo_pearson_mean=float(np.mean(pear)))
res["pass"] = bool(res["fd_rel_err_max"] < crit["fd_rel_err_max"] and res["loo_pearson_mean"] > crit["loo_pearson_min"])
os.makedirs(os.path.join(RES, "gates"), exist_ok=True)
json.dump(res, open(os.path.join(RES, "gates", "mlp_magic.json"), "w"), indent=1)
print(json.dumps(res, indent=1)); sys.exit(0 if res["pass"] else 1)
