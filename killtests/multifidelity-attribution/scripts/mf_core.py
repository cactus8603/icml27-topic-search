"""MAGIC 類精確 metagradient 的 JAX 核心（SGD+momentum 展開、反向重播）。

改寫自 andrewilyas/simple-magic-jax（CC BY-NC-SA 4.0，非商業、需標註、相同方式分享）。
差異：不用 equinox（純 JAX pytree）；整個 step 當成一個函數做 vjp；檢查點全部留在 GPU；
一次重播可用 vmap 同時處理多個查詢（qchunk）。學術研究用途，改寫版同授權。
"""
import json
import math
import os
import numpy as np
import jax
import jax.numpy as jnp

from common import load_cifar, query_indices  # 資料函數放在 common.py（torch 環境也要用）


# ---------------------------------------------------------------- 模型
def _conv_init(key, cin, cout):
    k1, k2 = jax.random.split(key)
    b = 1 / math.sqrt(cin * 9)
    return {"w": jax.random.uniform(k1, (cout, cin, 3, 3), minval=-b, maxval=b),
            "b": jax.random.uniform(k2, (cout,), minval=-b, maxval=b),
            "g": jnp.ones(cout), "s": jnp.zeros(cout)}


class ResNet9:
    """與 simple-magic-jax cifar.py 同架構（寬度倍率 W_MULT=1.25 → W=80），NHWC 輸入、OIHW 權重。"""

    def __init__(self, width=80, bn_eps=1e-3, bn_mom=0.99):
        self.W, self.eps, self.mom = width, bn_eps, bn_mom
        W = width
        # (名稱, cin, cout, pool)
        self.spec = [("c0", 3, W, 0), ("c1", W, 2 * W, 1), ("r1a", 2 * W, 2 * W, 0), ("r1b", 2 * W, 2 * W, 0),
                     ("c2", 2 * W, 4 * W, 1), ("c3", 4 * W, 8 * W, 1), ("r2a", 8 * W, 8 * W, 0), ("r2b", 8 * W, 8 * W, 0)]

    def init(self, key):
        ks = jax.random.split(key, len(self.spec) + 1)
        params = {n: _conv_init(k, ci, co) for k, (n, ci, co, _) in zip(ks, self.spec)}
        b = 1 / math.sqrt(8 * self.W)
        k1, k2 = jax.random.split(ks[-1])
        params["fc"] = {"w": jax.random.uniform(k1, (10, 8 * self.W), minval=-b, maxval=b),
                        "b": jax.random.uniform(k2, (10,), minval=-b, maxval=b)}
        bufs = {n: {"m": jnp.zeros(co), "v": jnp.ones(co)} for n, _, co, _ in self.spec}
        return params, bufs

    def _block(self, p, bf, x, train, pool):
        x = jax.lax.conv_general_dilated(x, p["w"], (1, 1), "SAME", dimension_numbers=("NHWC", "OIHW", "NHWC")) + p["b"]
        if train:
            m, v = x.mean((0, 1, 2)), x.var((0, 1, 2))
            nb = {"m": self.mom * bf["m"] + (1 - self.mom) * m, "v": self.mom * bf["v"] + (1 - self.mom) * v}
        else:
            m, v, nb = bf["m"], bf["v"], bf
        x = (x - m) / jnp.sqrt(v + self.eps) * p["g"] + p["s"]
        x = jax.nn.gelu(x)  # approximate=True，與 jax.nn.gelu 預設（及 eqx 版）相同
        if pool:
            x = jax.lax.reduce_window(x, 0.0, jax.lax.add, (1, 2, 2, 1), (1, 2, 2, 1), "VALID") / 4.0
        return x, nb

    def apply(self, params, bufs, x, train):
        nb, h = {}, x
        res_in = None
        for n, _, _, pool in self.spec:
            if n in ("r1a", "r2a"):
                res_in = h
            h, nb[n] = self._block(params[n], bufs[n], h, train, pool)
            if n in ("r1b", "r2b"):
                h = res_in + h
        h = jax.lax.reduce_window(h, 0.0, jax.lax.add, (1, 4, 4, 1), (1, 4, 4, 1), "VALID") / 16.0
        h = h.reshape(h.shape[0], -1)
        logits = h @ params["fc"]["w"].T + params["fc"]["b"]
        return logits, nb


class MLP:
    """閘門用的小 MLP（無 BN、無緩衝）。"""

    def __init__(self, din, hidden, dout=2):
        self.d = (din, hidden, dout)

    def init(self, key):
        din, h, dout = self.d
        k = jax.random.split(key, 2)
        return {"l1": {"w": jax.random.normal(k[0], (din, h)) / math.sqrt(din), "b": jnp.zeros(h)},
                "l2": {"w": jax.random.normal(k[1], (h, dout)) / math.sqrt(h), "b": jnp.zeros(dout)}}, {}

    def apply(self, params, bufs, x, train):
        h = jnp.tanh(x @ params["l1"]["w"] + params["l1"]["b"])
        return h @ params["l2"]["w"] + params["l2"]["b"], bufs


def augment(key, imgs, pad=4):
    """reflect-pad 4 + 隨機裁切 + 隨機水平翻轉（NHWC）。"""
    kf, kh, kw = jax.random.split(key, 3)
    n = imgs.shape[0]
    flip = jax.random.bernoulli(kf, 0.5, (n, 1, 1, 1))
    hh = jax.random.randint(kh, (n,), 0, 2 * pad + 1)
    ww = jax.random.randint(kw, (n,), 0, 2 * pad + 1)
    p = jnp.pad(imgs, ((0, 0), (pad, pad), (pad, pad), (0, 0)), mode="reflect")
    crop = jax.vmap(lambda im, a, b: jax.lax.dynamic_slice(im, (a, b, 0), (imgs.shape[1], imgs.shape[2], imgs.shape[3])))(p, hh, ww)
    return jnp.where(flip, crop[:, :, ::-1, :], crop)


# ---------------------------------------------------------------- 訓練與重播
class Trainer:
    """資料加權的 SGD+momentum 訓練。w（長度 n）是 metaparameter：loss = sum(l_i * w_i)/BS。"""

    def __init__(self, model, X, Y, hp, use_aug=True):
        self.m, self.X, self.Y, self.hp, self.use_aug = model, jnp.asarray(X), jnp.asarray(Y), hp, use_aug
        self.n = self.X.shape[0]
        self.bs = hp["bs"]
        assert self.n % self.bs == 0
        self.spe = self.n // self.bs
        self.T = self.spe * hp["epochs"]
        ks = 1 + 1 / (1 - hp["momentum"])
        self.peak = hp["peak_lr_base"] / ks
        self.wd = hp["wd_base"] / ks
        self._step = jax.jit(self._step_impl)
        self._bwd = jax.jit(self._bwd_impl)
        self._qgrad = jax.jit(jax.vmap(self._qgrad_one, in_axes=(None, 0, 0)))
        self._qloss = jax.jit(jax.vmap(self._qloss_one, in_axes=(None, 0, 0)))

    # --- 排程與資料順序（皆為訓練 seed 的函數）
    def lr(self, t):
        T = self.T
        return float(np.interp(t, [0, int(0.15 * T), T], [0.1 * self.peak, self.peak, 0.0]))

    def perms(self, seed):
        """每個 epoch 的排列 → (T, bs) 索引。"""
        out = []
        for e in range(self.hp["epochs"]):
            p = np.random.RandomState(seed * 1000 + e).permutation(self.n)
            out.append(p.reshape(self.spe, self.bs))
        return jnp.asarray(np.concatenate(out), dtype=jnp.int32)

    def init_state(self, seed):
        params, bufs = self.m.init(jax.random.PRNGKey(seed))
        return (params, jax.tree_util.tree_map(jnp.zeros_like, params), bufs)

    def _step_impl(self, state, idx, key, w, lr, X, Y):
        params, mom, bufs = state
        x = X[idx]
        if self.use_aug:
            x = augment(key, x)
        y, wb = Y[idx], w[idx]

        def loss_f(p):
            logits, nb = self.m.apply(p, bufs, x, True)
            ce = -jnp.take_along_axis(jax.nn.log_softmax(logits), y[:, None], 1)[:, 0]
            return (ce * wb).sum() / self.bs, nb

        (_, nb), g = jax.value_and_grad(loss_f, has_aux=True)(params)
        g = jax.tree_util.tree_map(lambda gi, pi: gi + self.wd * pi, g, params)
        mom = jax.tree_util.tree_map(lambda m_, gi: self.hp["momentum"] * m_ + gi, mom, g)
        params = jax.tree_util.tree_map(lambda p_, m_: p_ - lr * m_, params, mom)
        return (params, mom, nb)

    def step(self, state, perm, key0, t, w):
        return self._step(state, perm[t], jax.random.fold_in(key0, t), w, jnp.asarray(self.lr(t)), self.X, self.Y)

    def train(self, w, seed, keep_every=None):
        """回傳最終 state；keep_every 不為 None 時另回傳 {step: state}（含 0 與 T，皆留在裝置上）。"""
        perm, key0 = self.perms(seed), jax.random.PRNGKey(seed + 777)
        state = self.init_state(seed)
        ck = {}
        for t in range(self.T):
            if keep_every and t % keep_every == 0:
                ck[t] = state
            state = self.step(state, perm, key0, t, w)
        if keep_every:
            ck[self.T] = state
            return state, ck
        return state

    # --- 查詢（測試例）損失：eval 模式（用 running stats）
    def _qloss_one(self, sb, x, y):
        params, bufs = sb
        logits, _ = self.m.apply(params, bufs, x[None], False)
        return -jax.nn.log_softmax(logits)[0, y]

    def _qgrad_one(self, sb, x, y):
        return jax.grad(self._qloss_one)(sb, x, y)

    def query_losses(self, state, xq, yq):
        return self._qloss((state[0], state[2]), xq, yq)

    # --- 反向重播
    def _bwd_impl(self, state, idx, key, w, lr, ds, X, Y):
        f = lambda s, w_: self._step_impl(s, idx, key, w_, lr, X, Y)
        _, vjp_fn = jax.vjp(f, state, w)
        return jax.vmap(vjp_fn)(ds)  # ds 的第 0 軸是查詢；primal 重算只做一次

    def metagrad(self, w, seed, xq, yq, ckpt_every=50, ck_and_final=None, max_bwd=None):
        """對 xq,yq 的每個查詢，回傳 dloss_q/dw（形狀 Q×n）與最終 state 上的查詢損失。"""
        perm, key0 = self.perms(seed), jax.random.PRNGKey(seed + 777)
        final, ck = ck_and_final if ck_and_final is not None else self.train(w, seed, keep_every=ckpt_every)
        losses = self.query_losses(final, xq, yq)
        gp, gb = self._qgrad((final[0], final[2]), xq, yq)
        ds = (gp, jax.tree_util.tree_map(jnp.zeros_like, gp), gb)  # 每個葉子的第 0 軸是查詢
        Q = xq.shape[0]
        dY = jnp.zeros((Q, self.n), jnp.float32)
        nb_done = 0
        starts = sorted(k for k in ck if k < self.T)
        for s0 in reversed(starts):
            s1 = min(s0 + ckpt_every, self.T)
            seg = [ck[s0]]
            for t in range(s0, s1 - 1):  # 重算段內狀態
                seg.append(self.step(seg[-1], perm, key0, t, w))
            for t in reversed(range(s0, s1)):
                ds, dw = self._bwd(seg[t - s0], perm[t], jax.random.fold_in(key0, t), w, jnp.asarray(self.lr(t)), ds, self.X, self.Y)
                dY = dY + dw
                nb_done += 1
                if max_bwd and nb_done >= max_bwd:
                    return np.asarray(dY), np.asarray(losses)  # bench 用：只做前 max_bwd 步
            del seg
        return np.asarray(dY), np.asarray(losses)


def default_hp():
    return dict(bs=500, epochs=20, momentum=0.85, peak_lr_base=0.05, wd_base=1e-4)


def load_cfg(path):
    with open(path) as f:
        return json.load(f)


from common import removal_mask_np as removal_mask  # noqa: 同一函數，兩個環境共用
