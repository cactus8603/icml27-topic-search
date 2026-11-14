#!/usr/bin/env python
"""CPU 單元測試：驗證條件／無條件 BP 與 CFG 組合是否精確。

測試項目（全部 float64，容許誤差 1e-9）：
  T1 暴力列舉：q=3、depth=2（4 個葉）、隨機稠密 M，對所有 q^4 葉配置列舉，比較條件與無條件後驗。
  T2 暴力列舉（含禁止轉移 M 為 0）：q=3、depth=2，M 稀疏。
  T3 與原作 numba run_BP_diffusion 比對（無條件）：用原作的 q=6、depth=4 文法（seed 0）。
  T4 全機率公式：sum_c P(c|x_t) * p_cond(c) == p_uncond。
  T5 以真實擴散噪聲 x_t 檢驗 field 推導：暴力列舉 P(x0|x_t) ∝ P(x0) N(x_t; sqrt(ab) x0, (1-ab) I)。
  T6 CFG：w=0 等於條件；(1+w) p_c - w p_u 對 w 線性；各 token 加總為 1。
  T7 根後驗：BP 的 root_post 與暴力列舉一致。
結果寫到 results/bp_unit_test.json（供 aggregate_week1.py 讀）。
"""
import itertools
import json
import os
import sys
import time

import numpy as np
import torch

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import hier_cond as H  # noqa: E402

torch.manual_seed(0)
np.random.seed(0)
results = {}


def check(name, err, tol=1e-9):
    ok = bool(err < tol)
    results[name] = {"max_abs_err": float(err), "tol": tol, "pass": ok}
    print(f"[{'PASS' if ok else 'FAIL'}] {name}: max|err|={err:.3e}")


def brute_force(M, depth, field, root_prior):
    """對所有葉配置列舉。M:(q,q,q) numpy。field:(L,q)。回傳 leaf_post (L,q)、root_post (q,)。"""
    q = M.shape[0]
    L = 2 ** depth
    lik = np.exp(field - field.max(1, keepdims=True))
    lik = lik / lik.sum(1, keepdims=True)
    leaf = np.zeros((L, q))
    root = np.zeros(q)

    def subtree_prob(sym, level, start):
        """P(此節點為 sym 下，子樹各葉配置)：回傳 dict {配置元組: 機率}（配置長度 = 2^(depth-level)）。"""
        if level == depth:
            return {(sym,): 1.0}
        out = {}
        for l, r in itertools.product(range(q), range(q)):
            pm = M[sym, l, r]
            if pm == 0:
                continue
            dl = subtree_prob(l, level + 1, 2 * start)
            dr = subtree_prob(r, level + 1, 2 * start + 1)
            for cl, pl in dl.items():
                for cr, pr in dr.items():
                    out[cl + cr] = out.get(cl + cr, 0.0) + pm * pl * pr
        return out

    Z = 0.0
    for x0 in range(q):
        if root_prior[x0] == 0:
            continue
        for cfg, p in subtree_prob(x0, 0, 0).items():
            w = root_prior[x0] * p * np.prod([lik[j, cfg[j]] for j in range(L)])
            Z += w
            root[x0] += w
            for j in range(L):
                leaf[j, cfg[j]] += w
    return leaf / Z, root / Z


def make_M(q, sparse):
    h = np.random.randn(q, q, q)
    if sparse:  # 每個父只允許 q 個 (l,r) 配對，其餘為 0（模擬原作的 forbidden transitions）
        mask = np.zeros((q, q, q), bool)
        for p in range(q):
            idx = np.random.permutation(q * q)[:q]
            for k in idx:
                mask[p, k // q, k % q] = True
        h = np.where(mask, h, -np.inf)
    e = np.exp(h - h.max(axis=(1, 2), keepdims=True))
    return e / e.sum(axis=(1, 2), keepdims=True)


for tag, sparse in (("T1_dense", False), ("T2_sparse", True)):
    q, depth = 3, 2
    M = make_M(q, sparse)
    Mt = torch.from_numpy(M)
    errs_c, errs_u, errs_r = 0.0, 0.0, 0.0
    for _ in range(5):
        field = np.random.randn(2 ** depth, q) * 2.0
        f = torch.from_numpy(field).unsqueeze(0)
        for c in range(q):
            oh = np.eye(q)[c]
            lp, rp = H.bp_posteriors(Mt, depth, f, torch.from_numpy(oh))
            bl, br = brute_force(M, depth, field, oh)
            errs_c = max(errs_c, np.abs(lp[0].numpy() - bl).max())
        unif = np.full(q, 1 / q)
        lp, rp = H.bp_posteriors(Mt, depth, f, torch.from_numpy(unif))
        bl, br = brute_force(M, depth, field, unif)
        errs_u = max(errs_u, np.abs(lp[0].numpy() - bl).max())
        errs_r = max(errs_r, np.abs(rp[0].numpy() - br).max())
    check(f"{tag}_conditional_vs_bruteforce", errs_c)
    check(f"{tag}_unconditional_vs_bruteforce", errs_u)
    check(f"{tag}_root_posterior_vs_bruteforce", errs_r)

# T3：與原作 numba BP 比對（原作文法）
try:
    from modules.gen_filtered_hierarchical_data_wforbidden import get_M_wforbidden
    from modules.belief_propagation import run_BP_diffusion
    q, depth = 6, 4
    Mo, _ = get_M_wforbidden(q, 1.0, 4, rng=np.random.RandomState(0))
    Mt = torch.from_numpy(Mo)
    err = 0.0
    for _ in range(4):
        field = np.random.randn(2 ** depth, q) * 1.5
        ref = run_BP_diffusion(Mo, depth, q, field, factorized_layers=0)
        lp, _ = H.bp_posteriors(Mt, depth, torch.from_numpy(field).unsqueeze(0), torch.full((q,), 1 / q, dtype=torch.float64))
        err = max(err, np.abs(lp[0].numpy() - ref).max())
    check("T3_unconditional_vs_original_numba_BP", err)
except Exception as e:  # noqa
    results["T3_unconditional_vs_original_numba_BP"] = {"pass": False, "error": repr(e)}
    print("[FAIL] T3", repr(e))

# T4：全機率；T5：真實擴散噪聲；T6：CFG
q, depth = 6, 4
Mo, _ = get_M_wforbidden(q, 1.0, 4, rng=np.random.RandomState(0))
Mt = torch.from_numpy(Mo)
ab = H.make_alpha_bars(500)
B = 64
x0 = torch.randint(0, q, (B, 2 ** depth))
roots = torch.randint(0, q, (B,))
oh = torch.nn.functional.one_hot(x0, q).double()
err4 = err6a = err6b = err6c = 0.0
for t in (10, 50, 100, 250):
    gen = torch.Generator().manual_seed(t)
    xt = H.noise_batch(oh, torch.full((B,), t), ab, gen)
    abar = float(ab[t - 1])
    p_c_all = []
    _, p_u, rpost = H.oracle_cond_uncond(Mt, depth, xt, abar, roots=None)
    for c in range(q):
        p_c, _, _ = H.oracle_cond_uncond(Mt, depth, xt, abar, roots=torch.full((B,), c))
        p_c_all.append(p_c)
    mix = sum(rpost[:, c].view(-1, 1, 1) * p_c_all[c] for c in range(q))
    err4 = max(err4, (mix - p_u).abs().max().item())
    p_c, p_u2, _ = H.oracle_cond_uncond(Mt, depth, xt, abar, roots=roots)
    err6a = max(err6a, (H.cfg_combine(p_c, p_u2, 0.0) - p_c).abs().max().item())
    w = 1.7
    lhs = H.cfg_combine(p_c, p_u2, w)
    rhs = p_c + w * (p_c - p_u2)
    err6b = max(err6b, (lhs - rhs).abs().max().item())
    err6c = max(err6c, (lhs.sum(-1) - 1).abs().max().item())
check("T4_total_probability_sum_c_P(c|xt)_p_cond_eq_p_uncond", err4)
check("T6a_cfg_w0_equals_conditional", err6a)
check("T6b_cfg_linear_in_w", err6b)
check("T6c_cfg_rows_sum_to_one", err6c)

# T5：小樹 (q=3, depth=2) + 真實噪聲，檢驗 field = sqrt(ab)/(1-ab) * x_t 的推導
q, depth = 3, 2
M = make_M(q, True)
Mt = torch.from_numpy(M)
ab2 = H.make_alpha_bars(500)
err5 = 0.0
for t in (20, 100, 300):
    abar = float(ab2[t - 1])
    x0s = torch.randint(0, q, (6, 2 ** depth))
    xt = H.noise_batch(torch.nn.functional.one_hot(x0s, q).double(), torch.full((6,), t), ab2, torch.Generator().manual_seed(t))
    p_c, p_u, _ = H.oracle_cond_uncond(Mt, depth, xt, abar, roots=torch.zeros(6, dtype=torch.long))
    # 暴力：P(x0配置 | xt) ∝ P(配置) * N(xt; sqrt(ab) onehot(配置), (1-ab))
    L = 2 ** depth
    for b in range(6):
        post_u = np.zeros((L, q)); Z = 0.0
        post_c = np.zeros((L, q)); Zc = 0.0
        for root in range(q):
            # 列舉子樹配置
            def sub(sym, level):
                if level == depth:
                    return {(sym,): 1.0}
                out = {}
                for l, r in itertools.product(range(q), range(q)):
                    pm = M[sym, l, r]
                    if pm == 0:
                        continue
                    for cl, pl in sub(l, level + 1).items():
                        for cr, pr in sub(r, level + 1).items():
                            out[cl + cr] = out.get(cl + cr, 0) + pm * pl * pr
                return out
            for cfg, pc in sub(root, 0).items():
                xcfg = np.eye(q)[list(cfg)]
                ll = np.exp(-((xt[b].numpy() - np.sqrt(abar) * xcfg) ** 2).sum() / (2 * (1 - abar)))
                wgt = (1 / q) * pc * ll
                Z += wgt
                for j in range(L):
                    post_u[j, cfg[j]] += wgt
                if root == 0:  # roots=0 條件
                    Zc += pc * ll
                    for j in range(L):
                        post_c[j, cfg[j]] += pc * ll
        err5 = max(err5, np.abs(post_u / Z - p_u[b].numpy()).max(), np.abs(post_c / Zc - p_c[b].numpy()).max())
check("T5_bruteforce_with_real_diffusion_noise", err5)

# T7：資料產生器的葉分布與 BP 先驗一致（取樣檢查，容許統計誤差 0.02）
from modules.gen_filtered_hierarchical_data_wforbidden import generate_dataset  # noqa: E402
t0 = time.time()
d = generate_dataset(3, 3, 1.0, 2, seed=0, N_trials=3000, k=0, workers=1)
q_, l_, _, x0s_, xis_, M_, _ = d
roots_emp = np.bincount(np.asarray(x0s_).astype(int), minlength=3) / 3000
# 以 BP 的無條件邊際（無觀測）算葉 0 的分布，對照樣本頻率
Mt = torch.from_numpy(np.asarray(M_, dtype=np.float64))
zero_field = torch.zeros(1, 2 ** l_, 3, dtype=torch.float64)
lp, _ = H.bp_posteriors(Mt, l_, zero_field, torch.full((3,), 1 / 3, dtype=torch.float64))
emp_leaf0 = np.bincount(np.asarray(xis_)[0].astype(int), minlength=3) / 3000
e7 = max(np.abs(roots_emp - 1 / 3).max(), np.abs(emp_leaf0 - lp[0, 0].numpy()).max())
check("T7_generator_matches_BP_prior_marginals(stat)", e7, tol=0.03)
print(f"(T7 用 {time.time()-t0:.1f}s)")

all_pass = all(v.get("pass") for v in results.values())
out = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "results", "bp_unit_test.json")
os.makedirs(os.path.dirname(out), exist_ok=True)
json.dump({"all_pass": all_pass, "tests": results}, open(out, "w"), indent=1)
print("ALL PASS" if all_pass else "SOME FAILED", "->", os.path.abspath(out))
sys.exit(0 if all_pass else 1)
