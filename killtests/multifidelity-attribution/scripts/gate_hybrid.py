"""閘門 G2：混合碼三個恆等式（純 numpy，CPU 即可）。輸出 results/gates/hybrid_identities.json。"""
import json, os, sys
import numpy as np
sys.path.insert(0, os.path.dirname(__file__))
import hybrid as H

rs = np.random.RandomState(0)
n, k, B = 300, 20, 4
# 低秩+雜訊的 Y 與有相關但不同的 C
Y = rs.randn(n, 6) @ rs.randn(6, k) + 0.1 * rs.randn(n, k)
C = 0.6 * Y + rs.randn(n, k)
G = rs.randn(k, 40); G = G @ G.T
res = {}
U = H.select_U(G, B)
res["U_orthonormal"] = bool(np.allclose(U.T @ U, np.eye(B), atol=1e-8))
res["proj_idempotent"] = bool(np.allclose(H.proj(U) @ H.proj(U), H.proj(U), atol=1e-8))
# (1) α=0 等於 Spell（Spell 另以「量測再提升」獨立實作）
yh0, a0 = H.hybrid(Y, C, U, alpha=0.0)
res["alpha0_equals_spell"] = bool(np.allclose(yh0, H.spell(Y, U), atol=1e-10) and np.allclose(H.spell(Y, U), Y @ (U @ U.T)))
# (2) B 取滿等於 MAGIC（任意 C、任意 α）
Uf = H.select_U(G, k)
for al in (None, 0.0, 3.0):
    yh, _ = H.hybrid(Y, C, Uf, alpha=al)
    res[f"Bfull_equals_Y_alpha_{al}"] = bool(np.allclose(yh, Y, atol=1e-8))
# (3) C=Y 時輸出 Y，且 α̂=1
yh, al = H.hybrid(Y, Y, U)
res["C_eq_Y_outputs_Y"] = bool(np.allclose(yh, Y, atol=1e-8) and abs(al - 1) < 1e-10)
# 其他合理性：κ 在 [0,1]、殘差能量比例、混合誤差單調（α̂ 是量測子空間最小平方 → 該子空間殘差不大於 α=0 或 α=1）
res["kappa_in_01"] = 0 <= H.kappa(Y, U) <= 1
YU = Y @ U
e = lambda a: float(((YU - a * (C @ U)) ** 2).sum())
res["alpha_hat_is_LS"] = bool(e(H.fit_alpha(Y, C, U)) <= min(e(0.0), e(1.0)) + 1e-9)
res["pass"] = all(v for v in res.values() if isinstance(v, bool))
out = os.path.join(os.path.dirname(__file__), "..", "results", "gates")
os.makedirs(out, exist_ok=True)
json.dump(res, open(os.path.join(out, "hybrid_identities.json"), "w"), indent=1)
print(json.dumps(res, indent=1)); sys.exit(0 if res["pass"] else 1)
