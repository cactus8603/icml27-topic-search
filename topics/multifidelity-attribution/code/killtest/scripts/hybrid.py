"""混合估計（離線線性代數）。記號：Y 為 n×k（n 訓練樣本、k 查詢）的 dloss_q/dw_i 矩陣。
一次重播可量測「查詢組合」Y u（u∈R^k）。選 B 個正交方向 U（k×B）後：
  Spell 式：Ŷ = Y U Uᵀ                          （補空間視為 0）
  混合式：  Ŷ = Y U Uᵀ + α C (I − U Uᵀ)，α = <YU, CU>/<CU, CU>（量測子空間上的最小平方，不收縮）
"""
import numpy as np


def select_U(G, B, mode="gram", seed=0):
    """gram：G(k×k) 的前 B 個特徵向量（等價於 Spell 以 G_V 做貪婪選取，簡化版重現）；random：高斯正交化。"""
    k = G.shape[0]
    if mode == "random":
        q, _ = np.linalg.qr(np.random.RandomState(seed).randn(k, B))
        return q
    w, v = np.linalg.eigh((G + G.T) / 2)
    return v[:, np.argsort(-w)[:B]]


def proj(U):
    return U @ U.T


def measure(Y, U):
    """模擬 B 次重播：每次得到 Y u_b（n 維）。"""
    return Y @ U


def spell(Y, U):
    return measure(Y, U) @ U.T


def fit_alpha(Y, C, U):
    YU, CU = measure(Y, U), measure(C, U)
    d = float((CU * CU).sum())
    return float((YU * CU).sum()) / d if d > 0 else 0.0


def hybrid(Y, C, U, alpha=None):
    """alpha=None 時由量測子空間估計；回傳 (Ŷ, α)。"""
    if alpha is None:
        alpha = fit_alpha(Y, C, U)
    P = proj(U)
    return measure(Y, U) @ U.T + alpha * (C - C @ P), alpha


def kappa(Y, U):
    return float(((Y @ U) ** 2).sum() / (Y ** 2).sum())


def kappa_res(Y, C, U, alpha):
    R = Y - alpha * C
    return kappa(R, U)


def reliability(Y, C, U):
    """逐查詢可靠度估計 ρ̂_q：只用量測到的部分（不碰 Y 的補空間）。
    量測子空間內，查詢 q 的 Y 投影與 C 投影的餘弦：cos((YΠ)_q, (CΠ)_q)。"""
    P = proj(U)
    a, b = (Y @ P), (C @ P)
    num = (a * b).sum(0)
    den = np.linalg.norm(a, axis=0) * np.linalg.norm(b, axis=0) + 1e-30
    return num / den
