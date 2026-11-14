"""共用：設定、條件字串、指標（TPR@FPR、AUROC、cluster bootstrap）、探測器訓練。
只做監測與偵測，不含任何逃避訓練。"""
import json, os
import numpy as np

EXP = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


def load_config(path=None):
    with open(path or os.path.join(EXP, "configs", "config.json")) as f:
        return json.load(f)


def render_cond(cond, prompt, templates):
    """cond: 'clean' | 'T0:word' ...；回傳加了觸發/誘餌文字的 prompt。"""
    if cond == "clean":
        return prompt
    t, w = cond.split(":", 1)
    return templates[t].format(w=w, p=prompt)


# ------------------------------ 指標 ------------------------------
def auroc(pos, neg):
    """Mann-Whitney AUROC（處理並列）。"""
    pos = np.asarray(pos, float); neg = np.asarray(neg, float)
    if len(pos) == 0 or len(neg) == 0:
        return float("nan")
    allv = np.concatenate([pos, neg])
    order = allv.argsort(kind="mergesort")
    ranks = np.empty(len(allv)); sv = allv[order]
    i = 0
    while i < len(sv):
        j = i
        while j + 1 < len(sv) and sv[j + 1] == sv[i]:
            j += 1
        ranks[order[i:j + 1]] = (i + j) / 2 + 1
        i = j + 1
    return float((ranks[:len(pos)].sum() - len(pos) * (len(pos) + 1) / 2) / (len(pos) * len(neg)))


def thr_at_fpr(neg, fpr):
    """門檻：負例分數的 (1-fpr) 分位數（'higher'），判陽性條件為 score > thr，故負例 FPR <= fpr。"""
    return np.quantile(np.asarray(neg, float), 1 - fpr, method="higher")


def tpr_at_fpr(pos, neg, fpr=0.01):
    thr = thr_at_fpr(neg, fpr)
    return float((np.asarray(pos, float) > thr).mean())


def cluster_bootstrap_delta(sc_a, sc_b, neg_a, neg_b, B, seed, fpr=0.01, ci=0.95):
    """配對 cluster bootstrap。
    sc_*: [S, n_pos, C] 每 seed 的正例分數（C 個觸發條件，同一正例的各條件為同一 cluster）。
    neg_*: [S, n_neg]。回傳 (point, lo, hi, boot_array)。Δ = mean_seed[TPR_a - TPR_b]，TPR 對 C 個條件平均。"""
    S, n_pos, C = sc_a.shape
    n_neg = neg_a.shape[1]
    rng = np.random.default_rng(seed)

    def stat(ip, ineg):
        d = []
        for s in range(S):
            ta = np.quantile(neg_a[s, ineg], 1 - fpr, method="higher")
            tb = np.quantile(neg_b[s, ineg], 1 - fpr, method="higher")
            d.append((sc_a[s, ip] > ta).mean() - (sc_b[s, ip] > tb).mean())
        return float(np.mean(d))

    point = stat(np.arange(n_pos), np.arange(n_neg))
    boots = np.empty(B)
    for b in range(B):
        boots[b] = stat(rng.integers(0, n_pos, n_pos), rng.integers(0, n_neg, n_neg))
    a = (1 - ci) / 2
    return point, float(np.quantile(boots, a)), float(np.quantile(boots, 1 - a)), boots


def bootstrap_ci(values_fn, n, B, seed, ci=0.95):
    rng = np.random.default_rng(seed)
    bs = np.array([values_fn(rng.integers(0, n, n)) for _ in range(B)])
    a = (1 - ci) / 2
    return float(np.quantile(bs, a)), float(np.quantile(bs, 1 - a))


# ------------------------------ 探測器 ------------------------------
class Scaler:
    def fit(self, X):
        self.mu = X.mean(0, keepdims=True); self.sd = X.std(0, keepdims=True) + 1e-6
        return self

    def __call__(self, X):
        return (X - self.mu) / self.sd


class TorchLogistic:
    """L2 邏輯迴歸（目標 = C*sum logloss + 0.5*||w||²，與 sklearn 的 C 同義），torch LBFGS。
    不依賴 sklearn（環境沒有）；有 GPU 時用 GPU。"""
    def __init__(self, w, b, dev):
        self.w, self.b, self.dev = w, b, dev

    def prob(self, X):
        import torch
        with torch.no_grad():
            return torch.sigmoid(torch.as_tensor(X, dtype=torch.float32, device=self.dev) @ self.w + self.b).cpu().numpy()


def fit_logistic(X, y, C, max_iter, seed=0):
    import torch
    dev = "cuda" if torch.cuda.is_available() else "cpu"
    Xt = torch.as_tensor(X, dtype=torch.float32, device=dev); yt = torch.as_tensor(y, dtype=torch.float32, device=dev)
    w = torch.zeros(Xt.shape[1], device=dev, requires_grad=True); b = torch.zeros(1, device=dev, requires_grad=True)
    opt = torch.optim.LBFGS([w, b], lr=1.0, max_iter=max_iter, history_size=20, line_search_fn="strong_wolfe", tolerance_grad=1e-5, tolerance_change=1e-9)

    def closure():
        opt.zero_grad()
        loss = C * torch.nn.functional.binary_cross_entropy_with_logits(Xt @ w + b, yt, reduction="sum") + 0.5 * (w * w).sum()
        loss.backward(); return loss
    opt.step(closure)
    return TorchLogistic(w.detach(), b.detach(), dev)


def prob(clf, X):
    return clf.prob(X)


def estimate_subspace(deltas, energy, k_cap):
    """deltas: [n, D]（已標準化空間）。回傳 U [D, k]，k = 達到 energy 比例的最小維度（上限 k_cap），以及奇異值能量比。"""
    d = deltas - 0.0  # 不置中：我們要的是逃避位移本身的方向
    _, s, vt = np.linalg.svd(d, full_matrices=False)
    e = (s ** 2) / (s ** 2).sum()
    k = int(min(np.searchsorted(np.cumsum(e), energy) + 1, k_cap, len(s)))
    return vt[:k].T.copy(), e


def sub_idx(n, frac, seed, y=None):
    rng = np.random.default_rng(seed)
    if y is None:
        return np.sort(rng.choice(n, int(n * frac), replace=False))
    idx = []
    for c in np.unique(y):
        ii = np.where(y == c)[0]
        idx.append(rng.choice(ii, int(len(ii) * frac), replace=False))
    return np.sort(np.concatenate(idx))
