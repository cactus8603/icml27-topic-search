"""階層資料（原作 hierarchical data model）的「類別條件版」共用函式庫。

內容：
  1. 資料讀寫（沿用原作 gen_filtered_hierarchical_data_wforbidden.generate_dataset 產生的格式）
  2. 精確的條件／無條件 belief propagation（BP）：條件 = 根符號 x0（K=q 類，先驗均勻）
  3. 連續擴散（one-hot 加高斯噪聲，與原作 noise_schedules 一致）與 oracle 分數
  4. 帶類別條件與 condition dropout 的 transformer 去噪器（結構同原作 + 類別嵌入）

約定（全部以 x0-prediction 空間表示）：
  - 去噪器輸出 x0 各 token 的後驗分布 p(x0_leaf | x_t[, c])（softmax 後）。
  - 分數與 x0-hat 是線性關係：score = (sqrt(abar) * x0hat - x_t) / (1 - abar)，
    因此 CFG 的分數外插等價於 x0hat 的外插：
        p_cfg(w) = (1 + w) * p_cond - w * p_uncond          （w=0 即條件模型；w>0 外插，可能出現負值）
  - 場（field）= sqrt(abar_t) / (1 - abar_t) * x_t，與原作 backward_process_gt 一致（abar_t = alpha_bars[t-1]）。
"""
import math
import os
import sys

import numpy as np
import torch
import torch.nn as nn

BG_EXT = os.environ.get("BG_EXT", "/work/ChihChungHsu/cactus8603/icml/ext/biased-generalization")
if BG_EXT not in sys.path:
    sys.path.insert(0, BG_EXT)

from modules.embeddings import timestep_embedding  # noqa: E402  原作


# ---------------------------------------------------------------------------
# 1. 資料
# ---------------------------------------------------------------------------
def load_hier(path):
    """回傳 (q, depth, roots[N] int64, leaves[N, 2^depth] int64, M[q,q,q] float64)。"""
    q, depth_l, _sigma, x0s, xis, M, _restr = np.load(path, allow_pickle=True)
    leaves = torch.from_numpy(np.asarray(xis).astype(np.int64).T.copy())
    roots = torch.from_numpy(np.asarray(x0s).astype(np.int64))
    return int(q), int(depth_l), roots, leaves, np.asarray(M, dtype=np.float64)


# ---------------------------------------------------------------------------
# 2. 精確 BP（條件／無條件）
# ---------------------------------------------------------------------------
def _norm(x):
    return x / x.sum(-1, keepdim=True).clamp_min(1e-300)


def bp_posteriors(M, depth, field, root_prior):
    """二元樹上的精確 BP（機率空間、每步正規化）。

    M: (q,q,q) 張量，M[p,l,r] = P(左子=l, 右子=r | 父=p)
    field: (B, 2^depth, q)，葉的對數似然（未正規化）
    root_prior: (q,) 或 (B, q)；均勻 = 無條件，one-hot(c) = 以根符號 c 為條件
    回傳 (leaf_post (B,2^depth,q), root_post (B,q))
    """
    B, L, q = field.shape
    assert L == 2 ** depth
    M = M.to(field.dtype)
    lik = torch.softmax(field, dim=-1)  # 葉的「向上訊息」
    ups = [None] * (depth + 1)
    ups[depth] = lik
    for i in range(depth - 1, -1, -1):
        ch = ups[i + 1]
        m = torch.einsum("plr,bnl,bnr->bnp", M, ch[:, 0::2], ch[:, 1::2])
        ups[i] = _norm(m)
    rp = root_prior if root_prior.dim() == 2 else root_prior.unsqueeze(0).expand(B, q)
    ctx = [None] * (depth + 1)
    ctx[0] = rp.to(field.dtype).unsqueeze(1)
    for i in range(depth):
        ch = ups[i + 1]
        Lc, Rc = ch[:, 0::2], ch[:, 1::2]
        pi = ctx[i]
        cl = _norm(torch.einsum("bnp,plr,bnr->bnl", pi, M, Rc))
        cr = _norm(torch.einsum("bnp,plr,bnl->bnr", pi, M, Lc))
        ctx[i + 1] = torch.stack([cl, cr], dim=2).reshape(B, 2 ** (i + 1), q)
    leaf_post = _norm(ups[depth] * ctx[depth])
    root_post = _norm(ups[0][:, 0] * ctx[0][:, 0])
    return leaf_post, root_post


def oracle_cond_uncond(M, depth, xt, abar, roots=None):
    """xt: (B,L,q)。回傳 (p_cond (B,L,q), p_uncond (B,L,q), root_post (B,q))。
    p_cond 以 roots（(B,) 整數）為條件；roots=None 時只回傳無條件。"""
    q = xt.shape[-1]
    field = xt * (math.sqrt(abar) / (1.0 - abar))
    unif = torch.full((q,), 1.0 / q, dtype=field.dtype, device=field.device)
    p_u, root_post = bp_posteriors(M, depth, field, unif)
    p_c = None
    if roots is not None:
        oh = torch.nn.functional.one_hot(roots, q).to(field.dtype)
        p_c, _ = bp_posteriors(M, depth, field, oh)
    return p_c, p_u, root_post


def cfg_combine(p_c, p_u, w):
    """x0hat 空間的 CFG：(1+w) p_c - w p_u。w=0 為條件模型。"""
    return (1.0 + w) * p_c - w * p_u


# ---------------------------------------------------------------------------
# 3. 擴散（與原作 modules.noise_schedules 一致：線性 beta 乘 1000/T，T=500）
# ---------------------------------------------------------------------------
def make_alpha_bars(t_final=500, device="cpu", dtype=torch.float64):
    scale = 1000 / t_final
    beta = torch.linspace(scale * 1e-4, scale * 0.02, t_final, dtype=torch.float64)
    return torch.cumprod(1 - beta, dim=0).to(device=device, dtype=dtype)


def noise_batch(x0_onehot, t, alpha_bars, gen=None):
    """t: (B,) 整數 in 1..T；abar = alpha_bars[t-1]。"""
    ab = alpha_bars[t - 1].to(x0_onehot.dtype).view(-1, 1, 1)
    eps = torch.randn(x0_onehot.shape, device=x0_onehot.device, dtype=x0_onehot.dtype, generator=gen)
    return ab.sqrt() * x0_onehot + (1 - ab).sqrt() * eps


# ---------------------------------------------------------------------------
# 4. 模型：原作 TransformerForDiffusion + 類別嵌入（index q = 空條件）
# ---------------------------------------------------------------------------
class CondDenoiser(nn.Module):
    def __init__(self, seq_len, vocab, hidden=512, nhead=4, layers=8, model_channels=32, n_classes=None):
        super().__init__()
        self.seq_len, self.vocab, self.model_channels = seq_len, vocab, model_channels
        self.n_classes = vocab if n_classes is None else n_classes
        self.inp = nn.Linear(vocab, hidden)
        self.pos = nn.Embedding(seq_len, hidden)
        self.tim = nn.Linear(model_channels, hidden)
        self.cls = nn.Embedding(self.n_classes + 1, hidden)  # 最後一個索引 = null
        layer = nn.TransformerEncoderLayer(d_model=hidden, nhead=nhead, dim_feedforward=2 * hidden, batch_first=True)
        self.enc = nn.TransformerEncoder(layer, num_layers=layers, enable_nested_tensor=False)
        self.out = nn.Linear(hidden, vocab)

    def forward(self, x, t, c):
        """x:(B,L,q) t:(B,) c:(B,) 整數（null = n_classes）。回傳 logits (B,L,q)。"""
        B = x.shape[0]
        pos = self.pos(torch.arange(self.seq_len, device=x.device)).unsqueeze(0)
        h = self.inp(x) + pos + (self.tim(timestep_embedding(t, self.model_channels)) + self.cls(c)).unsqueeze(1)
        return self.out(self.enc(h))


def model_kwargs(args):
    return dict(hidden=args.hidden, nhead=args.nhead, layers=args.layers)
