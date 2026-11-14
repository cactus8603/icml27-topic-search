"""Teacher-forced NLL (all models, T grid) + per-position temperature-fit analysis (criterion 5)."""
import sys, os, re, time, json, argparse, numpy as np, torch
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from common import *
ap = argparse.ArgumentParser(); ap.add_argument("--setting", required=True); ap.add_argument("--limit", type=int, default=0)
ap.add_argument("--n-analysis", type=int, default=1000); ap.add_argument("--only", nargs="*", default=None); ap.add_argument("--out", default=None)
a = ap.parse_args()
torch.backends.cuda.matmul.allow_tf32 = False
from transformers import AutoModelForCausalLM, AutoTokenizer
kind = SETTINGS[a.setting]["kind"]; V = 151665
tok = AutoTokenizer.from_pretrained(SETTINGS[a.setting]["base"])
TS = [0.8, 0.9, 1.0, 1.1, 1.2]
if kind == "lm":
    d = np.load(f"{RUNS}/data/lm_test.npz"); X = np.concatenate([d["prompt"], d["cont"]], 1).astype(np.int64)
    if a.limit: X = X[:a.limit]
    seqs = [(x.tolist(), 64) for x in X]
else:
    D = json.load(open(f"{RUNS}/data/gsm_test.json")); seqs = []
    for x in D:
        pr = tok(tok.apply_chat_template([{"role": "user", "content": x["q"] + "\nPlease reason step by step, and put your final answer at the end."}], tokenize=False, add_generation_prompt=True), add_special_tokens=False).input_ids
        sol, fin = x["a"].split("####"); sol = re.sub(r"<<.*?>>", "", sol).strip() + f"\nThe final answer is {fin.strip()}"
        ids = tok(sol + "<|im_end|>", add_special_tokens=False).input_ids; seqs.append((pr + ids, len(pr)))
    if a.limit: seqs = seqs[:a.limit]
def load(p): return AutoModelForCausalLM.from_pretrained(p, torch_dtype=torch.float32).cuda().eval()
def batches(n, bs):
    order = sorted(range(n), key=lambda i: len(seqs[i][0]))
    for i in range(0, n, bs): yield order[i:i + bs]
def mk(idx):
    L = max(len(seqs[i][0]) for i in idx); ids = torch.full((len(idx), L), 151643); m = torch.zeros((len(idx), L), dtype=torch.bool); att = torch.zeros((len(idx), L), dtype=torch.long)
    for k, i in enumerate(idx):
        s, pl = seqs[i]; ids[k, :len(s)] = torch.tensor(s); att[k, :len(s)] = 1; m[k, pl:len(s)] = True  # target positions
    return ids.cuda(), att.cuda(), m.cuda()
@torch.no_grad()
def fwd(model, ids, att): return model(input_ids=ids).logits[:, :-1, :V].float()
def fit_beta(zb, pn_logp, iters=28):
    """beta s.t. softmax(beta*zb) closest (KL(pn||q)) to pn. zb:[N,V], pn_logp:[N,V]. Returns beta, residual KL."""
    pn = pn_logp.exp(); Ep = (pn * zb).sum(-1)
    lo = torch.full_like(Ep, 0.05); hi = torch.full_like(Ep, 20.0)
    for _ in range(iters):
        mid = (lo + hi) / 2; q = torch.softmax(mid[:, None] * zb, -1); f = (q * zb).sum(-1) - Ep
        pos = f > 0; hi = torch.where(pos, mid, hi); lo = torch.where(pos, lo, mid)
    beta = (lo + hi) / 2
    lq = torch.log_softmax(beta[:, None] * zb, -1); kl = (pn * (pn_logp - lq)).sum(-1)
    return beta, kl
models = model_list(a.setting); base_dir = models[0][1]
out = a.out or f"{RES}/nll_analysis_{a.setting}.json"
res = json.load(open(out)) if os.path.exists(out) else {}
base = load(base_dir)
def nll_pass(model, name, with_an):
    nll = {T: 0.0 for T in TS}; cnt = 0; an = {"beta": [], "kl_fit": [], "kl_raw": [], "kl_T1": []}; seen = 0
    for idx in batches(len(seqs), 32 if kind == "lm" else 16):
        ids, att, m = mk(idx); z = fwd(model, ids, att); tgt = ids[:, 1:]; mm = m[:, 1:]
        for T in TS:
            lp = torch.log_softmax(z / T, -1); nll[T] += -(lp.gather(-1, tgt.clamp(max=V - 1)[..., None])[..., 0] * mm).sum().item()
        cnt += mm.sum().item()
        if with_an and seen < a.n_analysis:
            zb = fwd(base, ids, att)[mm]; zn = z[mm]
            if name != "base":
                for c in range(0, zb.size(0), 512):
                    zbc, znc = zb[c:c + 512], zn[c:c + 512]; lpn = torch.log_softmax(znc, -1)
                    beta, kl = fit_beta(zbc, lpn); pn = lpn.exp()
                    kl_raw = (pn * (lpn - torch.log_softmax(zbc, -1))).sum(-1)
                    an["beta"].append(beta.cpu()); an["kl_fit"].append(kl.cpu()); an["kl_raw"].append(kl_raw.cpu())
            seen += len(idx)
    r = {"nll": {str(T): nll[T] / cnt for T in TS}, "n_tokens": cnt}
    if an["beta"]:
        beta = torch.cat(an["beta"]).numpy(); T_i = 1 / beta; kf = torch.cat(an["kl_fit"]).numpy(); kr = torch.cat(an["kl_raw"]).numpy()
        q = lambda x: {k: float(np.quantile(x, v)) for k, v in zip(["q05", "q25", "q50", "q75", "q95"], [.05, .25, .5, .75, .95])}
        r["analysis"] = {"n_pos": int(len(T_i)), "T_mean": float(T_i.mean()), "T_std": float(T_i.std()), "T_q": q(T_i), "beta_mean": float(beta.mean()),
                         "KL_raw_mean": float(kr.mean()), "KL_fit_mean": float(kf.mean()), "KL_fit_frac_of_raw": float(kf.mean() / max(kr.mean(), 1e-12)),
                         "T_at_bound_frac": float(((T_i > 19.9) | (T_i < 0.0501) | (beta < 0.0501) | (beta > 19.9)).mean())}
    return r
t0 = time.time()
for name, p in models:
    if a.only and name not in a.only: continue
    if name in res and not a.only: continue
    model = base if name == "base" else load(p)
    res[name] = nll_pass(model, name, with_an=True); res[name]["secs"] = time.time() - t0
    print(name, res[name]["nll"], res[name].get("analysis"), flush=True)
    if name != "base": del model; torch.cuda.empty_cache()
    jdump(res, out)
print("total_s", time.time() - t0)
