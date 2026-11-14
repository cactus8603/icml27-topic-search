import math, sys, os, time, json, numpy as np, torch, argparse, glob
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from common import *
ap = argparse.ArgumentParser()
ap.add_argument("--setting", required=True); ap.add_argument("--steps", type=int, default=400)
ap.add_argument("--save", type=int, nargs="+", default=STEPS); ap.add_argument("--limit", type=int, default=0)
ap.add_argument("--tag", default=""); ap.add_argument("--bs", type=int, default=32); ap.add_argument("--micro", type=int, default=16)
a = ap.parse_args()
from transformers import AutoModelForCausalLM
base = SETTINGS[a.setting]["base"]; kind = SETTINGS[a.setting]["kind"]
torch.manual_seed(0); np.random.seed(0)
# data: list of (ids, loss_mask)
if kind == "lm":
    P = np.load(f"{RUNS}/data/lm_selftrain.npz")["prompt"]
    G = np.load(f"{RUNS}/{a.setting}/gen_self/selftrain/T0.8_p0.9_m0.0_s0.npy")
    n = a.limit or len(P)
    seqs = [(np.concatenate([P[i], G[i]]), 64) for i in range(n)]  # loss only on generated part
else:
    R = json.load(open(f"{RUNS}/{a.setting}/gen_self/selftrain/default_s0.json"))
    R = [r for r in R if r["fin"] == "stop"]
    print("selftrain examples (finished):", len(R))
    eos = 151645  # <|im_end|>
    seqs = [(np.array(r["p"] + r["c"] + ([] if r["c"] and r["c"][-1] == eos else [eos])), len(r["p"])) for r in R]
    if a.limit: seqs = seqs[:a.limit]
model = AutoModelForCausalLM.from_pretrained(base, torch_dtype=torch.float32).cuda()
model.gradient_checkpointing_enable(); model.config.use_cache = False; model.train()
opt = torch.optim.AdamW(model.parameters(), lr=1e-5, betas=(0.9, 0.999), weight_decay=0.0)
WARM = 10
sched = torch.optim.lr_scheduler.LambdaLR(opt, lambda s: min(1.0, (s + 1) / WARM))
rng = np.random.RandomState(0); order = []
def batch_iter():
    while True:
        p = rng.permutation(len(seqs))
        for i in range(0, len(p) - a.bs + 1, a.bs): yield p[i:i + a.bs]
it = batch_iter(); t0 = time.time(); log = []
outdir = f"{RUNS}/{a.setting}/theta_s{a.tag}"; os.makedirs(outdir, exist_ok=True)
for step in range(1, a.steps + 1):
    idx = next(it); tot_loss = 0.0
    ntok = sum(len(seqs[i][0]) - seqs[i][1] for i in idx)
    for m in range(0, a.bs, a.micro):
        sub = idx[m:m + a.micro]; L = max(len(seqs[i][0]) for i in sub)
        ids = torch.full((len(sub), L), 151643); lab = torch.full((len(sub), L), -100); att = torch.zeros((len(sub), L), dtype=torch.long)
        for k, i in enumerate(sub):
            s, pl = seqs[i]; ids[k, :len(s)] = torch.tensor(s); att[k, :len(s)] = 1; lab[k, pl:len(s)] = torch.tensor(s[pl:])
        ids, lab, att = ids.cuda(), lab.cuda(), att.cuda()
        with torch.autocast("cuda", dtype=torch.bfloat16):
            logits = model(input_ids=ids).logits  # right padding + causal: no mask needed (padded SDPA gave NaN grads)
        loss = torch.nn.functional.cross_entropy(logits[:, :-1].float().reshape(-1, logits.size(-1)), lab[:, 1:].reshape(-1), ignore_index=-100, reduction="sum") / ntok
        if not torch.isfinite(loss):
            print("NONFINITE micro", step, m, [(int(i), len(seqs[i][0]), seqs[i][1]) for i in sub], "logits finite:", torch.isfinite(logits).all().item(), "maxabs", logits.abs().max().item(), flush=True); raise SystemExit(3)
        loss.backward(); tot_loss += loss.item()
    gn = torch.nn.utils.clip_grad_norm_(model.parameters(), 1e9).item()  # measure only (no clipping)
    if not math.isfinite(gn):
        print("NONFINITE GRAD at step", step, "batch", [(int(i), len(seqs[i][0]), seqs[i][1]) for i in idx], "loss", tot_loss, flush=True); raise SystemExit(4)
    opt.step(); sched.step(); opt.zero_grad(set_to_none=True)
    log.append(dict(step=step, loss=tot_loss, gnorm=gn))
    if step % 10 == 0 or step == 1: print(f"step {step} loss {tot_loss:.4f} gnorm {gn:.3f} t={time.time()-t0:.0f}s", flush=True)
    if step in a.save:
        model.save_pretrained(f"{outdir}/step{step}", safe_serialization=True); print("saved", step, flush=True)
jdump(log, f"{RES}/train_log_{a.setting}{a.tag}.json")
print("train_s", time.time() - t0)
