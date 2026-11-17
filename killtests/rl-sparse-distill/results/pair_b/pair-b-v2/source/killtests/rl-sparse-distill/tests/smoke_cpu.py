"""CPU 極小 smoke test（程式正確性閘門的 CPU 版；GPU 版用 slurm/00_correctness_gpu.sbatch 在真實模型上重跑）。
用法：python -I tests/smoke_cpu.py BASE_DIR RL_DIR   （RL 可為 base 加雜訊的副本，只為製造不一致位置）
檢查：
 T1 rl_replace(rl=base) == base 取樣（相同 seed，--no-compact）
 T2 all_replace == rl greedy
 T3 與原作 wrb RLReplaceDecoder 逐 token 相同（單題、無 padding、--no-compact）
 T4 多題左 padding + 壓縮 的 greedy == 各題單獨 greedy
 T5 replace 位置數合理（0 < 比例 < 1）並印出
 T6 （2026-11-14，第二探針 geo_mix）兩個模型相同時 geo_mix 輸出 == base 取樣（同 seed，逐字相同，替換數 0）
 T7 geo_mix 在 λ=1、temperature 0 時 == RL greedy；λ=0 取樣 == base 取樣（兩模型不同）
 T8 geo_mix 在 λ=1 時，不一致位置從 RL 的分布取樣：與獨立寫的逐步參考迴圈（不一致位置的 logits 直接取 RL，其餘取 base，
    同一個 generator）逐字相同
"""
import os, sys
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "scripts"))
from common import *
sys.path.insert(0, WRB)
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
import gen_replace as G
from wrb.decoding.rl_replace import RLReplaceDecoder, ReplaceConfig

base_dir, rl_dir = sys.argv[1], sys.argv[2]
DEV = sys.argv[3] if len(sys.argv) > 3 else "cpu"
MAXNEW = int(sys.argv[4]) if len(sys.argv) > 4 else 20
torch.set_num_threads(8)
tok = AutoTokenizer.from_pretrained(base_dir, padding_side="left")
if tok.pad_token is None:
    tok.pad_token = "<|endoftext|>"
mb = AutoModelForCausalLM.from_pretrained(base_dir, torch_dtype=torch.float32).to(DEV).eval()
mr = AutoModelForCausalLM.from_pretrained(rl_dir, torch_dtype=torch.float32).to(DEV).eval()
qs = ["What is 2+3? Explain briefly.", "Compute the sum of the first ten positive integers, and also tell me something interesting about it.", "Solve x^2=9."]
P = [build_prompt(q) for q in qs]
N = 4
ok = True


def gen(seed):
    g = torch.Generator(device=DEV); g.manual_seed(seed); return g


def run(cond, prompts, m_b, m_r, temp, compact, seed=7):
    return G.decode_batch(tok, m_b, m_r, prompts, N, cond, temp, 0.95, MAXNEW, 4096, gen(seed), compact=compact)


def check(name, cond):
    global ok
    print(("PASS " if cond else "FAIL ") + name)
    ok &= bool(cond)


r1 = run("rl_replace", P[:1], mb, mb, 0.6, False)
r0 = run("base", P[:1], mb, None, 0.6, False)
check("T1 rl=base -> base sampling", [x[0] for x in r1] == [x[0] for x in r0] and all(x[3] == 0 for x in r1))

ra = run("all_replace", P[:1], mb, mr, 0.6, False)
rg = run("rl", P[:1], None, mr, 0.0, False)
check("T2 all_replace == rl greedy", [x[0] for x in ra] == [x[0] for x in rg])

ref = RLReplaceDecoder(mb, mr, tok, ReplaceConfig(max_new_tokens=MAXNEW, temperature=0.6, top_p=0.95, seed=7))
ro = ref.generate(P[0], n_samples=N)
mine = run("rl_replace", P[:1], mb, mr, 0.6, False, seed=7)
same = all(o.text == m[0] and len(o.replacements) == m[3] for o, m in zip(ro, mine))
check("T3 identical to wrb RLReplaceDecoder (texts and replacement counts)", same)
print("   replacements per sample (wrb):", [len(o.replacements) for o in ro], "lengths", [o.length for o in ro])

multi = run("base", P, mb, None, 0.0, True)
single = [run("base", [p], mb, None, 0.0, True)[0] for p in P]
check("T4 padded+compacted batch greedy == single greedy", all(multi[i * N][0] == single[i][0] for i in range(3)))

rr = run("rl_replace", P, mb, mr, 0.6, True)
tot = sum(x[2] for x in rr); rep = sum(x[3] for x in rr)
check("T5 replace rate in (0,1): %.3f (%d/%d)" % (rep / tot, rep, tot), 0 < rep < tot)
# ---- T6–T8：geo_mix（第二探針）----
g1 = run("geo_mix", P[:1], mb, mb, 0.6, False)       # lam 預設 0.5；兩模型相同
check("T6 geo_mix(rl=base) == base sampling, zero disagreements", [x[0] for x in g1] == [x[0] for x in r0] and all(x[3] == 0 for x in g1))

def runlam(cond, prompts, temp, lam, seed=7, compact=False):
    return G.decode_batch(tok, mb, mr, prompts, N, cond, temp, 0.95, MAXNEW, 4096, gen(seed), compact=compact, lam=lam)

g_greedy = runlam("geo_mix", P[:1], 0.0, 1.0)
check("T7a geo_mix(lam=1, temperature 0) == RL greedy", [x[0] for x in g_greedy] == [x[0] for x in rg])
g_l0 = runlam("geo_mix", P[:1], 0.6, 0.0)
check("T7b geo_mix(lam=0) == base sampling", [x[0] for x in g_l0] == [x[0] for x in r0])

# T8 參考迴圈：單題、無 padding、不壓縮；逐步取 where(不一致, RL logits, base logits)，同 generator 取樣
@torch.no_grad()
def ref_lam1(prompt, temp, seed):
    enc = tok([prompt] * N, return_tensors="pt", padding=True, add_special_tokens=False)
    ids, attn = enc["input_ids"].to(DEV), enc["attention_mask"].to(DEV)
    pos = (attn.cumsum(-1) - 1).clamp(min=0)
    cb_, cr_ = DynamicCache(), DynamicCache()
    g = gen(seed); cur, ca, cp = ids, attn, pos
    toks = [[] for _ in range(N)]; alive = [True] * N; nd = 0
    stop = set(STOP_IDS)
    for t in range(MAXNEW):
        lb, cb_ = G.fwd(mb, cur, ca, cp, cb_); lr, cr_ = G.fwd(mr, cur, ca, cp, cr_)
        dis = G.greedy_from_logits(lb) != G.greedy_from_logits(lr)
        nxt = G.sample_from_logits(torch.where(dis.unsqueeze(-1), lr, lb), temp, 0.95, generator=g)
        for i in range(N):
            if alive[i]:
                toks[i].append(int(nxt[i])); nd += int(dis[i])
                if int(nxt[i]) in stop:
                    alive[i] = False
        if not any(alive):
            break
        cur = nxt.unsqueeze(-1); ca = torch.cat([ca, torch.ones_like(ca[:, :1])], -1); cp = cp[:, -1:] + 1
    out = []
    for tk in toks:
        body = tk[:-1] if tk and tk[-1] in stop and not alive[len(out)] else tk
        out.append(tok.decode(body, skip_special_tokens=True))
    return out, nd

from transformers import DynamicCache
ref_txt, ref_nd = ref_lam1(P[0], 0.6, 7)
g_l1 = runlam("geo_mix", P[:1], 0.6, 1.0)
check("T8 geo_mix(lam=1) == independent reference loop (RL logits at disagreements); n_repl %d == %d" % (sum(x[3] for x in g_l1), ref_nd),
      [x[0] for x in g_l1] == ref_txt and sum(x[3] for x in g_l1) == ref_nd)
print("ALL PASS" if ok else "SOME FAIL")
sys.exit(0 if ok else 1)
