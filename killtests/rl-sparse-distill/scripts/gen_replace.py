"""HF 雙模型逐步解碼（rl-replace，照 2610.24663 公開程式 wrb.decoding.rl_replace 的規則）。

規則（逐字）：b_t=argmax p_B, r_t=argmax p_R；b_t != r_t 則輸出 r_t，否則從 p_B 以 temperature 0.6、top-p 0.95 取樣。
取樣函式直接 import 原作 wrb.decoding.sampling（確保一致）。與原作迴圈的差異（皆為工程性）：
  1. 多題合批：左側 padding + 明確 position_ids（原作只批同一題，無 padding）。
  2. 已結束的列會從批次中移除（KV cache 以 batch_select_indices 壓縮），不再浪費算力；
     這會改變 RNG 消耗順序，所以與原作「逐 token 相同」只在 --no-compact 且單題批次時成立（tests/ 內驗證）。
conds：
  base         只用 base 取樣（HF 版對照，閘門用）
  rl           只用 rl 取樣（temperature 0 即 greedy）
  rl_replace   混合解碼（主角）
  all_replace  每個位置都強制輸出 r_t（閘門：必須等於 rl greedy）
  geo_mix      2026-11-14（第二探針 R3，見 PREREG）：不一致位置（argmax 不同）改從幾何混合
               softmax((1-lam)·logit_base + lam·logit_RL)（同 temperature、top-p）取樣，其餘位置照 base 取樣。
               lam=0 → 與 base 取樣相同；lam=1 → 不一致位置從 RL 的分布取樣（rl_replace 則是確定性輸出 RL 的 argmax）。
               兩個模型相同時沒有不一致位置，輸出逐位元等於 base 取樣（同 seed）。n_repl 記錄不一致位置數。
輸出：raw_{shard}.jsonl.gz，每題含 texts、finished、ntok、n_repl（被替換的位置數）。
"""
import argparse, json, os, sys, time
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from common import *

sys.path.insert(0, WRB)  # 原作程式（只 import 取樣函式）
import torch
from wrb.decoding.sampling import sample_from_logits, greedy_from_logits  # noqa: E402
from transformers import AutoModelForCausalLM, AutoTokenizer, DynamicCache  # noqa: E402


def fwd(model, ids, attn, pos, cache):
    out = model(input_ids=ids, attention_mask=attn, position_ids=pos, past_key_values=cache,
                use_cache=True, logits_to_keep=1)
    return out.logits[:, -1, :], out.past_key_values


@torch.no_grad()
def decode_batch(tok, mb, mr, prompts, n, cond, temperature, top_p, max_new, max_len, gen, compact=True, log=None, lam=0.5):
    dev = next((mb or mr).parameters()).device
    texts = [p for p in prompts for _ in range(n)]
    enc = tok(texts, return_tensors="pt", padding=True, add_special_tokens=False)
    ids, attn = enc["input_ids"].to(dev), enc["attention_mask"].to(dev)
    B, L = ids.shape
    max_new = min(max_new, max_len - L)
    pos = (attn.cumsum(-1) - 1).clamp(min=0)
    need_b = cond in ("base", "rl_replace", "all_replace", "geo_mix")
    need_r = cond in ("rl", "rl_replace", "all_replace", "geo_mix")
    cb = DynamicCache() if need_b else None
    cr = DynamicCache() if need_r else None
    stop = torch.tensor(STOP_IDS, device=dev)
    out_tok = torch.full((B, max_new), -1, dtype=torch.long, device=dev)
    n_repl = torch.zeros(B, dtype=torch.long, device=dev)
    finished = torch.zeros(B, dtype=torch.bool, device=dev)
    rows = torch.arange(B, device=dev)  # 目前批次各列對應的原始列
    cur_ids, cur_attn, cur_pos = ids, attn, pos
    steps_rows = 0
    act = torch.ones(B, dtype=torch.bool, device=dev)  # 目前批次中尚未結束的列
    for t in range(max_new):
        lb = lr = None
        if need_b:
            lb, cb = fwd(mb, cur_ids, cur_attn, cur_pos, cb)
        if need_r:
            lr, cr = fwd(mr, cur_ids, cur_attn, cur_pos, cr)
        if cond == "base":
            nxt = sample_from_logits(lb, temperature, top_p, generator=gen)
            dis = torch.zeros_like(nxt, dtype=torch.bool)
        elif cond == "rl":
            nxt = sample_from_logits(lr, temperature, top_p, generator=gen)
            dis = torch.zeros_like(nxt, dtype=torch.bool)
        elif cond == "geo_mix":
            btk, rtk = greedy_from_logits(lb), greedy_from_logits(lr)
            dis = btk != rtk
            # 只在不一致位置混合；其餘位置的 logits 與 base 完全相同（逐位元），故兩模型相同時等於 base 取樣。
            # 以 float32 混合再轉回原 dtype；lam=1 時 0*lb+1*lr 在 lb 有限時恰等於 lr。
            mix = ((1.0 - lam) * lb.float() + lam * lr.float()).to(lb.dtype)
            nxt = sample_from_logits(torch.where(dis.unsqueeze(-1), mix, lb), temperature, top_p, generator=gen)
        else:
            btk, rtk = greedy_from_logits(lb), greedy_from_logits(lr)
            dis = btk != rtk
            if cond == "all_replace":
                dis = torch.ones_like(dis)
            sampled = sample_from_logits(lb, temperature, top_p, generator=gen)
            nxt = torch.where(dis, rtk, sampled)
        steps_rows += rows.numel()
        w = rows[act]
        out_tok[w, t] = nxt[act]
        if cond in ("rl_replace", "all_replace", "geo_mix"):
            n_repl[w] += dis[act].long()
        done = (nxt.unsqueeze(-1) == stop).any(-1) & act
        finished[rows[done]] = True
        act = act & ~done
        if not bool(act.any()):
            break
        if compact and int(act.sum()) < 0.75 * rows.numel():
            keep = torch.nonzero(act).flatten()
            rows, nxt = rows[keep], nxt[keep]
            cur_attn, cur_pos = cur_attn[keep], cur_pos[keep]
            for c in (cb, cr):
                if c is not None:
                    c.batch_select_indices(keep)
            act = torch.ones_like(keep, dtype=torch.bool)
        cur_ids = nxt.unsqueeze(-1)
        cur_attn = torch.cat([cur_attn, torch.ones_like(cur_attn[:, :1])], dim=-1)
        cur_pos = cur_pos[:, -1:] + 1
    res = []
    out_tok = out_tok.cpu()
    for i in range(B):
        row = out_tok[i]
        m = int((row >= 0).sum())
        toks = row[:m].tolist()
        fin = bool(finished[i])
        if toks and toks[-1] in STOP_IDS and fin:
            body = toks[:-1]
        else:
            body = toks
        res.append((tok.decode(body, skip_special_tokens=True), fin, m, int(n_repl[i])))
    if log is not None:
        log["steps_rows"] = steps_rows
    return res


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--cond", required=True, choices=["base", "rl", "rl_replace", "all_replace", "geo_mix"])
    ap.add_argument("--lam", type=float, default=None, help="geo_mix 的 RL 權重 λ（geo_mix 必填；其他 cond 不可給）")
    ap.add_argument("--base-model", required=True)
    ap.add_argument("--rl-model", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--split", default="test")
    ap.add_argument("--n", type=int, default=128)
    ap.add_argument("--temperature", type=float, default=None)
    ap.add_argument("--top-p", type=float, default=None)
    ap.add_argument("--rows-per-batch", type=int, default=256, help="每批列數（題數 = rows/n）")
    ap.add_argument("--shard", default="0/1")
    ap.add_argument("--seed", type=int, default=1234)
    ap.add_argument("--limit", type=int, default=None)
    ap.add_argument("--no-compact", action="store_true")
    ap.add_argument("--device", default="cuda")
    ap.add_argument("--dtype", default=None)
    ap.add_argument("--max-new", type=int, default=None)
    ap.add_argument("--data", default=None)
    a = ap.parse_args()

    if a.cond == "geo_mix":
        assert a.lam is not None and 0.0 <= a.lam <= 1.0, "--cond geo_mix 需要 --lam in [0,1]"
    else:
        assert a.lam is None, "--lam 只用於 --cond geo_mix"
    pc = pairs_cfg()
    temp = pc["temperature"] if a.temperature is None else a.temperature
    top_p = pc["top_p"] if a.top_p is None else a.top_p
    si, sn = map(int, a.shard.split("/"))
    rows = shard(load_data(a.split, a.data), si, sn)
    if a.limit:
        rows = rows[: a.limit]
    outp = os.path.join(a.out, f"raw_{si}of{sn}.jsonl.gz")
    if os.path.exists(outp):
        print("exists, skip", outp); return
    dt = getattr(torch, a.dtype or pc["dtype"])
    tok = AutoTokenizer.from_pretrained(a.base_model, padding_side="left")
    if tok.pad_token is None:
        tok.pad_token = "<|endoftext|>"
    load = lambda p: AutoModelForCausalLM.from_pretrained(p, torch_dtype=dt).to(a.device).eval()
    need_b = a.cond in ("base", "rl_replace", "all_replace", "geo_mix")
    need_r = a.cond in ("rl", "rl_replace", "all_replace", "geo_mix")
    mb = load(a.base_model) if need_b else None
    mr = (mb if (need_r and a.rl_model == a.base_model and mb is not None) else load(a.rl_model)) if need_r else None
    if mb is not None and mr is not None:
        assert mb.config.vocab_size == mr.config.vocab_size, "vocab size differs"
    max_new = a.max_new or pc["max_new_tokens"]
    qpb = max(1, a.rows_per_batch // a.n)
    os.makedirs(a.out, exist_ok=True)
    tmp = outp + ".tmp"
    t0 = time.time()
    with open_gz_write(tmp) as f:
        for bi in range(0, len(rows), qpb):
            chunk = rows[bi: bi + qpb]
            gen = torch.Generator(device=a.device)
            gen.manual_seed(a.seed + 1000 * si + bi)
            lg = {}
            res = decode_batch(tok, mb, mr, [build_prompt(r["question"], pc) for r in chunk], a.n, a.cond,
                               temp, top_p, max_new, pc["max_model_len"], gen, compact=not a.no_compact, log=lg,
                               lam=(a.lam if a.cond == "geo_mix" else 0.5))
            for qi, r in enumerate(chunk):
                part = res[qi * a.n:(qi + 1) * a.n]
                f.write(json.dumps({"id": r["id"], "texts": [p[0] for p in part], "finished": [int(p[1]) for p in part],
                                    "ntok": [p[2] for p in part], "n_repl": [p[3] for p in part]}) + "\n")
            f.flush()
            print(f"batch {bi // qpb + 1}/{(len(rows) + qpb - 1) // qpb} elapsed {time.time() - t0:.0f}s "
                  f"row-steps {lg['steps_rows']}", flush=True)
    os.replace(tmp, outp)
    json.dump({"cond": a.cond, "base": a.base_model, "rl": a.rl_model, "temperature": temp, "top_p": top_p, "n": a.n,
               "max_new": max_new, "split": a.split, "shard": a.shard, "seed": a.seed, "compact": not a.no_compact,
               "engine": "hf-lockstep", "seconds": time.time() - t0, **({"lam": a.lam} if a.cond == "geo_mix" else {})},
              open(os.path.join(a.out, f"meta_{si}of{sn}.json"), "w"))


if __name__ == "__main__":
    main()
