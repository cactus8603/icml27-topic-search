"""GPU（或 --tiny 的 CPU smoke）：對固定範例 × 條件做 teacher-forced 前向，存
  (a) 回應 token 的逐層平均池化激活（L8 層，fp16）
  (b) 作者 13 個第 12 層邏輯探測器的原生分數（逐 token sigmoid 再對回應 token 平均，同 minimal demo）
以 shard 存檔、可續跑。只做監測資料的抽取，不訓練、不生成。
用法：python -I extract_acts.py --sets benign harm_train harm_test dec_train dec_test --out DIR [--limit N] [--bench]"""
import argparse, glob, json, os, sys, time
import numpy as np
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from common import load_config, render_cond

cfg = load_config()
PROBE_NAMES = ["HTML", "all-caps", "biology-focused", "chemistry-based", "comforting", "confused", "finnish",
               "german", "jokey", "literature-focused", "mathematical", "harmful", "deception"]


def expand_rows(sets, data_dir, limit=None):
    """範例 × 條件展開成 rows（list of dict，含 ex 欄位引用）。"""
    C = cfg["conds"]; rows = []
    for s in sets:
        with open(os.path.join(data_dir, s + ".jsonl")) as f:
            exs = [json.loads(l) for l in f]
        if limit:
            exs = [exs[i] for i in np.unique(np.linspace(0, len(exs) - 1, limit).astype(int))]
        for ex in exs:
            if s == "benign":
                key = ("benign_pos_H" if ex["role"] == "H" else "benign_pos") if ex["label"] else "benign_neg"
            else:
                key = f"{s}_{'pos' if ex['label'] else 'neg'}"
            for cond in C[key]:
                cond = cond.replace(":own", ":" + ex["word"])
                rows.append(dict(ex_id=ex["id"], set=s, concept=ex["concept"], label=ex["label"], role=ex["role"],
                                 cond=cond, prompt=ex["prompt"], response=ex["response"]))
    return rows


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--sets", nargs="+", required=True)
    ap.add_argument("--data-dir", default=os.path.join(cfg["paths"]["run_dir"], "data"))
    ap.add_argument("--out", required=True)
    ap.add_argument("--limit", type=int, default=None, help="每個 set 只取前 N 個範例（smoke／吞吐實測）")
    ap.add_argument("--batch-size", type=int, default=16)
    ap.add_argument("--shard-rows", type=int, default=2048)
    ap.add_argument("--tiny", action="store_true", help="CPU smoke：隨機初始化的小 Gemma2，不載入真實權重")
    ap.add_argument("--bench", action="store_true", help="只印吞吐（examples/s、tokens/s）不寫 shard")
    a = ap.parse_args()
    import torch
    from transformers import AutoTokenizer, AutoModelForCausalLM, AutoConfig
    os.makedirs(a.out, exist_ok=True)
    rows = expand_rows(a.sets, a.data_dir, a.limit)
    print(f"rows={len(rows)}", flush=True)

    mid = cfg["model"]["hf_id"]
    tok = AutoTokenizer.from_pretrained(mid)
    tok.padding_side = "left"
    if not tok.pad_token:
        tok.pad_token_id = tok.unk_token_id if tok.unk_token else tok.eos_token_id
    if a.tiny:
        c = AutoConfig.from_pretrained(mid)
        c.hidden_size = 64; c.intermediate_size = 128; c.num_hidden_layers = 6; c.num_attention_heads = 4
        c.num_key_value_heads = 2; c.head_dim = 16; c.layer_types = c.layer_types[:6]
        torch.manual_seed(0)
        model = AutoModelForCausalLM.from_config(c).float().eval()
        layers = [1, 2, 3, 4, 5]; dev = "cpu"; dtype = torch.float32; evl = 2
    else:
        dev = "cuda"; dtype = torch.bfloat16
        model = AutoModelForCausalLM.from_pretrained(mid, dtype=dtype, attn_implementation="eager").to(dev).eval()
        layers = cfg["layers_L8"]; evl = cfg["model"]["evaded_layer"]
    D = model.config.hidden_size
    # 作者探測器（只在 hidden=3584 時有意義；tiny 用隨機探測器只測程式路徑）
    pdir = os.path.join(cfg["paths"]["minimal_repo"], "probes")
    W = np.zeros((len(PROBE_NAMES), D), np.float32); Bv = np.zeros(len(PROBE_NAMES), np.float32)
    for i, n in enumerate(PROBE_NAMES):
        if D == 3584:
            d = torch.load(os.path.join(pdir, f"{n}_weights.pt"), map_location="cpu", weights_only=False)
            p = next(iter(d.values()))
            W[i] = np.asarray(p["weight"], np.float32).reshape(-1); Bv[i] = float(np.asarray(p["bias"]).reshape(-1)[0])
        else:
            W[i] = np.random.default_rng(i).normal(size=D)
    Wt = torch.tensor(W, device=dev); Bt = torch.tensor(Bv, device=dev)

    # 預先 tokenization
    def enc(r):
        p = render_cond(r["cond"], r["prompt"][: cfg["max_prompt_chars"]], cfg["templates"])
        pt = tok.apply_chat_template([{"role": "user", "content": p}], tokenize=False, add_generation_prompt=True)
        pi = tok(pt, add_special_tokens=False)["input_ids"]
        ri = tok(r["response"], add_special_tokens=False)["input_ids"][: cfg["max_resp_tokens"]]
        return pi, ri
    t0 = time.time()
    toks = [enc(r) for r in rows]
    order = np.argsort([len(p) + len(q) for p, q in toks])
    if a.bench:   # 2026-11-14 審查修正：吞吐實測改隨機順序（依長度排序會先跑最短列，高估吞吐）
        order = np.random.default_rng(0).permutation(len(rows))
        n_full = len(expand_rows(["benign", "harm_train", "harm_test", "dec_train", "dec_test"], a.data_dir))
    print(f"tokenized in {time.time()-t0:.1f}s; mean len={np.mean([len(p)+len(q) for p,q in toks]):.0f}", flush=True)

    n_sh = (len(rows) + a.shard_rows - 1) // a.shard_rows
    t_start = time.time(); done = 0; ntok = 0
    for sh in range(n_sh):
        fn = os.path.join(a.out, f"shard_{sh:04d}.npz")
        idx = order[sh * a.shard_rows:(sh + 1) * a.shard_rows]
        if os.path.exists(fn) and not a.bench:
            continue
        acts = np.zeros((len(idx), len(layers), D), np.float16); pr = np.zeros((len(idx), len(PROBE_NAMES)), np.float32)
        for b0 in range(0, len(idx), a.batch_size):
            bi = idx[b0:b0 + a.batch_size]
            seqs = [toks[i][0] + toks[i][1] for i in bi]; T = max(len(s) for s in seqs)
            ids = torch.full((len(bi), T), tok.pad_token_id, dtype=torch.long); am = torch.zeros((len(bi), T), dtype=torch.long)
            rm = torch.zeros((len(bi), T), dtype=torch.float32)
            for j, i in enumerate(bi):
                s = seqs[j]; ids[j, T - len(s):] = torch.tensor(s); am[j, T - len(s):] = 1
                rm[j, T - len(toks[i][1]):] = 1
            pos = (am.cumsum(1) - 1).clamp(min=0)
            with torch.no_grad():
                hs = model(input_ids=ids.to(dev), attention_mask=am.to(dev), position_ids=pos.to(dev),
                           output_hidden_states=True).hidden_states
            m = rm.to(dev); den = m.sum(1, keepdim=True)
            for li, l in enumerate(layers):
                h = hs[l + 1].float()
                acts[b0:b0 + len(bi), li] = ((h * m[..., None]).sum(1) / den).cpu().numpy().astype(np.float16)
            h = hs[evl + 1].float()
            sg = torch.sigmoid(h @ Wt.T + Bt)                       # [B,T,P]
            pr[b0:b0 + len(bi)] = ((sg * m[..., None]).sum(1) / den).cpu().numpy()
            ntok += int(am.sum()); done += len(bi)
            if a.bench and done >= 400:
                break
        if a.bench:
            dt = time.time() - t_start
            print(json.dumps(dict(rows_done=done, secs=dt, rows_per_s=done / dt, tok_per_s=ntok / dt,
                                  n_full_rows=n_full,   # 2026-11-14 審查修正：原本用 bench 列數估總時數（低估約 50 倍）
                                  est_total_gpu_h=n_full / (done / dt) / 3600)), flush=True)
            return
        np.savez(fn, acts=acts, probes=pr, row_idx=idx)
        meta = [{k: rows[i][k] for k in ("ex_id", "set", "concept", "label", "role", "cond")} for i in idx]
        with open(os.path.join(a.out, f"shard_{sh:04d}.meta.json"), "w") as f:
            json.dump(meta, f)
        print(f"shard {sh+1}/{n_sh} rows={len(idx)} elapsed={time.time()-t_start:.0f}s rows/s={done/(time.time()-t_start):.1f}", flush=True)
    json.dump(dict(layers=layers, probe_names=PROBE_NAMES, D=D, n_rows=len(rows), sets=a.sets, tiny=a.tiny),
              open(os.path.join(a.out, "cache_info.json"), "w"))


if __name__ == "__main__":
    main()
