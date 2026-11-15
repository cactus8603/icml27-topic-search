#!/usr/bin/env python
"""Λ 計算：防禦情境訓練損失梯度 與 非誘發情境特質 NLL 梯度 的內積。

定義（一步 SGD、學習率 η）：
    θ' = θ - η ∇L_def(θ)
    τ  = 非誘發情境下「特質回應」的 token 平均 log-prob = -L_τ
    Δτ ≈ -η <∇τ, ∇L_def> = +η <∇L_τ, ∇L_def>  ≡  Λ（取 η = 1）
Λ > 0 表示：在防禦情境下做一步訓練，會讓「非誘發情境」的特質機率上升（= 漏出）。

只對「實際可訓練的參數」計算：
  param_mode = lora_B     : LoRA 的 B 矩陣（step 0 時 B=0，A 的梯度恆為 0，所以只需 B）。
                            A 以 kaiming_uniform(seed) 初始化，可用多個 seed 平均。
  param_mode = full_layers: 指定層（chunk）的全部權重。
  groups 把參數依層切成多組（例如 2611.12864 的 5 個層段），每組各自報告 Λ。

不依賴 peft／兩篇論文的程式碼；LoRA 以 20 行自行實作（與 peft 預設初始化等價）。
輸出：指定跑次目錄下的 lambda_raw.json（逐項）與 lambda_summary.csv（彙整）。
"""
import argparse
import csv
import json
import math
import os
import random
import sys
import time
from contextlib import contextmanager

import torch
import torch.nn as nn
import torch.nn.functional as F
import yaml

PHASE_A_RESULTS = "/work/ChihChungHsu/cactus8603/icml/runs/inoculation-lambda/phasea_legacy/manual/results"


# ----------------------------------------------------------------------------
# 資料
# ----------------------------------------------------------------------------
def read_jsonl(path):
    with open(path, "r", encoding="utf-8") as f:
        return [json.loads(l) for l in f if l.strip()]


def to_messages(rec):
    if "messages" in rec:
        return [dict(m) for m in rec["messages"]]
    if "prompt" in rec and "completion" in rec:
        return [{"role": "user", "content": rec["prompt"]},
                {"role": "assistant", "content": rec["completion"]}]
    raise ValueError(f"unsupported record keys: {list(rec)}")


def select_records(spec, base_dir, exact=False):
    """spec: {file, n, offset, shuffle_seed, transform}. 先（可選）以固定種子洗牌，再切 [offset, offset+n)。"""
    path = spec["file"]
    if not os.path.isabs(path):
        path = os.path.join(base_dir, path)
    recs = read_jsonl(path)
    if spec.get("shuffle_seed") is not None:
        random.Random(spec["shuffle_seed"]).shuffle(recs)
    off, n = spec.get("offset", 0), spec.get("n", len(recs))
    recs = recs[off:off + n]
    if exact and len(recs) != n:
        raise ValueError(f"{path} contains {len(recs)} records for registered slice [{off}, {off+n})")
    msgs = [to_messages(r) for r in recs]
    if spec.get("transform") == "upper_assistant":      # allcaps 特質：助理回應轉大寫（Grant 資料管線同法）
        for m in msgs:
            for x in m:
                if x["role"] == "assistant":
                    x["content"] = x["content"].upper()
    return msgs


def load_prompts(path):
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)["prompts"]


def apply_context(messages, system=None, prefix=None):
    """複製 Grant 的 inject_prompt_into_chat(mode=prepend_user)：
    system=None 則不插入 system（用模型 chat template 的預設，2611.12864 的做法）。"""
    messages = [dict(m) for m in messages]
    has_sys = messages and messages[0]["role"] == "system"
    if system is not None:
        if has_sys:
            messages[0]["content"] = system
        else:
            messages.insert(0, {"role": "system", "content": system})
    if prefix:
        for m in messages:
            if m["role"] == "user":
                m["content"] = prefix + "\n\n" + m["content"]
                break
        else:
            raise ValueError("no user turn")
    return messages


def tokenize_items(tok, msgs_list, max_len, loss_mask):
    """回傳 list[(ids, labels)]；loss_mask = all（全 token，兩篇論文預設）| response。"""
    out = []
    for msgs in msgs_list:
        text = tok.apply_chat_template(msgs, tokenize=False)
        ids = tok(text, add_special_tokens=False, truncation=True, max_length=max_len)["input_ids"]
        labels = list(ids)
        if loss_mask == "response":
            ptxt = tok.apply_chat_template(msgs[:-1], tokenize=False, add_generation_prompt=True)
            k = len(tok(ptxt, add_special_tokens=False)["input_ids"])
            if k >= len(ids):
                continue
            labels[:k] = [-100] * k
        out.append((ids, labels))
    return out


def batches(items, micro, pad_id, device):
    for i in range(0, len(items), micro):
        chunk = items[i:i + micro]
        L = max(len(x[0]) for x in chunk)
        ids = torch.full((len(chunk), L), pad_id, dtype=torch.long)
        lab = torch.full((len(chunk), L), -100, dtype=torch.long)
        att = torch.zeros((len(chunk), L), dtype=torch.long)
        for j, (a, b) in enumerate(chunk):
            ids[j, :len(a)] = torch.tensor(a)
            lab[j, :len(b)] = torch.tensor(b)
            att[j, :len(a)] = 1
        yield ids.to(device), att.to(device), lab.to(device)


# ----------------------------------------------------------------------------
# 模型：層、LoRA-B、PPS hook
# ----------------------------------------------------------------------------
def get_layers(model):
    for path in ("model.layers", "model.model.layers", "transformer.h"):
        obj = model
        try:
            for p in path.split("."):
                obj = getattr(obj, p)
            return obj
        except AttributeError:
            continue
    raise RuntimeError("找不到 decoder layers")


class LoraB(nn.Module):
    """y = W x + (alpha/r) B (A x)；A 為固定 buffer、B 為 fp32 可訓練參數且初值 0。"""

    def __init__(self, base, r, alpha):
        super().__init__()
        self.base, self.r, self.scale = base, r, alpha / r
        self.register_buffer("A", torch.zeros(r, base.in_features), persistent=False)
        self.B = nn.Parameter(torch.zeros(base.out_features, r, dtype=torch.float32))

    def reset_A(self, seed):
        g = torch.Generator().manual_seed(seed)
        A = torch.empty(self.r, self.base.in_features)
        nn.init.kaiming_uniform_(A, a=math.sqrt(5), generator=g)   # 與 peft 預設相同
        self.A.copy_(A.to(self.A.device))

    def forward(self, x):
        y = self.base(x)
        z = F.linear(F.linear(x, self.A.to(x.dtype)), self.B.to(x.dtype))
        return y + self.scale * z


def install_lora(model, targets, layers_idx, r, alpha):
    layers = get_layers(model)
    mods = []
    for i in layers_idx:
        layer = layers[i]
        for name, m in list(layer.named_modules()):
            if isinstance(m, nn.Linear) and name.split(".")[-1] in targets:
                parent = layer
                parts = name.split(".")
                for p in parts[:-1]:
                    parent = getattr(parent, p)
                dev = m.weight.device
                lb = LoraB(m, r, alpha).to(dev)
                setattr(parent, parts[-1], lb)
                mods.append((i, lb))
    return mods


class PPSHook:
    """與 Grant 的 steering_hooks 同義：在 layers[layer-1] 的輸出加上 eps * vec[layer]（所有位置）。"""

    def __init__(self, model, vec_path, layer, eps, smoke=False):
        v = torch.load(vec_path, map_location="cpu", weights_only=True)
        assert v.dim() == 2, "預期 [n_blocks+1, hidden]"
        if smoke and v.size(1) != model.config.hidden_size:   # 只在 smoke：小模型維度不符時改用隨機向量（同範數）
            r = torch.randn(v.size(0), model.config.hidden_size, generator=torch.Generator().manual_seed(0))
            v = r / r.norm(dim=1, keepdim=True) * v.norm(dim=1, keepdim=True)
        self.vec = v[layer].float() * eps
        self.mod = get_layers(model)[layer - 1]
        self.h = None

    def __enter__(self):
        def hook(module, inp, out):
            if isinstance(out, tuple):
                return (out[0] + self.vec.to(out[0].device, out[0].dtype),) + out[1:]
            return out + self.vec.to(out.device, out.dtype)
        self.h = self.mod.register_forward_hook(hook)
        return self

    def __exit__(self, *a):
        self.h.remove()


@contextmanager
def nullctx():
    yield


# ----------------------------------------------------------------------------
# 梯度蒐集
# ----------------------------------------------------------------------------
def collect_grad(model, params, items, micro, pad_id, device):
    """回傳 (fp32 梯度 list, 平均 token NLL)。損失 = 全部 token 的平均 NLL（全域 token 平均，與批次切分無關）。"""
    n_tok = sum(sum(1 for t in lab[1:] if t != -100) for _, lab in items)
    assert n_tok > 0
    acc = [torch.zeros_like(p, dtype=torch.float32) for p in params]
    tot = 0.0
    for ids, att, lab in batches(items, micro, pad_id, device):
        logits = model(input_ids=ids, attention_mask=att).logits
        nll = F.cross_entropy(logits[:, :-1].float().reshape(-1, logits.size(-1)),
                              lab[:, 1:].reshape(-1), ignore_index=-100, reduction="sum")
        (nll / n_tok).backward()
        tot += nll.item()
        for a, p in zip(acc, params):
            if p.grad is not None:
                a += p.grad.float()
                p.grad = None
        del logits, nll
    return acc, tot / n_tok


def token_count(items):
    return sum(t != -100 for _, labels in items for t in labels[1:])


def group_stats(ga, gb, group_slices):
    """各組的 <ga,gb>、|ga|^2、|gb|^2。"""
    res = {}
    for name, idxs in group_slices.items():
        d = na = nb = 0.0
        for i in idxs:
            d += (ga[i] * gb[i]).sum().item()
            na += (ga[i] * ga[i]).sum().item()
            nb += (gb[i] * gb[i]).sum().item()
        res[name] = (d, na, nb)
    return res


# ----------------------------------------------------------------------------
# 主流程
# ----------------------------------------------------------------------------
def resolve_groups(cfg, n_layers):
    groups = []
    for g in cfg.get("groups", [{"name": "all", "layers": "all"}]):
        if g.get("layers") == "all":
            groups.append((g["name"], list(range(n_layers))))
        else:
            groups.append((g["name"], list(g["layers"])))
    ch = cfg.get("chunks")
    if ch:
        size = ch.get("size", 3)
        for p in ch["rel_positions"]:
            s = int(round(p * (n_layers - size)))
            groups.append((f"chunk_{int(round(p * 100)):03d}_L{s}-{s + size - 1}", list(range(s, s + size))))
    return groups


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", required=True)
    ap.add_argument("--out", default=None, help="預設共享 runs/inoculation-lambda/phasea_legacy/manual/results/<name>")
    ap.add_argument("--smoke", action="store_true", help="CPU 極小測試：覆寫模型、樣本數、防禦數")
    ap.add_argument("--model", default=None, help="覆寫 config 的 model")
    ap.add_argument("--device", default=None)
    ap.add_argument("--finite_diff", action="store_true",
                    help="閘門 1：實際做 1 步 SGD（B -= eta*gL），量 Δτ，與預測 eta*Λ 比；需 param_mode=lora_B，建議 dtype=float32")
    ap.add_argument("--fd_target", type=float, default=None, help="覆寫 criteria 的 fd_target_dtau")
    args = ap.parse_args()

    here = os.path.dirname(os.path.abspath(__file__))
    root = os.path.dirname(here)
    cfg = yaml.safe_load(open(args.config))
    data_root = cfg.get("data_root", "")

    device = args.device or ("cuda" if torch.cuda.is_available() and not args.smoke else "cpu")
    model_id = args.model or cfg["model"]
    dtype = {"bfloat16": torch.bfloat16, "float32": torch.float32}[cfg.get("dtype", "bfloat16")]
    if args.smoke:
        dtype = torch.float32
    crit = yaml.safe_load(open(os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "configs", "criteria.yaml")))
    if args.finite_diff:
        assert cfg["param_mode"] == "lora_B"
        cfg["dtype"] = crit["gate1"]["fd_dtype"]
        cfg["n_folds"] = 1
        cfg["lora"]["seeds"] = cfg["lora"]["seeds"][:crit["gate1"]["fd_n_lora_seeds"]]
        cfg["microbatch"] = min(cfg.get("microbatch", 4), 2)
    name = cfg["name"] + ("_fd" if args.finite_diff else "") + ("_smoke" if args.smoke else "")
    out_dir = args.out or os.path.join(PHASE_A_RESULTS, name)
    os.makedirs(out_dir, exist_ok=False)
    dtype = {"bfloat16": torch.bfloat16, "float32": torch.float32}[cfg.get("dtype", "bfloat16")]
    if args.smoke:
        dtype = torch.float32
    n_train = cfg["train"]["n"]
    micro = cfg.get("microbatch", 4)
    seeds = cfg.get("lora", {}).get("seeds", [0])
    n_folds = cfg.get("n_folds", 1)
    if args.smoke:
        cfg["train"]["n"] = n_train = 2
        for t in cfg["tau_sets"].values():
            t["n"] = 2
        cfg["defenses"] = cfg["defenses"][:3]
        seeds, n_folds, micro = seeds[:1], 1, 2
        cfg["max_len"] = min(cfg.get("max_len", 512), 512)

    from transformers import AutoModelForCausalLM, AutoTokenizer
    print(f"[load] {model_id} device={device} dtype={dtype}", flush=True)
    tok = AutoTokenizer.from_pretrained(model_id)
    model = AutoModelForCausalLM.from_pretrained(model_id, torch_dtype=dtype).to(device)
    model.eval()                       # 關閉 dropout，求確定性梯度
    for p in model.parameters():
        p.requires_grad_(False)
    layers = get_layers(model)
    n_layers = len(layers)
    pad_id = tok.pad_token_id if tok.pad_token_id is not None else tok.eos_token_id
    groups = resolve_groups(cfg, n_layers)
    union_layers = sorted({i for _, ls in groups for i in ls})
    print(f"[groups] n_layers={n_layers} " + "; ".join(f"{g}:{l[0]}-{l[-1]}" for g, l in groups), flush=True)

    # --- 可訓練參數 ---
    mode = cfg["param_mode"]
    lora_mods = []
    if mode == "lora_B":
        lc = cfg["lora"]
        lora_mods = install_lora(model, set(lc["targets"]), union_layers, lc["r"], lc["alpha"])
        params, owner = [], []
        for li, lb in lora_mods:
            lb.B.requires_grad_(True)
            params.append(lb.B)
            owner.append(li)
    elif mode == "full_layers":
        params, owner = [], []
        for li in union_layers:
            for p in layers[li].parameters():
                p.requires_grad_(True)
                params.append(p)
                owner.append(li)
    else:
        raise ValueError(mode)
    group_slices = {g: [i for i, o in enumerate(owner) if o in set(ls)] for g, ls in groups}
    print(f"[params] n_tensors={len(params)} n_elems={sum(p.numel() for p in params)/1e6:.1f}M", flush=True)

    # --- 資料 ---
    max_len = cfg.get("max_len", 512)
    loss_mask = cfg.get("loss_mask", "all")
    train_msgs = select_records(cfg["train"], data_root, exact=not args.smoke)
    tau_data = {k: select_records(v, data_root, exact=not args.smoke) for k, v in cfg["tau_sets"].items()}
    contexts = cfg["contexts"]            # list of {name, system, prefix | prefix_file}
    for c in contexts:
        if c.get("prefix_file"):
            c["prefix"] = load_prompts(os.path.join(data_root, c["prefix_file"]))[c.get("prefix_index", 0)]
    base_sys = cfg.get("baseline_system_prompt", None)

    # --- 防禦：每個防禦 = 一組 (prefix 列表 | None) + 可選 PPS ---
    defenses = []
    for d in cfg["defenses"]:
        e = {"name": d["name"], "kind": d.get("kind", "none"), "meta": {k: v for k, v in d.items() if k != "name"}}
        if e["kind"] == "ip":
            prompts = load_prompts(os.path.join(data_root, d["prompt_file"]))
            e["prefixes"] = prompts                   # 銀行取期望：每筆樣本與每個提示都配對，等權平均
        else:
            e["prefixes"] = [None]
        if e["kind"] == "pps":
            e["pps"] = (os.path.join(data_root, d["vector"]), d.get("layer", 20), d["eps"])
        defenses.append(e)

    def tau_items(tset, ctx):
        msgs = [apply_context(m, system=ctx.get("system", base_sys), prefix=ctx.get("prefix")) for m in tau_data[tset]]
        items = tokenize_items(tok, msgs, max_len, "response")
        if not args.smoke and len(items) != len(msgs):
            raise ValueError(f"{tset}/{ctx['name']} retained {len(items)} of {len(msgs)} frozen responses")
        return items

    def def_items(d, sl):
        msgs = []
        for m in train_msgs[sl]:
            for pf in d["prefixes"]:
                msgs.append(apply_context(m, system=base_sys, prefix=pf))
        return tokenize_items(tok, msgs, max_len, loss_mask)

    rows = []
    fd_state = {}
    tau_cache = {"_ctxs": contexts}
    t0 = time.time()
    fold_slices = [slice(i * n_train // n_folds, (i + 1) * n_train // n_folds) for i in range(n_folds)]
    for seed in seeds:
        if mode == "lora_B":
            for _, lb in lora_mods:
                lb.reset_A(seed * 100003 + id_hash(lb, lora_mods))
        # 1) τ 梯度：每個 (tau_set, context)。stream_tau=true（大模型全參數層段）時不快取，逐一算、用完即丟。
        stream = cfg.get("stream_tau", False)
        gtau, tau_nll, tau_counts = {}, {}, {}

        def compute_tau(ts, ctx):
            items = tau_items(ts, ctx)
            g, nll = collect_grad(model, params, items, micro, pad_id, device)
            tau_nll[(ts, ctx["name"])] = nll
            tau_counts[(ts, ctx["name"])] = (len(items), token_count(items))
            print(f"[seed {seed}] tau {ts}/{ctx['name']} n={len(items)} nll={nll:.4f} ({time.time()-t0:.0f}s)", flush=True)
            return g

        def iter_tau():
            for ts in tau_data:
                for ctx in contexts:
                    key = (ts, ctx["name"])
                    if stream:
                        yield key, compute_tau(ts, ctx)
                    else:
                        yield key, gtau[key]

        if not stream:
            for ts in tau_data:
                for ctx in contexts:
                    gtau[(ts, ctx["name"])] = compute_tau(ts, ctx)
        # 2) 防禦訓練梯度
        for d in defenses:
            for fi, sl in enumerate(fold_slices):
                items = def_items(d, sl)
                cm = PPSHook(model, *d["pps"], smoke=args.smoke) if d["kind"] == "pps" else nullctx()
                with cm:
                    gl, loss = collect_grad(model, params, items, micro, pad_id, device)
                for key, gt in iter_tau():
                    for g, (dot, na, nb) in group_stats(gl, gt, group_slices).items():
                        rows.append(dict(seed=seed, fold=fi, defense=d["name"], kind=d["kind"], tau_set=key[0], ctx=key[1],
                                         group=g, dot=dot, gL_norm=math.sqrt(na), gT_norm=math.sqrt(nb),
                                         cos=dot / (math.sqrt(na) * math.sqrt(nb) + 1e-30),
                                         train_loss=loss, tau_nll=tau_nll[key], n_train_items=len(items),
                                         n_train_sequences=len(train_msgs[sl]), n_prefixes=len(d["prefixes"]),
                                         n_train_tokens=token_count(items), n_tau_items=tau_counts[key][0], n_tau_tokens=tau_counts[key][1]))
                    del gt
                if args.finite_diff:
                    assert not stream, "有限差分需快取 τ 梯度（LoRA 模式）"
                    fd_apply(rows, gl, gtau, tau_nll, tau_cache, model, params, micro, pad_id, device, tau_items,
                             fd_state, crit["gate1"], args.fd_target, d["name"], seed, fi)
                del gl
                print(f"[seed {seed}] defense {d['name']} fold {fi} loss={loss:.4f} ({time.time()-t0:.0f}s)", flush=True)
        del gtau
    raw_path = os.path.join(out_dir, "lambda_raw.json")
    with open(raw_path + ".pending", "w") as stream:
        json.dump(dict(config=cfg, model=model_id, rows=rows, job_id=os.environ.get("SLURM_JOB_ID")), stream, indent=1)
    os.replace(raw_path + ".pending", raw_path)
    write_summary(rows, os.path.join(out_dir, "lambda_summary.csv"))
    print(f"[done] {len(rows)} rows -> {out_dir}", flush=True)


@torch.no_grad()
def nll_nograd(model, items, micro, pad_id, device):
    n_tok = sum(sum(1 for t in lab[1:] if t != -100) for _, lab in items)
    tot = 0.0
    for ids, att, lab in batches(items, micro, pad_id, device):
        lg = model(input_ids=ids, attention_mask=att).logits
        tot += F.cross_entropy(lg[:, :-1].float().reshape(-1, lg.size(-1)), lab[:, 1:].reshape(-1),
                               ignore_index=-100, reduction="sum").item()
    return tot / n_tok


def fd_apply(rows, gl, gtau, tau_nll, tau_cache, model, params, micro, pad_id, device, tau_items, st, g1, fd_target,
             dname, seed, fold):
    """對剛加入 rows 的列補上有限差分：eta 由 'none' 防禦校準，使預測 Δτ = fd_target（nats）。
    以 B -= eta*gL 做一步（中央差分：另做 +eta 反向，取 (nll(-)-nll(+))/2 以消除二階項）。"""
    target = fd_target if fd_target is not None else g1["fd_target_dtau"]
    mine = [r for r in rows if r["seed"] == seed and r["fold"] == fold and r["defense"] == dname and "dtau_true" not in r]
    if dname == "none" or ("eta", seed) not in st:
        ref = [r for r in mine if r["ctx"] == g1["fd_ctx"]][0]
        assert ref["dot"] > 0, "none 防禦的 Λ <= 0，無法校準 eta；請檢查資料／實作"
        st[("eta", seed)] = target / ref["dot"]
    eta = st[("eta", seed)]
    # 2026-11-14 修正：限制步長 eta*||gL|| <= fd_max_step（信賴半徑），避免低 cos 的 trait（spanish/optimistic）
    # 因 eta 被校準得很大、或 PPS 梯度範數大 10 倍，使一步落入非線性區（見 PREREG 修改紀錄）。
    max_step = g1.get("fd_max_step")
    if max_step:
        gn = mine[0]["gL_norm"]
        eta = min(eta, max_step / gn)
    for key in gtau:
        if key[1] != g1["fd_ctx"]:
            continue
        if key not in tau_cache:
            ts, cn = key
            ctx = [c for c in tau_cache["_ctxs"] if c["name"] == cn][0]
            tau_cache[key] = tau_items(ts, ctx)
        items = tau_cache[key]
        nll0 = tau_nll[key]
        for p, g in zip(params, gl):
            p.data.add_(g, alpha=-eta)
        nll_minus = nll_nograd(model, items, micro, pad_id, device)
        for p, g in zip(params, gl):
            p.data.add_(g, alpha=2 * eta)
        nll_plus = nll_nograd(model, items, micro, pad_id, device)
        for p, g in zip(params, gl):
            p.data.add_(g, alpha=-eta)           # 還原
        for r in mine:
            if (r["tau_set"], r["ctx"]) == key:
                r["eta"] = eta
                r["dtau_true"] = nll0 - nll_minus          # 單邊（沿 -g 一步）
                r["dtau_central"] = (nll_plus - nll_minus) / 2
                r["dtau_pred"] = eta * r["dot"]


def id_hash(lb, mods):
    """讓不同模組用不同（但可重現）的 A 種子。"""
    for k, (_, m) in enumerate(mods):
        if m is lb:
            return k
    return 0


def write_summary(rows, path):
    """按 token 數合併各 seed 的 fold 內積，再等權平均 seed。"""
    import statistics as st
    keyf = lambda r: (r["defense"], r["tau_set"], r["group"], r["ctx"])
    agg = {}
    for r in rows:
        agg.setdefault(keyf(r), []).append(r)
    out = []
    for k, rs in agg.items():
        by_seed = {}
        for row in rs:
            by_seed.setdefault(row["seed"], []).append(row)
        seed_dots = [sum(row["n_train_tokens"] * row["dot"] for row in folds) / sum(row["n_train_tokens"] for row in folds)
                     if all("n_train_tokens" in row for row in folds) else st.mean(row["dot"] for row in folds)
                     for folds in by_seed.values()]
        out.append(dict(defense=k[0], tau_set=k[1], group=k[2], ctx=k[3], n=len(rs),
                        lam=st.mean(seed_dots),
                        lam_sd=st.pstdev([x["dot"] for x in rs]) if len(rs) > 1 else 0.0,
                        cos=st.mean(x["cos"] for x in rs),
                        gL_norm=st.mean(x["gL_norm"] for x in rs), gT_norm=st.mean(x["gT_norm"] for x in rs),
                        train_loss=st.mean(x["train_loss"] for x in rs)))
    # ctx 平均
    meank = {}
    for o in out:
        meank.setdefault((o["defense"], o["tau_set"], o["group"]), []).append(o)
    for k, os_ in meank.items():
        out.append(dict(defense=k[0], tau_set=k[1], group=k[2], ctx="MEAN", n=sum(o["n"] for o in os_),
                        lam=st.mean(o["lam"] for o in os_), lam_sd=st.mean(o["lam_sd"] for o in os_),
                        cos=st.mean(o["cos"] for o in os_), gL_norm=st.mean(o["gL_norm"] for o in os_),
                        gT_norm=st.mean(o["gT_norm"] for o in os_), train_loss=st.mean(o["train_loss"] for o in os_)))
    with open(path, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(out[0].keys()))
        w.writeheader()
        w.writerows(out)


if __name__ == "__main__":
    main()
