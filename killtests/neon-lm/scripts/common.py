import os, json
ROOT = "/work/ChihChungHsu/cactus8603"
RUNS = os.environ.get("NEON_RUNS", f"{ROOT}/icml/runs/neon-lm")
KT = f"{ROOT}/icml27-topic-search/killtests/neon-lm"
RES = os.environ.get("NEON_RES", f"{KT}/results")
MODELS = f"{ROOT}/models"
SETTINGS = {
    "A": dict(base=f"{MODELS}/Qwen2.5-1.5B", kind="lm"),
    "B": dict(base=f"{MODELS}/Qwen2.5-0.5B", kind="lm"),
    "C": dict(base=f"{MODELS}/Qwen2.5-1.5B-Instruct", kind="gsm"),
}
STEPS = [100, 400]
W_NEG = [0.25, 0.5, 1.0, 1.5, 2.0]
W_POS = [-0.5, -1.0]
def wtag(w): return f"w{w:+.2f}"
def model_list(setting):
    """list of (name, dir). name: base, s{step}_w{w}"""
    out = [("base", SETTINGS[setting]["base"])]
    for s in STEPS:
        for w in W_NEG + W_POS:
            out.append((f"s{s}_{wtag(w)}", f"{RUNS}/{setting}/merged/s{s}_{wtag(w)}"))
    return out
# sampling grid: (temperature, top_p, min_p)
def grid():
    g = [(T, p, 0.0) for T in (0.7,0.8,0.9,1.0,1.1,1.2,1.3) for p in (0.9,0.95,1.0)]
    g += [(1.0, 1.0, m) for m in (0.02, 0.05, 0.1)]
    return g
def gname(c): return f"T{c[0]}_p{c[1]}_m{c[2]}"
def jdump(o, p):
    os.makedirs(os.path.dirname(p), exist_ok=True)
    json.dump(o, open(p, "w"), indent=1)
