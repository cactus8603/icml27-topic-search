"""vLLM generation. One process = one model, many configs x seeds."""
import sys, os, re, time, json, argparse, numpy as np
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from common import *
ap = argparse.ArgumentParser()
ap.add_argument("--setting", required=True); ap.add_argument("--name", required=True)
ap.add_argument("--model-dir", required=True)
ap.add_argument("--split", required=True)  # selftrain|dev|test
ap.add_argument("--configs", default="grid")  # grid | "T,p,m;T,p,m" | default
ap.add_argument("--seeds", type=int, nargs="+", default=[0])
ap.add_argument("--limit", type=int, default=0)
ap.add_argument("--out", default=None)
ap.add_argument("--save-text", action="store_true")
ap.add_argument("--tag", default="")
a = ap.parse_args()
kind = SETTINGS[a.setting]["kind"]
out = a.out or f"{RUNS}/{a.setting}/gen{a.tag}/{a.name}/{a.split}"
os.makedirs(out, exist_ok=True)
os.environ["VLLM_USE_FLASHINFER_SAMPLER"] = "0"
from vllm import LLM, SamplingParams
from transformers import AutoTokenizer
tok = AutoTokenizer.from_pretrained(SETTINGS[a.setting]["base"])
t0 = time.time()
llm = LLM(a.model_dir, tokenizer=SETTINGS[a.setting]["base"], dtype="float16", gpu_memory_utilization=0.85,
          max_model_len=1024 if kind == "gsm" else 384, enable_prefix_caching=True,
          generation_config="vllm", seed=0)
print("load_s", time.time() - t0, flush=True)
if a.configs == "grid": cfgs = grid()
elif a.configs == "default": cfgs = [("default", 0, 0)]
else: cfgs = [tuple(float(x) for x in c.split(",")) for c in a.configs.split(";")]

def last_number(s):
    m = re.findall(r"-?\d[\d,]*\.?\d*", s)
    if not m: return None
    x = m[-1].replace(",", "").rstrip(".")
    try: return float(x)
    except: return None
def gold(ans): return float(ans.split("####")[-1].strip().replace(",", ""))

if kind == "lm":
    d = np.load(f"{RUNS}/data/lm_{a.split}.npz"); P = d["prompt"]
    if a.limit: P = P[:a.limit]
    prompts = [{"prompt_token_ids": p.tolist()} for p in P]
else:
    D = json.load(open(f"{RUNS}/data/gsm_{a.split}.json"))
    if a.limit: D = D[:a.limit]
    INSTR = "\nPlease reason step by step, and put your final answer at the end."
    prompts_txt = [tok.apply_chat_template([{"role": "user", "content": x["q"] + INSTR}], tokenize=False, add_generation_prompt=True) for x in D]
    prompts = [{"prompt_token_ids": tok(t, add_special_tokens=False).input_ids} for t in prompts_txt]
    G = np.array([gold(x["a"]) for x in D])

for c in cfgs:
  for seed in a.seeds:
    fn = f"{out}/{gname(c) if c[0]!='default' else 'default'}_s{seed}"
    if os.path.exists(fn + ".npy") or os.path.exists(fn + ".npz"): continue
    t1 = time.time()
    if c[0] == "default":  # model default sampling (Qwen2.5-Instruct generation_config)
        kw = dict(temperature=0.7, top_p=0.8, top_k=20, repetition_penalty=1.1)
    else:
        kw = dict(temperature=c[0], top_p=c[1], min_p=c[2], top_k=-1)
    if kind == "lm":
        sps = [SamplingParams(max_tokens=256, ignore_eos=True, seed=seed * 1000003 + i, **kw) for i in range(len(prompts))]
        outs = llm.generate(prompts, sps, use_tqdm=False)
        arr = np.array([list(o.outputs[0].token_ids) for o in outs], dtype=np.int32)
        assert arr.shape[1] == 256, arr.shape
        np.save(fn + ".npy", arr)
        ntok = arr.size
    else:
        n = 1 if a.split == "selftrain" else 16
        sps = [SamplingParams(n=n, max_tokens=512, seed=seed * 1000003 + i, **kw) for i in range(len(prompts))]
        outs = llm.generate(prompts, sps, use_tqdm=False)
        if a.split == "selftrain":
            recs = [dict(q=i, p=prompts[i]["prompt_token_ids"], c=list(o.outputs[0].token_ids), fin=o.outputs[0].finish_reason) for i, o in enumerate(outs)]
            json.dump(recs, open(fn + ".json", "w")); open(fn + ".npy", "w").write("done")
            ntok = sum(len(r["c"]) for r in recs)
        else:
            corr = np.zeros((len(outs), n), dtype=bool); L = np.zeros((len(outs), n), dtype=np.int32); texts = []
            for i, o in enumerate(outs):
                for j, s in enumerate(o.outputs):
                    v = last_number(s.text); corr[i, j] = (v is not None and abs(v - G[i]) < 1e-4); L[i, j] = len(s.token_ids)
                if a.save_text: texts.append([s.text for s in o.outputs])
            np.savez(fn + ".npz", corr=corr, length=L)
            if a.save_text: json.dump(texts, open(fn + ".texts.json", "w"))
            ntok = int(L.sum())
            print(f"  pass@1={corr.mean():.4f} pass@16={corr.any(1).mean():.4f}", end="")
    print(f"{a.name} {a.split} {c} seed{seed} {time.time()-t1:.1f}s tok={ntok} tok/s={ntok/(time.time()-t1):.0f}", flush=True)
