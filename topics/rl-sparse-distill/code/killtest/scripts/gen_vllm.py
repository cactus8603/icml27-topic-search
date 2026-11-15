"""單一模型取樣（base／rl／WiSE-FT 點）。vLLM。raw 輸出：raw_{shard}.jsonl.gz（完整生成，gzip）。

用法例：
  gen_vllm.py --model PATH --out results/A/base --split test --n 128 --temperature 0.6 --top-p 0.95
  gen_vllm.py --model PATH --out results/A/rl_greedy --split all --n 1 --temperature 0 --top-p 1   # 閘門 0 重現用
"""
import argparse, os, sys, json
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from common import *
from artifacts import artifact_receipt, atomic_json, parse_shard, publish_raw, require, sha256, read_json

ap = argparse.ArgumentParser()
ap.add_argument("--model", required=True)
ap.add_argument("--out", required=True)
ap.add_argument("--split", default="test")
ap.add_argument("--n", type=int, default=128)
ap.add_argument("--temperature", type=float, default=None)
ap.add_argument("--top-p", type=float, default=None)
ap.add_argument("--shard", default="0/1")
ap.add_argument("--seed", type=int, default=1234)
ap.add_argument("--limit", type=int, default=None, help="只跑前 N 題（吞吐實測用）")
ap.add_argument("--gpu-mem", type=float, default=0.90)
a = ap.parse_args()

pc = pairs_cfg()
temp = pc["temperature"] if a.temperature is None else a.temperature
top_p = pc["top_p"] if a.top_p is None else a.top_p
si, sn = parse_shard(a.shard)
rows = shard(load_data(a.split), si, sn)
if a.limit:
    rows = rows[: a.limit]
outp = os.path.join(a.out, f"raw_{si}of{sn}.jsonl.gz")
meta_path = os.path.join(a.out, f"meta_{si}of{sn}.json")
if os.path.exists(outp):
    if os.environ.get("RSD_RUN_ID"):
        p = read_json(meta_path).get("provenance", {})
        require(p.get("run_id") == os.environ["RSD_RUN_ID"] and p.get("raw_sha256") == sha256(outp),
                "existing raw shard is not from this run")
    print("exists, skip", outp); sys.exit(0)

from vllm import LLM, SamplingParams
llm = LLM(a.model, dtype=pc["dtype"], max_model_len=pc["max_model_len"], gpu_memory_utilization=a.gpu_mem,
          seed=a.seed, enable_prefix_caching=True, tokenizer=os.environ.get("TOKENIZER_PATH") or None)
sp = SamplingParams(n=a.n, temperature=temp, top_p=top_p, max_tokens=pc["max_new_tokens"],
                    stop_token_ids=STOP_IDS, seed=None, skip_special_tokens=True)
outs = llm.generate([build_prompt(r["question"], pc) for r in rows], sp)
with publish_raw(outp) as tmp, open_gz_write(tmp) as f:
    for r, o in zip(rows, outs, strict=True):
        f.write(json.dumps({"id": r["id"], "texts": [c.text for c in o.outputs],
                            "finished": [int(c.finish_reason == "stop") for c in o.outputs],
                            "ntok": [len(c.token_ids) for c in o.outputs]}) + "\n")
meta = {"model": a.model, "temperature": temp, "top_p": top_p, "n": a.n, "max_tokens": pc["max_new_tokens"],
        "split": a.split, "shard": a.shard, "seed": a.seed, "engine": "vllm"}
if os.environ.get("RSD_RUN_ID") and not a.limit:
    meta["provenance"] = artifact_receipt(outp, os.environ["PAIR"], os.path.basename(a.out), "vllm",
                                          os.environ.get("SLURM_JOB_ID"), pc["dtype"], pc["max_model_len"])
    if "RSD_WISE_LAMBDA" in os.environ:
        meta["wise_lambda"] = float(os.environ["RSD_WISE_LAMBDA"])
        meta["wise_models"] = {"base": os.environ["RSD_WISE_BASE"], "rl": os.environ["RSD_WISE_RL"]}
atomic_json(meta_path, meta)
print("done", outp)
