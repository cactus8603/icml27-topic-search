"""Pair B reproduction and separate short-budget qualification."""

import argparse
import gzip
from importlib.metadata import version
import json
from pathlib import Path
import subprocess
import sys

SCRIPT = Path(__file__).resolve().parent
sys.path.insert(0, str(SCRIPT))
from aggregate import gate0, per_q
from artifacts import (
    atomic_json, check_gate_minus1, grader_receipt, load_verified, publish_raw,
    read_json, record_correctness, require, require_current_correctness,
    sha256, validate_records,
)
from common import STOP_IDS, build_prompt, cfg, load_data, pairs_cfg


def model_path(shared, side, settings, pairs):
    name = pairs["pairs"]["B"][side].replace("/", "--")
    return shared / "hf_home" / "hub" / f"models--{name}" / "snapshots" / settings["snapshots"][side]


def run_script(name, *args):
    subprocess.run([sys.executable, "-BI", str(SCRIPT / name), *map(str, args)], check=True)


def check_inputs(shared, settings, pairs):
    from transformers import AutoTokenizer

    require(version("vllm") == "0.33.0", "vLLM 版本與凍結設定不符")
    grader_receipt()
    for side in ("base", "rl"):
        path = model_path(shared, side, settings, pairs)
        index = path / "model.safetensors.index.json"
        weights = set(read_json(index)["weight_map"].values()) if index.is_file() else {"model.safetensors"}
        require(weights and all((path / file).is_file() and (path / file).stat().st_size > 0
                                for file in weights), f"模型分片不齊：{side}")
        config = read_json(path / "config.json")
        require(config["max_position_embeddings"] >= settings["reproduction"]["max_model_len"],
                f"權重的原生 context 太短：{side}")
        tokenizer = AutoTokenizer.from_pretrained(path, local_files_only=True)
        require(tokenizer.convert_tokens_to_ids(["<|im_end|>", "<|endoftext|>"]) == STOP_IDS,
                f"Tokenizer 停止 token 不符：{side}")
        prompt = build_prompt("Check the template.", pairs, "B")
        user_prompt = "Question:\nCheck the template.\nAnswer:\nLet's think step by step.\n"
        require(tokenizer.apply_chat_template([{"role": "user", "content": user_prompt}],
                                              tokenize=False, add_generation_prompt=True) == prompt,
                f"短預算模板與凍結文字不符：{side}")
    run_script("pair_b_upstream.py", "check", shared / "tmp" / settings["upstream_cache"])


def long_receipt(run, side, settings, pairs, shared, commit, job, version_id):
    spec = settings["reproduction"]
    return {
        "commit": commit, "job_id": job, "side": side, "vllm": version_id,
        "snapshot": model_path(shared, side, settings, pairs).name, "dtype": pairs["dtype"],
        "seed": spec["seed"], "n": 1, "max_tokens": spec["max_tokens"],
        "max_model_len": spec["max_model_len"], "stop": spec["stop"], "stop_ids": STOP_IDS,
        "sampling": spec[side], "prompts_sha256": sha256(run / "upstream-prompts.json"),
        "settings_sha256": sha256(SCRIPT.parent / "configs" / "pair_b.json"),
        "raw_sha256": sha256(run / f"upstream-{side}.jsonl.gz"),
    }


def sample_long(run, side, settings, pairs, shared, commit, job):
    from vllm import LLM, SamplingParams

    prompts = read_json(run / "upstream-prompts.json")
    spec = settings["reproduction"]
    sampling = SamplingParams(n=1, **spec[side], max_tokens=spec["max_tokens"],
                              stop=spec["stop"], stop_token_ids=STOP_IDS)
    engine = LLM(str(model_path(shared, side, settings, pairs)), dtype=pairs["dtype"],
                 generation_config="vllm", max_model_len=spec["max_model_len"],
                 gpu_memory_utilization=settings["gpu_memory_utilization"], seed=spec["seed"])
    path = run / f"upstream-{side}.jsonl.gz"
    with publish_raw(path) as draft, gzip.open(draft, "wt", encoding="utf-8") as target:
        for name in spec["datasets"]:
            rows = [prompt for prompt in prompts if prompt["source"] == name]
            output = engine.generate([row["prompt"] for row in rows], sampling)
            for row, sample in zip(rows, output, strict=True):
                require(sample.prompt == row["prompt"] and len(sample.outputs) == 1,
                        "vLLM 輸出未對齊原作 prompt")
                require(len(sample.prompt_token_ids) + spec["max_tokens"] <= spec["max_model_len"],
                        "原作重現的 prompt 無法保留完整生成預算")
                completion = sample.outputs[0]
                require(completion.finish_reason in ("stop", "length"),
                        "vLLM 原作重現輸出未正常回收")
                target.write(json.dumps({
                    "id": row["id"], "texts": [completion.text],
                    "ntok": [len(completion.token_ids)],
                    "finished": [int(completion.finish_reason == "stop")],
                }, ensure_ascii=False) + "\n")
    atomic_json(run / f"upstream-{side}-receipt.json",
                long_receipt(run, side, settings, pairs, shared, commit, job, version("vllm")))


def read_long(run, side, settings, pairs, shared, commit):
    receipt = read_json(run / f"upstream-{side}-receipt.json")
    job = receipt["job_id"]
    require(str(job).isdigit() and receipt == long_receipt(run, side, settings, pairs, shared, commit, job, "0.33.0"),
            f"原作重現憑證不符：{side}")
    with gzip.open(run / f"upstream-{side}.jsonl.gz", "rt", encoding="utf-8") as source:
        rows = {}
        for line in source:
            rec = json.loads(line)
            key = rec.pop("id")
            require(key not in rows, f"原作重現有重複題目：{key}")
            rows[key] = rec
    ids = [row["id"] for row in read_json(run / "upstream-prompts.json")]
    validate_records(rows, ids, 1, settings["reproduction"]["max_tokens"], False, raw=True)
    return rows, receipt


def score_long(run, settings, pairs, criteria, shared, commit):
    cache = shared / "tmp" / settings["upstream_cache"]
    spec = settings["reproduction"]
    result = {"commit": commit, "upstream_commit": settings["upstream_commit"], "models": {}}
    for side in ("base", "rl"):
        rows, receipt = read_long(run, side, settings, pairs, shared, commit)
        score_path = run / f"upstream-{side}-scores.json"
        run_script("pair_b_upstream.py", "score", cache, score_path,
                   "--input", run / f"upstream-{side}.jsonl.gz",
                   "--prompts", run / "upstream-prompts.json")
        require(sha256(run / f"upstream-{side}.jsonl.gz") == receipt["raw_sha256"],
                f"原作取樣檔在計分期間變更：{side}")
        scores = read_json(score_path)
        perf = {}
        for dataset, count in spec["datasets"].items():
            group = [(key, correct, timeout) for key, name, correct, timeout in scores if name == dataset]
            require(len(group) == count and len({row[0] for row in group}) == count,
                    f"原作計分題數不符：{dataset}")
            name = spec["source_names"][dataset]
            correct = sum(row[1] for row in group)
            accuracy = 100.0 * correct / count
            reported = pairs["pairs"]["B"][f"reported_{side}_greedy"][name]
            perf[name] = {
                "n": count, "correct": correct, "pp": accuracy, "ref_pp": reported,
                "difference_pp": accuracy - reported, "grading_timeouts": sum(row[2] for row in group),
                "length_stops": sum(1 - rows[key]["finished"][0] for key, _, _ in group),
                "correct_length_stops": sum(correct and not rows[key]["finished"][0]
                                            for key, correct, _ in group),
            }
        result["models"][side] = {"receipt": receipt, "performance": perf, "scores_sha256": sha256(score_path)}
    gate = criteria["gate0"]
    rl = result["models"]["rl"]["performance"]
    result["reproduce_ok"] = all(abs(rl[name]["difference_pp"]) <= gate["reproduce_tol_pp"] + 1e-12
                                 for name in gate["reproduce_sources"])
    atomic_json(run / "reproduction.json", result)
    print("配對 B 原作重現閘門", "通過" if result["reproduce_ok"] else "未通過")


def score_short(run, pairs, criteria, commit):
    data = load_data("all")
    ids = [row["id"] for row in data if row["split"] == "test"]
    ecfg = criteria["eval"]
    require(len(ids) >= ecfg["test_questions_min"], "短預算題數不足")
    root = run / "short"
    gate = check_gate_minus1(root, "B", pairs)
    require_current_correctness(gate, "base")
    loaded = {side: load_verified(root / "B" / side, "B", side, pairs, ecfg, data, gate)
              for side in ("base", "rl")}
    paired = {side: per_q(graded, ids, ecfg["k_main"]) for side, graded in loaded.items()}
    reproduction = read_json(run / "reproduction.json")
    require(reproduction["commit"] == commit and reproduction["reproduce_ok"],
            "原作重現未通過或來自其他凍結版")
    repro_rl = reproduction["models"]["rl"]["performance"]
    faithful = {name: perf["pp"] for name, perf in repro_rl.items()}
    result = gate0(paired, {}, pairs["pairs"]["B"], criteria, ecfg, data, reproduction=faithful)
    require(result["reproduce_ok"] == reproduction["reproduce_ok"], "重現閘門判讀不一致")
    result["commit"] = commit
    result["pair"] = "B"
    result["reproduction_sha256"] = sha256(run / "reproduction.json")
    result["verified_grade_sha256"] = {
        side: sha256(root / "B" / side / "graded_verified.json") for side in loaded
    }
    result["short_questions"] = len(ids)
    atomic_json(run / "pair_b_gate0.json", result)
    print("配對 B", result["verdict"])


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("stage", choices=["check", "prepare", "correct", "long", "score-long", "short", "score-short"])
    parser.add_argument("shared", type=Path)
    parser.add_argument("run", type=Path, nargs="?")
    parser.add_argument("commit", nargs="?")
    parser.add_argument("job", nargs="?")
    parser.add_argument("side", nargs="?", choices=["base", "rl"])
    args = parser.parse_args()
    settings, pairs, criteria = cfg("pair_b.json"), pairs_cfg(), cfg("criteria.json")
    run, shared, commit = args.run, args.shared, args.commit
    if args.stage == "check":
        check_inputs(shared, settings, pairs)
    elif args.stage == "prepare":
        run_script("pair_b_upstream.py", "prepare", shared / "tmp" / settings["upstream_cache"],
                   run / "upstream-prompts.json")
    elif args.stage == "correct":
        root = run / "short"
        root.mkdir(exist_ok=True)
        base, rl = (model_path(shared, side, settings, pairs) for side in ("base", "rl"))
        log_file = root / f"correctness_B_{args.job}.log"
        with log_file.open("x", encoding="utf-8") as log:
            subprocess.run([sys.executable, "-BI", str(SCRIPT.parent / "tests" / "smoke_cpu.py"),
                            str(base), str(rl), "cuda", "64"], stdout=log, stderr=subprocess.STDOUT, check=True)
        record_correctness(root, "B", str(base), str(rl), args.job)
    elif args.stage == "long":
        sample_long(run, args.side, settings, pairs, shared, commit, args.job)
    elif args.stage == "score-long":
        score_long(run, settings, pairs, criteria, shared, commit)
    elif args.stage == "short":
        reproduction = read_json(run / "reproduction.json")
        require(reproduction["commit"] == commit and reproduction["reproduce_ok"],
                "先完成並通過同一凍結版的原作重現")
        root = run / "short"
        gate = check_gate_minus1(root, "B", pairs)
        require_current_correctness(gate, args.side)
        model = model_path(shared, args.side, settings, pairs)
        require(gate[args.side] == str(model), "短預算的模型與真實 GPU 正確性檢查不符")
        for index in range(settings["short_shards"]):
            run_script("gen_vllm.py", "--model", model, "--out", root / "B" / args.side,
                       "--shard", f"{index}/{settings['short_shards']}")
    else:
        for side in ("base", "rl"):
            run_script("grade.py", run / "short" / "B" / side, "--pair", "B", "--condition", side,
                       "--n-shards", settings["short_shards"], "--workers", 16)
        score_short(run, pairs, criteria, commit)


if __name__ == "__main__":
    main()
