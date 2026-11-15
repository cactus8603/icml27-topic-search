"""Run the pinned Oat-Zero public evaluation on the two Gate 0(a) panels."""

import argparse
import ast
from collections import Counter
from contextlib import contextmanager, redirect_stdout
import faulthandler
from hashlib import sha256
from importlib import import_module
from importlib.metadata import version
import json
import math
import os
from pathlib import Path
import re
from stat import S_ISFIFO
import subprocess
import sys
import time
from types import SimpleNamespace


HERE = Path(__file__).resolve().parent
ROOT = Path("/work/ChihChungHsu/cactus8603/icml")
ASSETS = ROOT / "tmp/oat-author-grading-dfca49d"
AUTHOR = "dfca49dd460ee7cc8e4a5a162c876a7fd6993b87"
REVISION = "a98e477854071157a450e57dd45fd0684b6fa38a"
MODEL = ROOT / "hf_home/hub/models--sail--Qwen2.5-Math-1.5B-Oat-Zero/snapshots" / REVISION
OUTPUT = ROOT / "runs/rsd-pair-a-author"
GRADE_PYTHON = ROOT / "envs/vllm033/bin/python"
MATH_LIB = ROOT.parent / "icml27-topic-search/killtests/rl-sparse-distill/pylib"
CODE = {"evaluate_model.py": "ea785b1d9ae6bfbc3b7c4a586e7d90389d61c0e42ce3c91fdd68a7648663fa5f",
        "understand_r1_zero/math_grader.py": "eae505bc232c5c2f8495e2bb3b5ab80d4e475502f501864c378d09ce5dab72ee"}
MODEL_METADATA = {
    "config.json": "412c2d0d66ffa6931bdbb997318d96ac37b8b757b8d1caf66bed7681aafcea3d",
    "generation_config.json": "e09953c6cb9802928c320c09ceba567110954b3d8f9166bc673110e8b547ddc7",
    "tokenizer.json": "9c5ae00e602b8860cbd784ba82a8aa14e8feecec692e7076590d014d7b7fdafa",
    "tokenizer_config.json": "6c90413caefc621f6fcfb8d9e5ce8ba9c949f5b65bcf8a4808f994fb9c51cf91",
    "vocab.json": "ca10d7e9fb3ed18575dd1e277a2579c16d108e32f27439684afa0e10b1440910",
    "merges.txt": "8831e4f1a044471340f7c0a83d7bd71306a5b867e95fd870f74d0c5308a904d5",
    "special_tokens_map.json": "6676f091c8bc4d1b50146427cfde92073402866b87b6e39223227931b70083e9",
}
WEIGHT_BLOB = "52c28c6bfbece78cb3e32267a338f89feb29eb8fa36a54f13b8c30c5aefa609c"
SHARED_BLOB = "3ac7bbc7d53ab7216672e462d8d84ed8a2e71c5f1819e85a738db11cfeb5875a"
PANELS = {
    "math": (500, 74.2, "d383d13c807e2904d0db6f8d98496a0574c0a1d51f40331b920c849cf226ef5a"),
    "minerva": (272, 25.7, "23d05849fd365b9dae2ce5f6c4b6af3baecbbd0a88623abec3f0692d346bae2a"),
}
EOS = 151643
LIMIT = 3000
SMOKE_LIMIT = 16
CONTEXT = 4096
TOLERANCE_PP = 2.0
VLLM = "0.8.4"
ENVIRONMENT_SHA = "e1a7dfb35caf99aa4775deb4a9df455091e1f818a6db1fac905e425ca51fa41b"
GRADER_PACKAGES = {"sympy": "1.14.0", "pyarrow": "26.0.1", "math-verify": "0.9.0",
                   "latex2sympy2-extended": "1.11.0", "antlr4-python3-runtime": "4.13.2"}
GRADER_SOURCE = {
    MATH_LIB / "math_verify": "5a582e9a176cd1dee8bf76fe3a48e7089b1b579061f869e9653656d159f52733",
    MATH_LIB / "latex2sympy2_extended": "d649a3be412d0e2579bdd00d4450fbe1f34ff0a0b03c028c6bfae090079514cf",
    MATH_LIB / "antlr4": "b5a71acff22f961c7b9875224d0778b442f0c45bbbbc3e4c3b546e1ea2d58ada",
    ASSETS / "pylatexenc-2.10/pylatexenc": "d5de0a0cd8121efb74d48d5ad475e21cda58ef97a0b02432b3062636a5c6b11d",
}


def require(condition, message):
    if not condition:
        raise ValueError(message)


def progress(stage):
    if os.environ.get("OAT_REPORT_STAGE") == "1":
        os.write(3, f"[codex] Oat-Zero stage={stage}\n".encode())


def digest(path):
    result = sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            result.update(block)
    return result.hexdigest()


def author_arrow(task):
    return ASSETS / "datasets/evaluation_suite" / task / "data-00000-of-00001.arrow"


def source_check(check_panels):
    for name, fingerprint in CODE.items():
        require(digest(ASSETS / name) == fingerprint, f"Author source differs: {name}")
    if check_panels:
        for task, (_, _, fingerprint) in PANELS.items():
            require(digest(author_arrow(task)) == fingerprint, f"Author Arrow file differs: {task}")
    for name, fingerprint in MODEL_METADATA.items():
        require(digest(MODEL / name) == fingerprint, f"Pinned model metadata differs: {name}")
    for name in ("config.json", "generation_config.json"):
        config = json.loads((MODEL / name).read_bytes())
        require(config["eos_token_id"] == EOS, f"Expected native EOS in {name}")
    weight = MODEL / "model.safetensors"
    direct = (MODEL.parents[1] / "blobs").resolve(strict=True) / WEIGHT_BLOB
    deduplicated = (MODEL.parents[2] / "blobs" / SHARED_BLOB[:2]).resolve() / SHARED_BLOB
    require(weight.is_symlink() and weight.readlink() == Path("../../blobs") / WEIGHT_BLOB
            and weight.resolve(strict=True) in {direct, deduplicated} and weight.stat().st_size == 3_087_467_144,
            "Missing pinned RL snapshot weight")
    require(digest(weight) == WEIGHT_BLOB, "RL snapshot weight content differs")
    model_config = json.loads((MODEL / "config.json").read_bytes())
    require(model_config["max_position_embeddings"] == CONTEXT, "Author checkpoint context differs")


def load_author(**bindings):
    evaluate = ASSETS / "evaluate_model.py"
    parsed = ast.parse(evaluate.read_bytes(), filename=str(evaluate))
    names = {"apply_qwen_math_template", "main"}
    definitions = [node for node in parsed.body if isinstance(node, ast.FunctionDef) and node.name in names]
    require({node.name for node in definitions} == names and len(definitions) == len(names), "Unknown author entry points")
    namespace = {"time": time} | bindings
    exec(compile(ast.Module(body=definitions, type_ignores=[]), str(evaluate), "exec"), namespace)
    return namespace["main"]


def load_arrow(export_values):
    import pyarrow as pa

    panels = {}
    for task, (count, _, _) in PANELS.items():
        with pa.memory_map(str(author_arrow(task))) as stream:
            table = pa.ipc.open_stream(stream).read_all()
            require(table.num_rows == count and {"problem", "answer"} <= set(table.column_names),
                    f"Unexpected published Arrow count or schema: {task}")
            if export_values:
                panels[task] = {field: table[field].to_pylist() for field in ("problem", "answer")}
    return panels


def rpc_read(stream):
    line = stream.readline(8 * 1024 * 1024)
    require(line.endswith("\n"), "Author grading process closed or exceeded its pipe record limit")
    return json.loads(line)


def rpc_write(stream, data):
    stream.write(json.dumps(data, allow_nan=False) + "\n")
    stream.flush()


def grader_worker(mode):
    progress("grader_sources")
    source_check(check_panels=mode != "grade-smoke")
    for root, expected in GRADER_SOURCE.items():
        hashed = sha256()
        for path in sorted(root.rglob("*.py")):
            hashed.update(path.relative_to(root).as_posix().encode() + b"\0")
            hashed.update(bytes.fromhex(digest(path)))
        require(hashed.hexdigest() == expected, f"Frozen grader package code differs: {root.name}")
    sys.path[:0] = [str(MATH_LIB), str(ASSETS / "pylatexenc-2.10"), str(ASSETS)]
    versions = {name: version(name) for name in GRADER_PACKAGES}
    require(versions == GRADER_PACKAGES, "Different author grading packages from 26525")
    progress("grader_import")
    import pylatexenc
    from understand_r1_zero.math_grader import boxed_reward_fn

    require(pylatexenc.__version__ == "2.10" and Path(pylatexenc.__file__).resolve().is_relative_to(ASSETS.resolve()),
            "Unexpected author grader pylatexenc package")
    rpc_output = sys.stdout
    with redirect_stdout(sys.stderr):
        info, reward = boxed_reward_fn(r"\boxed{42}", "42", fast=False)
        require(info["formatted"] and reward == 1, "Author scorer failed synthetic known answer")
        panels = load_arrow(mode == "grade-private") if mode != "grade-smoke" else {}
        progress("grader_ready")
        rpc_write(rpc_output, {"panels": panels, "versions": versions | {"pylatexenc": "2.10"}})
        for line in sys.stdin:
            text, gold, fast = json.loads(line)
            info, reward = boxed_reward_fn(text, gold, fast=fast)
            rpc_write(rpc_output, [{"formatted": bool(info["formatted"])}, float(reward)])


@contextmanager
def grading_session(mode):
    stage_fds = (3,) if os.environ.get("OAT_REPORT_STAGE") == "1" else ()
    with subprocess.Popen([str(GRADE_PYTHON), "-IBu", str(__file__), mode], stdin=subprocess.PIPE,
                          stdout=subprocess.PIPE, text=True, encoding="utf-8",
                          env=os.environ | {"CUDA_VISIBLE_DEVICES": ""}, pass_fds=stage_fds) as proc:
        try:
            header = rpc_read(proc.stdout)

            def grader(text, gold, fast):
                rpc_write(proc.stdin, [text, gold, fast])
                return rpc_read(proc.stdout)

            yield header, grader
            proc.stdin.close()
            require(proc.wait(timeout=30) == 0, "The isolated author grader exited with an error")
        finally:
            if proc.poll() is None:
                proc.kill()


def score_results(accuracies):
    require(list(accuracies) == list(PANELS), "Author evaluator reported unexpected panel order or coverage")
    output = {}
    for task, (count, reference, _) in PANELS.items():
        accuracy = float(accuracies[task])
        require(math.isfinite(accuracy) and 0 <= accuracy <= 1 and abs(accuracy * count - round(accuracy * count)) < 1e-8,
                "A greedy panel did not return one binary result per question")
        correct = round(accuracy * count)
        delta = 100 * correct / count - reference
        output[task] = {"n": count, "correct": correct, "accuracy_pct": 100 * correct / count,
                        "reference_pct": reference, "difference_pp": delta,
                        "within_2pp": abs(delta) <= TOLERANCE_PP + 1e-12}
    return {"state": "PASS" if all(row["within_2pp"] for row in output.values()) else "UNTESTABLE",
            "scope": "gate0a_public_author_evaluation_only", "panels": output}


def run_author(model_path, panels, engine, sample, grader, max_tokens=LIMIT, trace_init=False):
    import numpy as np

    calls = iter(panels.items())
    technical = {}
    summaries = []
    decode = {}

    def make_sampling(**kwargs):
        expected = {"n": 1, "temperature": 0, "top_p": 1, "max_tokens": max_tokens, "logprobs": 2}
        require(set(kwargs) == set(expected) | {"seed"} and all(kwargs[key] == val for key, val in expected.items()),
                "Unexpected author sampling arguments")
        decode.update(kwargs)
        return sample(**kwargs)

    def make_model(model_name, **kwargs):
        require(model_name == str(model_path) and kwargs == {"swap_space": 32, "max_model_len": CONTEXT,
                                                             "dtype": "bfloat16", "enable_prefix_caching": True},
                "Unexpected author engine parameters")
        progress("engine_init")
        if trace_init:
            faulthandler.dump_traceback_later(8, repeat=True)
        try:
            llm = engine(model_name, **kwargs)
        finally:
            if trace_init:
                faulthandler.cancel_dump_traceback_later()
        require(llm.get_tokenizer().eos_token_id == EOS, "vLLM tokenizer does not use the published EOS")
        progress("engine_ready")

        def generate(prompts, params):
            task, panel = next(calls)
            require(len(prompts) == len(panel["problem"]), "Wrong published dataset batch size")
            require(not params.stop and not params.stop_token_ids and not params.ignore_eos,
                    "Author route added a stop or ignored model EOS")
            progress(f"generate_{task}")
            outputs = llm.generate(prompts, params)
            counts = Counter()
            for prompt, item in zip(prompts, outputs, strict=True):
                require(prompt == item.prompt and len(item.outputs) == 1, "Misordered or missing greedy output")
                answer = item.outputs[0]
                require(answer.finish_reason in ("stop", "length") and len(answer.token_ids) <= max_tokens,
                        "Incomplete author output or budget exceeded")
                require(answer.stop_reason in (None, EOS), "An additional stop overrode the model EOS")
                counts[answer.finish_reason] += 1
            technical[task] = dict(counts)
            progress(f"score_{task}")
            return outputs

        return SimpleNamespace(generate=generate)

    def capture(*args):
        # The author prints a single unlabeled dictionary with the task accuracies.
        if len(args) == 1 and isinstance(args[0], dict):
            summaries.append(args[0])

    def reader(path):
        require(path == str(ASSETS / "datasets/evaluation_suite"), "Unexpected dataset root")
        return panels

    main = load_author(vllm=SimpleNamespace(LLM=make_model, SamplingParams=make_sampling), np=np,
                       load_from_disk=reader, boxed_reward_fn=grader, print=capture)
    arguments = dict(model_name=str(model_path), tasks=list(panels), dataset_name=str(ASSETS / "datasets/evaluation_suite"))
    if max_tokens != LIMIT:
        arguments["max_tokens"] = max_tokens
    main(**arguments)
    require(next(calls, None) is None and len(summaries) == 1, "Missing single author run summary")
    return summaries[0], technical, decode


def preflight(actual_sampling=None):
    def gen(prompts, _params):
        require(all(text.startswith("<|im_start|>system\nPlease reason step by step") for text in prompts),
                "Author did not use the Qwen-Math template")
        answer = SimpleNamespace(text=r"\boxed{42}", finish_reason="stop", stop_reason=None, token_ids=[EOS])
        return [SimpleNamespace(prompt=text, outputs=[answer]) for text in prompts]

    def stub_sampling(**kwargs):
        return SimpleNamespace(**kwargs, stop=[], stop_token_ids=[], ignore_eos=False)

    stub = SimpleNamespace(generate=gen, get_tokenizer=lambda: SimpleNamespace(eos_token_id=EOS))
    panels = {task: {"problem": ["Synthetic question"], "answer": ["42"]} for task in PANELS}
    with grading_session("grade-preflight") as (header, grader):
        require(not header["panels"], "CPU source preflight must not export panel texts")
        metrics, completion, _ = run_author(MODEL, panels, lambda *a, **kw: stub, actual_sampling or stub_sampling, grader)
    require(all(metrics[task] == 1 and completion[task] == {"stop": 1} for task in PANELS), "Author CPU stub differs")
    return {"mode": "cpu_source_and_stub_only", "author_commit": AUTHOR, "model_revision": REVISION, "eos": EOS,
            "panel_counts": {task: row[0] for task, row in PANELS.items()}, "author_entry_stub": "passed",
            "grading_packages": header["versions"]}


def runtime_check():
    require(sys.version_info[:2] == (3, 10) and version("vllm") == VLLM,
            "The primary route requires the published Python 3.10 / vLLM 0.8.4 environment")
    require(version("transformers") == "4.51.3" and version("numpy") == "1.26.4", "Resolved author runtime differs")
    require(digest(Path(sys.executable).parent.parent / "installed-packages.txt") == ENVIRONMENT_SHA,
            "Resolved author package inventory differs")
    progress("triton_build_import")
    # The V1 GPU worker reaches this import; wheel dependency metadata omits setuptools on Python 3.10.
    import_module("triton.runtime.build")
    require(version("triton") == "3.2.0" and version("setuptools") == "80.9.0", "Resolved Triton build dependencies differ")
    import torch
    import vllm
    from transformers import AutoTokenizer

    require(torch.__version__.split("+")[0] == "2.6.0" and torch.version.cuda == "12.4", "Wrong official vLLM wheel ABI")
    tokenizer = AutoTokenizer.from_pretrained(str(MODEL), local_files_only=True)
    require(tokenizer.eos_token_id == EOS and tokenizer.convert_tokens_to_ids("<|im_end|>") == 151645,
            "Native tokenizer EOS or historical stop identity differs")
    return torch, vllm


def smoke_receipt(smoke_job, freeze):
    require(re.fullmatch(r"[1-9][0-9]*", smoke_job or ""), "Use the successful same-freeze GPU smoke job")
    path = OUTPUT / f"smoke-{smoke_job}.json"
    require(path.is_file() and not path.is_symlink(), "Missing direct GPU smoke receipt")
    data = json.loads(path.read_bytes())
    identity = {"phase": "gpu_smoke", "slurm_job_id": smoke_job, "freeze_commit": freeze,
                "model_snapshot": REVISION, "vllm": VLLM, "installed_packages_inventory_sha256": ENVIRONMENT_SHA,
                "eos_token_id": EOS, "synthetic_requests": 2, "synthetic_token_limit": SMOKE_LIMIT}
    require(all(data.get(key) == value for key, value in identity.items()), "GPU smoke receipt differs from this run")
    return digest(path)


def main():
    cli = argparse.ArgumentParser(description=__doc__)
    cli.add_argument("mode", choices=("preflight", "runtime-preflight", "smoke", "run",
                                      "grade-preflight", "grade-smoke", "grade-private"))
    cli.add_argument("--freeze")
    cli.add_argument("--smoke-job")
    args = cli.parse_args()
    if args.mode.startswith("grade-"):
        pipes = all(S_ISFIFO(os.fstat(stream.fileno()).st_mode) for stream in (sys.stdin, sys.stdout))
        require(pipes and (args.mode != "grade-private" or re.fullmatch(r"[1-9][0-9]*", os.environ.get("SLURM_JOB_ID", ""))),
                "Panel texts may only pass over a private pipe during a recorded job")
        grader_worker(args.mode)
        return
    if args.mode == "preflight":
        print(json.dumps(preflight(), sort_keys=True))
        return

    progress("runtime_check")
    torch, vllm = runtime_check()
    if args.mode == "runtime-preflight":
        print(json.dumps(preflight(vllm.SamplingParams) | {"runtime": VLLM, "torch": torch.__version__,
                                                           "triton": version("triton"), "setuptools": version("setuptools"),
                                                           "mode": "runtime_preflight"}, sort_keys=True))
        return

    job = os.environ.get("SLURM_JOB_ID", "")
    require(re.fullmatch(r"[1-9][0-9]*", job) and re.fullmatch(r"[0-9a-f]{40}", args.freeze or ""),
            "Use a recorded Slurm job and full review freeze")
    smoke = args.mode == "smoke"
    output = OUTPUT / (f"smoke-{job}.json" if smoke else f"{job}.json")
    require(not output.exists() and not output.is_symlink(), "Output must be a new job artifact")
    prior = None if smoke else smoke_receipt(args.smoke_job, args.freeze)
    require(torch.cuda.device_count() == 1 and "H200" in torch.cuda.get_device_name() and torch.cuda.is_bf16_supported(),
            "Expected one NVIDIA H200 with BF16")
    started = time.monotonic()
    with grading_session("grade-smoke" if smoke else "grade-private") as (header, grader):
        if smoke:
            require(not header["panels"], "GPU smoke must not open a published dataset")
            panels = {task: {"problem": [f"What is {i}+1?"], "answer": [str(i + 1)]}
                      for i, task in enumerate(PANELS, 1)}
        else:
            panels = header["panels"]
        require(list(panels) == list(PANELS), "Unexpected author panel order")
        accuracies, completion, decode = run_author(MODEL, panels, vllm.LLM, vllm.SamplingParams, grader,
                                                    max_tokens=SMOKE_LIMIT if smoke else LIMIT, trace_init=smoke)
    report = {"author_commit": AUTHOR, "freeze_commit": args.freeze, "slurm_job_id": job,
            "model_snapshot": REVISION, "author_code_sha256": CODE,
            "weight_blob": WEIGHT_BLOB, "model_metadata_sha256": MODEL_METADATA,
            "python": sys.version.split()[0], "vllm": VLLM, "torch": torch.__version__, "cuda": torch.version.cuda,
            "transformers": version("transformers"), "grading_packages": header["versions"],
            "installed_packages_inventory_sha256": ENVIRONMENT_SHA,
            "eos_token_id": EOS, "author_sampling": decode, "finish_reasons": completion,
            "elapsed_seconds": time.monotonic() - started}
    if smoke:
        report |= {"phase": "gpu_smoke", "synthetic_requests": 2, "synthetic_token_limit": SMOKE_LIMIT}
    else:
        report |= score_results(accuracies) | {"phase": "gate0a", "smoke_job": args.smoke_job, "smoke_receipt_sha256": prior,
                                              "arrow_sha256": {task: row[2] for task, row in PANELS.items()}}
    progress("publish_receipt")
    output.parent.mkdir(parents=True, exist_ok=True)
    temporary = output.with_name(f"{output.name}.partial")
    with temporary.open("x") as stream:
        json.dump(report, stream, sort_keys=True, indent=2, allow_nan=False)
        stream.write("\n")
    try:
        os.link(temporary, output)
    finally:
        temporary.unlink()


if __name__ == "__main__":
    main()
