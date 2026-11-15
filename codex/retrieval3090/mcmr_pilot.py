"""[codex] Fixed-policy MCMR development pilot; never searches policies or trains.

Direct CLI (prepare is CPU only):
  python -B mcmr_pilot.py --data DATA --images DATA/pilot-images --out RUN \
      --policy m1/top200/min_cond_core --stage prepare

Query/candidate selection is the existing hash-fixed 128/8320 pilot. Native
candidate evidence (image + product metadata) is shared by baseline and policy.
CS-CLIP, when required by the frozen policy, remains an extra image-only index.
"""
from __future__ import annotations

import argparse
import ast
from contextlib import contextmanager
from copy import deepcopy
import fcntl
import gc
import hashlib
from importlib.metadata import version, PackageNotFoundError
import json
import os
from pathlib import Path
import re
import subprocess
import sys
from time import monotonic
from types import ModuleType
from urllib.parse import urlparse

import numpy as np

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]
SOURCE = "2daedf522c6d64b860277657d3067b61da3a0f2b"
CORE_CODE = "400cf919838ceeecf4c94e3ac178a4270abcc4c5"
CONFIG_PATH = "killtests/compositional-retrieval/configs/params_headroom.json"
MODEL_PATH = "killtests/compositional-retrieval/scripts/headroom_models.py"
COMMON_PATH = "killtests/compositional-retrieval/scripts/common.py"
CONFIG_SHA = "862591c60c5609f16727d902993df0cba9cab96ded72611d69d263a02fe4ad1d"
MCMR_REV = "12d53f4063804ee6a976024c7375ab2fbe4368c8"
RAW_SHA = {"query.jsonl": "28925c2d015a75fda763d7bd3092bfd01bf96c8d3e8cd2397702e1fda049e68c",
           "candidates.jsonl": "05242fb414333a99c562a986edb8b2d1997db1726f731b6131ab97bf557fcbc0"}
ARCHIVE_SHA = "7cbda9a5fbdba1240590885c576b4b31dccd047f14d2cc5b8a8e87c94bda2b86"
PILOT_SHA = "3939d973295631c3c843e8aa3c02c4acf847fe8d2ce70fdfbf7779d04d7331ce"
BUDGET = 50
SEED = 2026111664
FORBIDDEN = "retrieval-independent-panel"
SCOPE = "Fixed-COCO-policy external development pilot; 128 queries / 8320 candidates; not full native benchmark reproduction"
CLAUSE_PROTOCOL = {"version": 1, "rule": "split after sentence punctuation before uppercase, or semicolon; keep verbatim spans",
                   "atomic_conditions": False, "split_and_or": False,
                   "coreference": "unresolved; query-only clause diagnostic, not gold logical decomposition"}


def safe(path):
    path = Path(path).expanduser().resolve()
    if FORBIDDEN in str(path):
        raise ValueError("Sealed independent panel is forbidden")
    return path


def canonical(value):
    return json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(",", ":"), allow_nan=False).encode()


def digest(value):
    return hashlib.sha256(canonical(value)).hexdigest()


def sha(path):
    with safe(path).open("rb") as stream:
        return hashlib.file_digest(stream, "sha256").hexdigest()


def read_json(path):
    return json.loads(safe(path).read_text())


def atomic_json(path, value):
    path = safe(path)
    pending = path.with_suffix(path.suffix + ".partial")
    pending.write_text(json.dumps(value, ensure_ascii=False, indent=2, sort_keys=True, allow_nan=False) + "\n")
    pending.replace(path)


def fixed_json(path, value):
    if path.exists():
        if read_json(path) != value:
            raise ValueError(f"Frozen artifact differs: {path}")
    else:
        atomic_json(path, value)


def git_bytes(ref, path):
    return subprocess.check_output(["git", "-C", str(ROOT), "show", f"{ref}:{path}"])


def frozen_config():
    body = git_bytes(SOURCE, CONFIG_PATH)
    config = json.loads(body)
    if hashlib.sha256(body).hexdigest() != CONFIG_SHA:
        raise ValueError("Frozen model configuration SHA changed")
    return config


def clauses(text):
    """Exact, lossless offsets; no sentence-level conjunction rewriting."""
    if not isinstance(text, str) or not text.strip():
        raise ValueError("Query text is empty")
    edges = [0]
    for match in re.finditer(r"(?<=[.!?])\s+(?=[A-Z])|;\s*", text):
        # Keep punctuation and whitespace with the preceding span.
        edges.append(match.end())
    edges.append(len(text))
    spans = [{"start": left, "end": right, "verbatim": text[left:right], "text": text[left:right].strip()}
             for left, right in zip(edges, edges[1:]) if text[left:right].strip()]
    if "".join(row["verbatim"] for row in spans) != text:
        raise ValueError("Clause splitting failed to retain the complete original query")
    return spans


def parse_policy(value, depth=None, calibration=None):
    """IDs use the corrected analysis_v2 naming; no data-dependent selection."""
    match = re.fullmatch(r"m1/top(100|200)/(.+)", value)
    if match:
        explicit = int(match[1])
        if depth is not None and depth != explicit:
            raise ValueError("Policy ID and --depth disagree")
        depth, value = explicit, match[2]
    depth = 200 if depth is None else depth
    if depth not in (100, 200):
        raise ValueError("Only frozen depth 100 or 200 is supported")
    value = value.replace("llr_", "logsigmoid_")
    fusion = re.fullmatch(r"fuse_(.+)_w(0\.25|0\.5|0\.75)", value)
    base, weight = (fusion[1], float(fusion[2])) if fusion else (value, None)
    logic = re.fullmatch(r"logic_(core_embed|cs_clip)_(paperparams_transfer|train_presence_calibrated)_(holistic|mean_cond|min_cond|product_cond|logsigmoid_cond|lcse_and)", base)
    basic = re.fullmatch(r"(core|core_full|cs_full|min_cond_(core|cs)|logsigmoid_(core|cs)_t(0\.5|1|2))", base)
    if not logic and not basic:
        raise ValueError(f"Unsupported frozen policy: {value}")
    if fusion and base not in {"min_cond_core", "min_cond_cs", "logsigmoid_core_t1", "logsigmoid_cs_t1", "cs_full"}:
        raise ValueError("Fusion is limited to the original frozen variants")
    backbone = (logic[1] if logic else "cs_clip" if "cs" in base else "core_embed")
    result = {"schema": 1, "policy_id": f"m1/top{depth}/{value}", "score_id": value,
              "base_id": base, "depth": depth, "budget": BUDGET, "backbone": backbone,
              "fusion_weight": weight, "selection_source": "frozen-COCO-development; no MCMR policy search",
              "clause_protocol": CLAUSE_PROTOCOL,
              "needs_conditions": bool(logic or "cond" in base or "logsigmoid" in base),
              "needs_templates": bool(logic), "lcse": None,
              "score_semantics": "logsigmoid is an uncalibrated product heuristic, not an LLR"}
    if logic:
        import logic_baselines as L
        params = dict(L.PAPER_CLIP_DEFAULTS)
        if logic[2] == "train_presence_calibrated":
            if calibration is None:
                raise ValueError("Frozen train-calibrated policy requires --lcse-calibration")
            params = L.validate_calibration_artifact(calibration, backbone)
        result["lcse"] = {"variant": logic[2], "operation": logic[3], "parameters": params,
                          "templates": list(L.CONCEPT_TEMPLATES), "renormalize_template_mean": False,
                          "domain_transfer": "COCO image-presence calibration transferred unchanged; not calibrated on product clauses/joint vectors"}
    elif calibration is not None:
        raise ValueError("Calibration supplied for a policy which does not use it")
    return result


def render_product(row):
    """Preserve native candidate information and explicit price/date field names."""
    fields = []
    for key in ("title", "description", "features", "price", "Date First Available"):
        value = row.get(key)
        if isinstance(value, list):
            value = " ".join(str(part) for part in value if part is not None)
        if value is not None and str(value).strip():
            fields.append(f"{key}: {str(value).strip()}")
    return "\n".join(fields)


def build_panel(pilot, candidate_rows, receipt, images, *, verify_images=True, enforce_counts=True):
    from PIL import Image
    images = safe(images)
    ids = pilot["candidate_ids"]
    queries = pilot["queries"]
    if len(ids) != len(set(ids)) or len({str(q["id"]) for q in queries}) != len(queries):
        raise ValueError("Duplicate candidate/query IDs")
    if enforce_counts and (len(ids), len(queries), pilot.get("seed")) != (8320, 128, SEED):
        raise ValueError("Only the fixed 128/8320 development pilot is allowed")
    if receipt.get("archive_sha256") != ARCHIVE_SHA:
        raise ValueError("Images were not extracted from the pinned official archive")
    index, wanted = {}, set(ids)
    for row in candidate_rows:
        cid = str(row.get("candidate_id", ""))
        if cid not in wanted:
            continue
        if cid in index:
            raise ValueError("Duplicate native candidate ID")
        index[cid] = row
    if set(index) != set(ids):
        raise ValueError("Pilot candidate missing from native metadata")
    candidates = []
    for cid in ids:
        filename = receipt.get("candidate_id_to_filename", {}).get(cid)
        expected = receipt.get("images", {}).get(filename, {})
        if not filename or Path(filename).name != filename or not expected.get("sha256"):
            raise ValueError("Candidate/image receipt alignment missing")
        native_images = index[cid].get("images") or []
        first = native_images[0] if native_images else {}
        native_url = next((first.get(key) for key in ("url", "large", "image") if first.get(key)), "")
        if Path(urlparse(native_url).path).name != filename:
            raise ValueError("Native candidate image URL differs from extraction mapping")
        path = safe(images / filename)
        if path.parent != images or not path.is_file():
            raise ValueError("Candidate image is absent or escapes its image directory")
        if verify_images and (path.stat().st_size != expected["bytes"] or sha(path) != expected["sha256"]):
            raise ValueError("Pilot image differs from verified extraction")
        with Image.open(path) as image:
            width, height = image.size
            image.verify()
        text = render_product(index[cid])
        if not text or min(width, height) <= 0:
            raise ValueError("Candidate has empty metadata or invalid image")
        candidates.append({"id": cid, "image": filename, "sha256": expected["sha256"],
                           "bytes": expected["bytes"], "width": width, "height": height, "text": text})
    prepared = []
    for query in queries:
        positive = query["positive_ids"]
        if not positive or not set(positive) <= set(ids) or len(positive) != len(set(positive)):
            raise ValueError("All required positives must remain in the gallery")
        prepared.append({"id": str(query["id"]), "text": query["text"],
                         "positive_ids": positive, "clauses": clauses(query["text"])})
    return {"schema": 1, "scope": SCOPE, "dataset_revision": MCMR_REV,
            "candidates": candidates, "queries": prepared, "clause_protocol": CLAUSE_PROTOCOL,
            "candidate_text": "complete labeled title/description/features/price/date; no character truncation",
            "candidate_order": "existing fixed pilot manifest order"}


def prepare(args, policy=None):
    data, images, out = safe(args.data), safe(args.images), safe(args.out)
    for name, expected in RAW_SHA.items():
        if sha(data / name) != expected:
            raise ValueError(f"Pinned native MCMR metadata changed: {name}")
    pilot_path = data / "pilot_manifest.json"
    if sha(pilot_path) != PILOT_SHA:
        raise ValueError("Only the original hash-fixed pilot manifest may be used")
    receipt = read_json(images / "extraction.json")
    if receipt.get("pilot_sha256") != sha(pilot_path):
        raise ValueError("Extraction belongs to a different pilot manifest")
    with (data / "candidates.jsonl").open() as stream:
        panel = build_panel(read_json(pilot_path), (json.loads(line) for line in stream if line.strip()), receipt, images)
    # Bind selected query texts/positives to the original public query metadata.
    original = {str(row["qid"]): row for row in map(json.loads, (data / "query.jsonl").read_text().splitlines())}
    for query in panel["queries"]:
        raw = original.get(query["id"], {})
        if raw.get("query") != query["text"] or set(map(str, raw.get("pos_ids", []))) != set(query["positive_ids"]):
            raise ValueError("Pilot query no longer matches native query metadata")
    panel["source_files"] = {name: sha(data / name) for name in (*RAW_SHA, "pilot_manifest.json")}
    panel["source_files"]["extraction.json"] = sha(images / "extraction.json")
    out.mkdir(parents=True, exist_ok=True)
    fixed_json(out / "panel.json", panel)
    if policy is not None:
        fixed_json(out / "policy.json", policy)
    elif (out / "policy.json").exists():
        raise ValueError("A method-independent index directory must not contain a policy")
    atomic_json(out / "prepare-status.json", {"stage": "prepared", "scope": SCOPE,
                "candidates": len(panel["candidates"]), "queries": len(panel["queries"]),
                "panel_sha256": sha(out / "panel.json"),
                "policy_sha256": sha(out / "policy.json") if policy is not None else None,
                "clauses": {str(k): sum(len(q["clauses"]) == k for q in panel["queries"])
                            for k in sorted({len(q["clauses"]) for q in panel["queries"]})}})
    return panel


def frozen_module(name, path):
    module = ModuleType(name)
    module.__file__ = str(ROOT / path)
    exec(compile(git_bytes(SOURCE, path), module.__file__, "exec"), module.__dict__)
    return module


def checkpoint_helpers():
    """Use the exact cache_v2 functions without importing its COCO-only launcher."""
    path = HERE / "cache_v2.py"
    tree = ast.parse(path.read_text(), filename=str(path))
    functions = {"atomic_json", "fingerprint", "file_sha", "batches"}
    selected = [node for node in tree.body if isinstance(node, (ast.Import, ast.ImportFrom))
                and not (isinstance(node, ast.Import) and any(alias.name == "cache_scores" for alias in node.names))
                or isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)) and node.name in functions]
    module = ModuleType("mcmr_shared_checkpoints")
    exec(compile(ast.Module(body=selected, type_ignores=[]), str(path), "exec"), module.__dict__)
    if not all(hasattr(module, name) for name in functions):
        raise ValueError("cache_v2 checkpoint interface changed")
    return module


def script_sources(require_committed=False):
    result = {}
    for filename in ("mcmr_pilot.py", "cache_v2.py", "analysis_v2.py", "logic_baselines.py"):
        path = HERE / filename
        result[filename] = sha(path)
        if require_committed and git_bytes("HEAD", str(path.relative_to(ROOT))) != path.read_bytes():
            raise ValueError(f"GPU stages require committed source: {filename}")
    # The legacy pure-NumPy policy implementation is also part of the adapter.
    import analysis_v2 as V
    result["legacy_analysis"] = sha(V.LEGACY_PATH)
    if require_committed and git_bytes("HEAD", str(V.LEGACY_PATH.relative_to(ROOT))) != V.LEGACY_PATH.read_bytes():
        raise ValueError("Legacy policy implementation has uncommitted changes")
    return result


def snapshots(config, policy=None, *, verify_weights=True, required=None):
    hf = safe(os.environ.get("HF_HOME", ROOT / "hf_home"))
    wanted = required if required is not None else {"core_embed", "core_reranker"} | ({"cs_clip"} if policy["backbone"] == "cs_clip" else set())
    receipt_path = ROOT / "dl/retrieval3090/public-download-receipt.json"
    receipt = read_json(receipt_path) if verify_weights else {}
    paths = {}
    for name in sorted(wanted):
        spec = config["models"][name]
        folder = hf / "hub" / ("models--" + spec["repo"].replace("/", "--")) / "snapshots" / spec["revision"]
        for filename in spec["files"]:
            path = folder / filename
            if not path.is_file():
                raise ValueError(f"Missing frozen model file: {path}")
            if verify_weights:
                row = receipt.get("files", {}).get(str(path), {})
                if (row.get("revision") != spec["revision"] or row.get("repo") != spec["repo"]
                        or row.get("sha256") != sha(path) or row.get("size") != path.stat().st_size):
                    raise ValueError(f"Model file does not match pinned download receipt: {path}")
        paths[name] = folder
    return paths


class JointAdapter:
    """Thin inference-only extension of the frozen original Encoder/CrossScorer."""
    def __init__(self, frozen, name, snapshot, config, device="cuda"):
        self.frozen, self.name = frozen, name
        config = deepcopy(config)
        if name == "core_embed":
            config[name]["max_tokens"] = 4096
            self.inner = frozen.Encoder(name, snapshot, config, device)
        else:
            config[name]["max_tokens"] = 8192
            self.inner = frozen.CrossScorer(name, snapshot, config, device)

    def messages(self, rows):
        f, config = self.frozen, self.inner.config
        if self.name == "core_embed":
            return [f.conversation(text, image, config["image_system"]) for text, image in rows]
        output = []
        for query, text, image in rows:
            message = f.reranker_conversation(query, image, config)
            message[-1]["content"].append({"type": "text", "text": text})
            output.append(message)
        return output

    def run(self, rows):
        import torch
        messages = self.messages(rows)
        inputs = self.frozen.qwen_inputs(self.inner.processor, messages, self.inner.config).to(self.inner.device)
        with torch.inference_mode():
            if self.name == "core_embed":
                hidden = self.inner.model(**inputs, use_cache=False).last_hidden_state[:, -1]
                return torch.nn.functional.normalize(hidden.float(), dim=-1).cpu().numpy()
            logits = self.inner.model(**inputs, use_cache=False, logits_to_keep=1).logits[:, -1].float()
            return (logits[:, self.inner.yes] - logits[:, self.inner.no]).cpu().numpy()


def unit_vectors(array, rows=None, *, mean_templates=False):
    array = np.asarray(array)
    if array.ndim != 2 or (rows is not None and len(array) != rows) or not np.isfinite(array).all():
        raise ValueError("Malformed or nonfinite vector table")
    norm = np.linalg.norm(array, axis=1)
    if ((norm <= 0).any() or (mean_templates and (norm > 1+2e-4).any())
            or (not mean_templates and not np.allclose(norm, 1, atol=2e-4))):
        raise ValueError("Expected unit vectors (four-template means are explicitly unnormalized)")
    return array


def text_table(panel, policy):
    plain = sorted({c["text"] for q in panel["queries"] for c in q["clauses"]})
    if policy["needs_templates"]:
        return plain, [template.format(text) for text in plain for template in policy["lcse"]["templates"]]
    return plain, plain


def admission(panel, vectors, policy):
    """Only cheap vectors and query text enter selection; never positives/cross."""
    import analysis_v2 as V
    import logic_baselines as L
    A = V.A
    n = len(panel["candidates"])
    image = unit_vectors(vectors["core_embed_image"], n)
    full = unit_vectors(vectors["core_embed_full"], len(panel["queries"])) @ image.T
    core = np.argsort(-full, axis=1, kind="stable")
    depth = policy["depth"]
    if n < depth:
        raise ValueError("Gallery is smaller than the fixed admission depth")
    name = policy["backbone"]
    feature_images = unit_vectors(vectors[f"{name}_image"], n)
    holistic = unit_vectors(vectors[f"{name}_full"], len(panel["queries"])) @ feature_images.T
    plain, _ = text_table(panel, policy)
    table = {text: i for i, text in enumerate(plain)}
    condition = None
    if policy["needs_conditions"]:
        condition_vectors = unit_vectors(vectors[f"{name}_condition"], len(plain), mean_templates=policy["needs_templates"])
        condition = condition_vectors @ feature_images.T
    chosen = []
    for qi, query in enumerate(panel["queries"]):
        top = core[qi:qi+1, :depth]
        base = policy["base_id"]
        if base in {"core", "core_full"}:
            scores = -np.arange(depth, dtype=float)[None, :]
        elif base == "cs_full":
            scores = A.zscore(holistic[qi:qi+1])[:, top[0]]
        else:
            raw = condition[[table[c["text"]] for c in query["clauses"]]]
            if policy["lcse"]:
                cond = raw[:, top[0]].T[None, :, :]
                scores = L.aggregate_scores(holistic[qi:qi+1, top[0]], cond,
                                           **policy["lcse"]["parameters"])[policy["lcse"]["operation"]]
            else:
                if (raw.std(axis=1) <= 0).any():
                    raise ValueError("Zero-variance condition scores")
                z = A.zscore(raw)[:, top[0]].T[None, :, :]
                if base.startswith("min_cond_"):
                    scores = z.min(axis=-1)
                else:
                    temperature = float(base.rsplit("_t", 1)[1])
                    scores = A.log_sigmoid(z / temperature).sum(axis=-1)
            if policy["fusion_weight"] is not None:
                w = policy["fusion_weight"]
                scores = (1-w)*A.rank_percentile(-np.arange(depth, dtype=float)[None, :]) + w*A.rank_percentile(scores)
        if base == "cs_full" and policy["fusion_weight"] is not None:
            w = policy["fusion_weight"]
            scores = (1-w)*A.rank_percentile(-np.arange(depth, dtype=float)[None, :]) + w*A.rank_percentile(scores)
        if not np.isfinite(scores).all():
            raise ValueError("Nonfinite fixed-policy admission score")
        chosen.append(A.admit(top, scores, BUDGET)[0])
    return core, np.asarray(chosen)


def cross_union(core, selected):
    if selected.shape != (len(core), BUDGET):
        raise ValueError("Policy must admit exactly 50 candidates per query")
    pairs = []
    for qi, (order, chosen) in enumerate(zip(core, selected, strict=True)):
        if len(set(chosen.tolist())) != BUDGET or not set(chosen) <= set(order):
            raise ValueError("Malformed candidate selection")
        wanted = set(order[:BUDGET]) | set(chosen)
        pairs.extend([[qi, int(cid)] for cid in order if cid in wanted])
    return pairs


def connected_groups(panel):
    parent = list(range(len(panel["queries"])))
    def find(i):
        while parent[i] != i:
            parent[i] = parent[parent[i]]
            i = parent[i]
        return i
    images = {c["id"]: c["sha256"] for c in panel["candidates"]}
    owner = {}
    for i, query in enumerate(panel["queries"]):
        for cid in query["positive_ids"]:
            key = images[cid]
            if key in owner:
                a, b = find(i), find(owner[key])
                parent[max(a, b)] = min(a, b)
            owner[key] = i
    return [find(i) for i in range(len(parent))]


def analyze_arrays(panel, vectors, policy, pairs, cross, *, draws=10000):
    import analysis_v2 as V
    core, selected = admission(panel, vectors, policy)
    expected = cross_union(core, selected)
    if pairs != expected or np.asarray(cross).shape != (len(pairs),) or not np.isfinite(cross).all():
        raise ValueError("Cross scores are not the exact fixed-policy/baseline union")
    lookup = {(qi, ci): float(value) for (qi, ci), value in zip(pairs, cross, strict=True)}
    ids = [c["id"] for c in panel["candidates"]]
    relevance = np.array([[cid in q["positive_ids"] for cid in ids] for q in panel["queries"]], dtype=bool)
    if (relevance.sum(axis=1) == 0).any():
        raise ValueError("Missing required positives")
    output, measured = {}, {}
    for name, prefix in (("core50", core[:, :BUDGET]), ("fixed_policy", selected)):
        scores = np.array([[lookup[(qi, int(ci))] for ci in row] for qi, row in enumerate(prefix)])
        order = V.A.final_order(core, prefix, scores)
        measured[name] = V.measure(order, relevance, BUDGET)
        output[name] = V.summary(measured[name])
    groups = connected_groups(panel)
    paired = V.paired(measured["core50"], measured["fixed_policy"], V.bootstrap_plan(groups, draws=draws, seed=SEED))
    if len(set(groups)) < 2:
        for value in paired.values():
            value["ci95_pp"] = None
            value["bootstrap_se_pp"] = None
            value["ci_half_width_pp"] = None
            value["normal_approx_mde80_pp"] = None
    return {"schema": 1, "scope": SCOPE, "policy": policy, "metrics": output,
            "paired_against_core50": paired, "queries": len(panel["queries"]), "candidates": len(ids),
            "bootstrap": {"seed": SEED, "draws": draws, "clusters": len(set(groups)),
                          "unit": "connected queries sharing positive product ID or identical positive image SHA; otherwise query IID",
                          "limitation": "does not account for unobserved semantic dependence or shared negative products"},
            "cost": {"deployment_pairs_per_arm": BUDGET*len(panel["queries"]),
                     "offline_union_pairs": len(pairs), "cross_used_for_admission": False,
                     "vector_storage_bytes": {key: int(np.asarray(vectors[key]).nbytes) for key in vectors
                                              if key.endswith(("_image", "_full", "_condition"))},
                     "candidate_inputs": {"core_embed": "joint image + complete native metadata",
                                          "cs_clip": "additional image-only index; lacks metadata" if policy["backbone"] == "cs_clip" else "not loaded"},
                     "latency_claim": "equal cross counts only; this pilot does not establish matched end-to-end latency"},
            "interpretation": "Fixed external development transfer; no MCMR tuning, no full-benchmark or independent-confirmation claim"}


@contextmanager
def writer(out):
    out.mkdir(parents=True, exist_ok=True)
    with (out / "writer.lock").open("a+") as stream:
        fcntl.flock(stream, fcntl.LOCK_EX | fcntl.LOCK_NB)
        yield


def environment():
    import torch
    def package(name):
        try:
            return version(name)
        except PackageNotFoundError:
            return None
    if not torch.cuda.is_available():
        raise ValueError("GPU stages require an allocated CUDA device")
    gpu = torch.cuda.get_device_name(0)
    capability = list(torch.cuda.get_device_capability(0))
    if "RTX 3090" not in gpu or capability != [8, 6]:
        raise ValueError("This development index/cache is restricted to RTX 3090 capability 8.6")
    drivers = sorted(set(line.strip() for line in subprocess.check_output(
        ["nvidia-smi", "--query-gpu=driver_version", "--format=csv,noheader"], text=True).splitlines() if line.strip()))
    if len(drivers) != 1:
        raise ValueError("A single verified NVIDIA driver version is required")
    def sdpa_option(name):
        getter = getattr(torch.backends.cuda, name, None)
        return getter() if callable(getter) else None
    return {"python": sys.version, "torch": torch.__version__, "cuda": torch.version.cuda,
            "transformers": package("transformers"), "open_clip_torch": package("open_clip_torch"),
            "torchvision": package("torchvision"), "Pillow": package("Pillow"), "tokenizers": package("tokenizers"),
            "numpy": np.__version__, "device": gpu, "capability": capability,
            "driver": drivers[0], "cudnn": torch.backends.cudnn.version(),
            "attention": "sdpa", "dtype": "bfloat16 CORE; float32 CS-CLIP",
            "matmul_allow_tf32": torch.backends.cuda.matmul.allow_tf32,
            "cudnn_allow_tf32": torch.backends.cudnn.allow_tf32,
            "bf16_reduced_precision_reduction": torch.backends.cuda.matmul.allow_bf16_reduced_precision_reduction,
            "sdpa_backends": {key: sdpa_option(key) for key in (
                "flash_sdp_enabled", "math_sdp_enabled", "mem_efficient_sdp_enabled", "cudnn_sdp_enabled",
                "fp16_bf16_reduction_math_sdp_allowed")},
            "float32_matmul_precision": torch.get_float32_matmul_precision(),
            "cudnn_benchmark": torch.backends.cudnn.benchmark, "cudnn_deterministic": torch.backends.cudnn.deterministic,
            "deterministic_algorithms": torch.are_deterministic_algorithms_enabled(),
            "actual_cpu_threads": {"intraop": torch.get_num_threads(), "interop": torch.get_num_interop_threads()},
            "kernel_environment": {key: os.environ.get(key) for key in (
                "CUBLAS_WORKSPACE_CONFIG", "NVIDIA_TF32_OVERRIDE", "CUDA_MODULE_LOADING", "CUDA_LAUNCH_BLOCKING",
                "CUDA_DEVICE_MAX_CONNECTIONS", "OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS")}}


def placement():
    """Recorded for scheduling/audit; same-model 3090 placements may share an index."""
    import torch
    return {"visible_devices": os.environ.get("CUDA_VISIBLE_DEVICES"),
            "logical_device": 0, "uuid": str(getattr(torch.cuda.get_device_properties(0), "uuid", "unavailable"))}


def processor_metadata(processor):
    image = processor.image_processor
    return {"class": type(processor).__name__, "tokenizer_class": type(processor.tokenizer).__name__,
            "padding_side": processor.tokenizer.padding_side, "image_class": type(image).__name__,
            "image_backend": getattr(image, "backend", None),
            "image_config": json.loads(json.dumps(image.to_dict(), default=str))}


def index_identity(args, panel, config, snapshot, execution_environment, processor, sources):
    """Everything that determines shared forward inputs/outputs, with no policy."""
    return {"schema": "mcmr-core-shared-index-v1", "source": SOURCE, "official_CORE_code": CORE_CODE,
            "sources": sources, "model_config_sha256": CONFIG_SHA, "model_revision": snapshot.name,
            "panel_sha256": digest(panel),
            "candidate_inputs_sha256": digest(panel["candidates"]),
            "query_inputs_sha256": digest([q["text"] for q in panel["queries"]]),
            "candidates": len(panel["candidates"]), "queries": len(panel["queries"]), "embedding_dim": 2048,
            "candidate_system": config["core_embed"]["image_system"],
            "query_system": config["core_embed"]["text_system"], "max_tokens": 4096,
            "image_pixels": {k: v for k, v in config["core_embed"].items() if "pixels" in k},
            "batch_sizes": {"joint_image": args.joint_batch, "text": config["batch_size"]["embed_text"]},
            "processor": processor, "environment": execution_environment}


def read_shared_index(directory, expected):
    directory = safe(directory)
    with (directory / "writer.lock").open("r") as lock:
        fcntl.flock(lock, fcntl.LOCK_SH | fcntl.LOCK_NB)
        if (directory / "policy.json").exists():
            raise ValueError("Shared index must be independent of a selected policy")
        meta = read_json(directory / "index-meta.json")
        if (meta.get("state") != "COMPLETE" or meta.get("identity") != expected
                or meta.get("index_sha256") != sha(directory / "index.npz")):
            raise ValueError("Shared index is incomplete or source/data/model/environment/batch differs")
        with np.load(directory / "index.npz", allow_pickle=False) as arrays:
            if set(arrays.files) != {"core_embed_image", "core_embed_full"}:
                raise ValueError("Shared index may only contain policy-independent CORE vectors")
            vectors = {key: np.array(arrays[key], copy=True) for key in arrays.files}
        image = unit_vectors(vectors["core_embed_image"], expected["candidates"])
        full = unit_vectors(vectors["core_embed_full"], expected["queries"])
        if image.shape[1] != expected["embedding_dim"] or full.shape[1] != expected["embedding_dim"]:
            raise ValueError("Shared index embedding dimensions differ")
        return vectors, {"path": str(directory), "meta_sha256": sha(directory / "index-meta.json"),
                         "index_sha256": meta["index_sha256"], "read_only_reuse": True}


def run_index(args, panel):
    """Encode only immutable candidate + baseline query vectors, without a policy."""
    import torch
    config = frozen_config()
    execution_environment = environment()
    paths = snapshots(config, required={"core_embed"})
    sources = script_sources(require_committed=True)
    frozen = frozen_module("mcmr_frozen_index_models", MODEL_PATH)
    model = JointAdapter(frozen, "core_embed", paths["core_embed"], config)
    expected = index_identity(args, panel, config, paths["core_embed"], execution_environment,
                              processor_metadata(model.inner.processor), sources)
    out, images = safe(args.out), safe(args.images)
    cp = out / "index-checkpoints"
    cp.mkdir(exist_ok=True)
    fixed_json(cp / "provenance.json", expected)
    helper = checkpoint_helpers()
    def stage(name, items, batch, compute):
        return helper.batches(cp / (name+".npy"), items, batch, compute,
                              expected | {"stage": name, "items_sha256": digest(items)}, status=out/"progress.json")
    def joint(rows):
        from PIL import Image
        inputs = []
        for index in rows:
            candidate = panel["candidates"][index]
            with Image.open(images / candidate["image"]) as image:
                inputs.append((candidate["text"], image.convert("RGB")))
        return model.run(inputs)
    began = monotonic()
    image = stage("core_joint", list(range(len(panel["candidates"]))), args.joint_batch, joint)
    full = stage("core_full", [q["text"] for q in panel["queries"]], config["batch_size"]["embed_text"], model.inner.encode)
    unit_vectors(image, len(panel["candidates"]))
    unit_vectors(full, len(panel["queries"]))
    if image.shape[1] != 2048 or full.shape[1] != 2048:
        raise ValueError("Frozen CORE index must have 2048-dimensional vectors")
    del model
    gc.collect(); torch.cuda.empty_cache()
    pending = out / "index.partial.npz"
    np.savez(pending, core_embed_image=image, core_embed_full=full)
    pending.replace(out / "index.npz")
    atomic_json(out / "index-meta.json", {"state": "COMPLETE", "identity": expected,
                "scope": "Method-independent fixed MCMR pilot candidate joint index and baseline full-query vectors; no policy or research metric",
                "index_sha256": sha(out / "index.npz"), "active_wall_seconds": monotonic()-began,
                "placement": placement(),
                "commit": subprocess.check_output(["git", "-C", str(ROOT), "rev-parse", "HEAD"], text=True).strip()})
    atomic_json(out / "progress.json", {"stage": "index_complete", "candidates": len(panel["candidates"]),
                                        "queries": len(panel["queries"]), "research_metrics": False})


def run_gpu(args, panel, policy, *, smoke=False):
    import torch
    frozen = frozen_module("mcmr_frozen_models", MODEL_PATH)
    common = frozen_module("mcmr_frozen_common", COMMON_PATH)
    config = frozen_config()
    paths = snapshots(config, policy)
    sources = script_sources(require_committed=True)
    out, images = safe(args.out), safe(args.images)
    execution_environment = environment()
    shared_vectors, shared_receipt = {}, None
    if getattr(args, "index", None):
        if smoke:
            raise ValueError("Smoke must exercise the adapter directly; do not supply --index")
        from transformers import AutoProcessor
        processor = AutoProcessor.from_pretrained(paths["core_embed"], local_files_only=True, trust_remote_code=False)
        processor.tokenizer.padding_side = "left"
        expected = index_identity(args, panel, config, paths["core_embed"], execution_environment,
                                  processor_metadata(processor), sources)
        shared_vectors, shared_receipt = read_shared_index(args.index, expected)
        del processor
    identity = {"schema": 1, "scope": SCOPE, "source": SOURCE, "official_CORE_code": CORE_CODE,
                "commit": subprocess.check_output(["git", "-C", str(ROOT), "rev-parse", "HEAD"], text=True).strip(),
                "sources": sources, "panel_sha256": sha(out / "panel.json"), "policy_sha256": sha(out / "policy.json"),
                "model_config_sha256": CONFIG_SHA, "model_revisions": {n: p.name for n, p in paths.items()},
                "adapter": {"query_system": config["core_embed"]["text_system"],
                            "candidate_system": config["core_embed"]["image_system"],
                            "prompt_protocol": "COCO-frozen query adapter; differs from official MCMR is_query=False behavior",
                            "joint_tokens": 4096, "cross_tokens": 8192,
                            "image_pixels": {n: {k: v for k, v in config[n].items() if "pixels" in k} for n in ("core_embed", "core_reranker")},
                            "batch_sizes": {"joint_image": args.joint_batch, "cross": args.cross_batch,
                                            "text": config["batch_size"]["embed_text"], "cs_image": config["batch_size"]["embed_image"]},
                            "batch_changes_from_COCO": {"joint_image": [config["batch_size"]["embed_image"], args.joint_batch],
                                                        "cross": [config["batch_size"]["cross_pair"], args.cross_batch]},
                            "CS_candidate_input": "image only; additional cheap index without metadata" if policy["backbone"] == "cs_clip" else None},
                "environment": execution_environment, "shared_index": shared_receipt}
    cp = out / ("smoke-checkpoints" if smoke else "checkpoints")
    cp.mkdir(exist_ok=True)
    fixed_json(cp / "provenance.json", identity)
    helper = checkpoint_helpers()
    def stage(name, items, batch, compute):
        if not items:
            raise ValueError("Cannot encode an empty stage")
        return helper.batches(cp / (name + ".npy"), items, batch, compute,
                              identity | {"stage": name, "items_sha256": digest(items)}, status=out / "progress.json")
    def open_image(index):
        from PIL import Image
        with Image.open(images / panel["candidates"][index]["image"]) as image:
            return image.convert("RGB")
    queries = [q["text"] for q in panel["queries"]]
    plain, condition_texts = text_table(panel, policy)
    index = list(range(min(2, len(panel["candidates"])))) if smoke else list(range(len(panel["candidates"])))
    vectors = shared_vectors
    began = monotonic()
    texts = queries[:2] if smoke else queries
    if not shared_vectors or (policy["backbone"] == "core_embed" and policy["needs_conditions"]):
        model = JointAdapter(frozen, "core_embed", paths["core_embed"], config)
        if not shared_vectors:
            # Each compute call validates total image+text tokens before its forward.
            vectors["core_embed_image"] = stage("core_joint", index, args.joint_batch,
                lambda rows: model.run([(panel["candidates"][i]["text"], open_image(i)) for i in rows]))
            vectors["core_embed_full"] = stage("core_full", texts, config["batch_size"]["embed_text"], model.inner.encode)
        if policy["backbone"] == "core_embed" and policy["needs_conditions"]:
            encoded = stage("core_condition", condition_texts[:2] if smoke else condition_texts,
                            config["batch_size"]["embed_text"], model.inner.encode)
            if not smoke and policy["needs_templates"]:
                encoded = encoded.reshape(len(plain), 4, -1).mean(axis=1)
            vectors["core_embed_condition"] = encoded
        del model
        gc.collect(); torch.cuda.empty_cache()
    if policy["backbone"] == "cs_clip":
        # Frozen ClipEncoder imports common lazily. Avoid mutating a live COCO module.
        previous = sys.modules.get("common")
        sys.modules["common"] = common
        try:
            model = frozen.ClipEncoder("cs_clip", paths["cs_clip"], config, "cuda")
        finally:
            if previous is None:
                sys.modules.pop("common", None)
            else:
                sys.modules["common"] = previous
        token_lengths = None
        if hasattr(model.tokenizer, "encode"):
            token_lengths = {"full_queries_over_77_tokens": sum(len(model.tokenizer.encode(t))+2 > 77 for t in texts),
                             "condition_texts_over_77_tokens": sum(len(model.tokenizer.encode(t))+2 > 77 for t in condition_texts),
                             "rule": "Frozen CS-CLIP tokenizer truncates to 77 including boundary tokens; no product metadata is supplied"}
        fixed_json(cp / "cs-tokenization.json", {"counts": token_lengths,
                   "truncation_protocol": "unchanged frozen open_clip tokenizer; image-only candidate index"})
        vectors["cs_clip_image"] = stage("cs_image", index, config["batch_size"]["embed_image"],
                                         lambda rows: model.encode([open_image(i) for i in rows], images=True))
        vectors["cs_clip_full"] = stage("cs_full", texts, config["batch_size"]["embed_text"], model.encode)
        if policy["needs_conditions"]:
            encoded = stage("cs_condition", condition_texts[:2] if smoke else condition_texts,
                            config["batch_size"]["embed_text"], model.encode)
            if not smoke and policy["needs_templates"]:
                encoded = encoded.reshape(len(plain), 4, -1).mean(axis=1)
            vectors["cs_clip_condition"] = encoded
        del model
        gc.collect(); torch.cuda.empty_cache()
    if smoke:
        pairs = [[0, i] for i in index]
    else:
        core, selected = admission(panel, vectors, policy)
        pairs = cross_union(core, selected)
    fixed_json(cp / "cross-pairs.json", pairs)
    scorer = JointAdapter(frozen, "core_reranker", paths["core_reranker"], config)
    cross = stage("cross_union", pairs, args.cross_batch,
        lambda rows: scorer.run([(queries[qi], panel["candidates"][ci]["text"], open_image(ci)) for qi, ci in rows]))
    del scorer
    gc.collect(); torch.cuda.empty_cache()
    if smoke:
        atomic_json(out / "smoke.json", {"state": "PASS", "research_metrics": False, "pairs": len(pairs),
                                       "unit_vectors": all(unit_vectors(v) is not None for v in vectors.values()),
                                       "identity": identity, "placement": placement()})
        return
    pending = out / "cache.partial.npz"
    np.savez(pending, **vectors, cross=cross, pairs=np.asarray(pairs, dtype=np.int64))
    pending.replace(out / "cache.npz")
    atomic_json(out / "meta.json", {"state": "COMPLETE", "identity": identity,
                                   "placement": placement(),
                                   "cache_sha256": sha(out / "cache.npz"), "offline_union_pairs": len(pairs),
                                   "active_wall_seconds": monotonic()-began,
                                   "cross_admission_dependency": False, "scope": SCOPE})


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ("data", "images", "out"):
        parser.add_argument("--" + name, required=True, type=Path)
    parser.add_argument("--policy")
    parser.add_argument("--depth", type=int, choices=(100, 200))
    parser.add_argument("--lcse-calibration", type=Path)
    parser.add_argument("--stage", required=True, choices=("prepare", "index", "smoke", "cache", "analyze"))
    parser.add_argument("--index", type=Path, help="Read-only reuse of a completed, exactly matching method-independent index")
    parser.add_argument("--joint-batch", type=int, default=1)
    parser.add_argument("--cross-batch", type=int, default=1)
    args = parser.parse_args()
    if not 1 <= args.joint_batch <= 8 or not 1 <= args.cross_batch <= 4:
        parser.error("Batch sizes must be positive and no larger than frozen COCO batches")
    for name in ("data", "images", "out"):
        setattr(args, name, safe(getattr(args, name)))
    if args.index:
        args.index = safe(args.index)
        if args.index == args.out:
            parser.error("Index and policy output must use separate directories")
    if args.stage == "index":
        if args.policy is not None or args.lcse_calibration is not None or args.index is not None or args.depth is not None:
            parser.error("Method-independent index stage must not specify policy/calibration/depth/another index")
    elif args.policy is None:
        parser.error("This stage requires a frozen --policy")
    calibration = read_json(args.lcse_calibration) if args.lcse_calibration else None
    policy = parse_policy(args.policy, args.depth, calibration) if args.policy is not None else None
    if args.out == args.data or args.out == args.images or args.out in args.images.parents or args.out in args.data.parents:
        parser.error("Output must have its own run directory, separate from source metadata/images")
    if args.lcse_calibration:
        policy["calibration_sha256"] = sha(args.lcse_calibration)
    with writer(args.out):
        if args.stage == "index":
            panel = prepare(args)
            run_index(args, panel)
            print(json.dumps({"stage": "index_complete", "out": str(args.out), "research_metrics": False}), flush=True)
            return
        if args.stage == "prepare":
            prepare(args, policy)
            print(json.dumps(read_json(args.out / "prepare-status.json")), flush=True)
            return
        panel = read_json(args.out / "panel.json")
        if read_json(args.out / "policy.json") != policy:
            raise ValueError("Requested policy differs from score-blind prepared policy")
        # Re-verify the public source data and all extracted images before inference.
        if args.stage in ("smoke", "cache"):
            prepare(args, policy)
            run_gpu(args, panel, policy, smoke=args.stage == "smoke")
            return
        meta = read_json(args.out / "meta.json")
        identity = meta.get("identity", {})
        expected_models = frozen_config()["models"]
        required_models = {"core_embed", "core_reranker"} | ({"cs_clip"} if policy["backbone"] == "cs_clip" else set())
        if (meta.get("state") != "COMPLETE" or meta.get("cache_sha256") != sha(args.out / "cache.npz")
                or identity.get("panel_sha256") != sha(args.out / "panel.json")
                or identity.get("policy_sha256") != sha(args.out / "policy.json")
                or identity.get("sources") != script_sources()
                or identity.get("source") != SOURCE or identity.get("model_config_sha256") != CONFIG_SHA
                or identity.get("model_revisions") != {name: expected_models[name]["revision"] for name in required_models}
                or meta.get("cross_admission_dependency") is not False):
            raise ValueError("Cache is incomplete or source/panel/policy integrity differs")
        with np.load(args.out / "cache.npz", allow_pickle=False) as arrays:
            report = analyze_arrays(panel, arrays, policy, arrays["pairs"].tolist(), arrays["cross"])
        report["provenance"] = meta
        atomic_json(args.out / "analysis.json", report)
        print(json.dumps({"stage": "analyzed", "report": str(args.out / "analysis.json"), "scope": SCOPE}), flush=True)


if __name__ == "__main__":
    main()
