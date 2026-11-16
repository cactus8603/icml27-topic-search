"""[codex] Isolated native paired producer; original FP32 vectors and GPU scalar scores."""
import importlib.util
from pathlib import Path
import subprocess
import sys
from types import SimpleNamespace

ROOT = Path(__file__).resolve().parents[2]
HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT))
from codex.retrieval_matrix_native_p0.evidence import checked, entry, read, sha, write
from codex.retrieval_matrix_paired_p0 import data, metrics, models, qualification as Q

R = Q.private_module("paired_private_native_producer", ROOT/"codex/retrieval_matrix_native_p0/runner.py")
R.data, R.metrics, R.models, R.Q = data, metrics, SimpleNamespace(Siglip2=models.PairedModel, verify=models.verify), Q
_source_gate, _runtime, _index = R.source_gate, R.runtime, R.index


def source_gate(path, plan, commit):
    result = _source_gate(path, plan, commit)
    if subprocess.check_output(["git", "-C", str(ROOT), "status", "--porcelain", "--", str(HERE.relative_to(ROOT))], text=True).strip():
        raise ValueError("Paired P0 requires its own clean, independently reviewed, committed source")
    return result


def runtime(plan, *, gpu):
    actual = _runtime(plan, gpu=gpu)
    import torch
    torch.backends.cudnn.benchmark = True
    actual["paired_backend"] = {"cudnn_benchmark": torch.backends.cudnn.benchmark}
    return actual


def load_plan(path):
    plan = Q.validate_plan(read(path))
    for filename, expected in plan["source_files"].items():
        if sha(ROOT/filename) != expected:
            raise ValueError("Paired frozen source bytes differ: " + filename)
    for value in plan["external_sources"].values():
        checked(value)
    checked(plan["dependencies"])
    return plan


def smoke(plan_path, plan, out, freeze_commit):
    import numpy as np
    import torch
    from PIL import Image
    source_gate(plan_path, plan, freeze_commit)
    environment = runtime(plan, gpu=True)
    identity = R.base_identity(plan_path, plan)
    out.mkdir(parents=True, exist_ok=True)
    target = out/"smoke.json"
    if target.exists():
        return R.require_smoke(plan_path, plan, out)
    try:
        model = models.PairedModel(plan["model_spec"])
        if any(p.device.type != "cuda" or (p.is_floating_point() and p.dtype != torch.float32) for p in model.model.parameters()):
            raise ValueError("Author paired CLIC must remain full FP32 on its assigned GPU")
        path = out/"synthetic-color.jpg"
        Image.new("RGB", (96, 64), (64, 128, 192)).save(path, format="JPEG")
        checks = {}
        for image in (True, False):
            item = str(path) if image else "A synthetic color image."
            first = model.encode([item], image=image)
            repeated = model.encode([item], image=image)
            if (first.shape != (1, plan["model_spec"]["embedding_dim"]) or first.dtype != np.float32
                    or not np.isfinite(first).all() or not np.allclose(np.linalg.norm(first, axis=1), 1., atol=1e-5)):
                raise ValueError("Author paired synthetic encoder output is invalid")
            cosine, error = float(np.sum(first*repeated)), float(np.max(np.abs(first-repeated)))
            passed = cosine >= plan["settings"]["smoke_cosine_min"] and error <= plan["settings"]["smoke_max_abs"]
            checks["images" if image else "texts"] = {"cosine": cosine, "max_abs": error, "passed": passed}
        if not all(row["passed"] for row in checks.values()):
            raise ValueError("Fixed author batch1 synthetic smoke is not repeatable within frozen technical thresholds")
        value = {"state": "PASS", "identity": identity, "freeze_commit": freeze_commit, "runtime": environment,
            "model_signature": model.signature(), "checks": checks, "technical_only": True, "research_metrics": False}
        write(target, value)
        return value
    except Exception as error:
        write(target, {"state": "FAILED", "identity": identity, "freeze_commit": freeze_commit,
            "runtime": environment, "checks": locals().get("checks", {}), "technical_only": True,
            "research_metrics": False, "error_type": type(error).__name__, "error": str(error)})
        raise


def index(plan_path, plan, out, name, freeze_commit, producer):
    import numpy as np
    from codex.retrieval_scaling import runner as checkpoint
    _index(plan_path, plan, out, name, freeze_commit, producer)
    bundle = data.load(plan["benchmarks"][name]["data"])
    folder = out/name/"index"
    original = read(folder/"receipt.json")
    images = np.load(checked(original["arrays"]["images"]), mmap_mode="r", allow_pickle=False)
    texts = np.load(checked(original["arrays"]["texts"]), mmap_mode="r", allow_pickle=False)
    lookup = {row["id"]: index for index, row in enumerate(bundle["labels"])}
    def compute(items):
        result = []
        for item in items:
            index = lookup[item]
            row = bundle["labels"][index]
            start, end = row["text_start"], row["text_start"]+row["text_options"]
            result.append(metrics.score_rows(images[index:index+1], texts[start:end],
                [{**row, "text_start": 0}], device="cuda")[0])
        return np.stack(result)
    score_path = folder/"author-scalar-scores.npy"
    checkpoint.checkpoint(score_path, list(lookup), 1, compute,
        {**original["identity"], "modality": "author_scalar_scores"}, progress=folder/"score-progress.json")
    # Preserve the original embedding receipt unchanged. The complete final
    # receipt adds score/state evidence and binds the earlier receipt by SHA.
    final = {**original, "embedding_receipt": entry(folder/"receipt.json"),
        "author_scores": entry(score_path), "author_scores_checkpoint": entry(score_path.with_suffix(".state.json"))}
    write(folder/"final-receipt.json", final)


def analyze(plan_path, plan, out, name):
    import numpy as np
    folder = out/name/"index"
    receipt_path = folder/"final-receipt.json"
    receipt = read(receipt_path)
    original = read(checked(receipt["embedding_receipt"]))
    for key in ("state", "identity", "arrays", "checkpoints", "counts", "smoke", "research_metrics"):
        if original[key] != receipt[key]:
            raise ValueError("Paired complete receipt changed original embedding provenance")
    bundle = data.load(plan["benchmarks"][name]["data"])
    scores = np.load(checked(receipt["author_scores"]), mmap_mode="r", allow_pickle=False)
    value = {"schema": 2, "state": "COMPLETE", "owner": "[codex]", "producer": receipt["identity"]["producer"],
        "identity": receipt["identity"], "data_identity": bundle["identity"], "counts": bundle["counts"],
        "arrays": receipt["arrays"], "author_scores": receipt["author_scores"], "index_receipt": entry(receipt_path),
        "score_profile": metrics.PROFILE, "primary_metric_descriptors": plan["benchmarks"][name]["primary_metrics"],
        "metrics": metrics.from_scores(scores, bundle["labels"]), "qualification_state": "P0_REQUIRED", "cpu_only": True}
    Q.verify_report(plan, sha(plan_path), name, value)
    write(out/name/"results/report.json", value)
    return value


def audit(plan_path, plan, out):
    from PIL import Image
    from codex.retrieval_matrix_extra import preflight
    import torch
    dependencies = R.dependency_gate(plan)
    models.verify(plan["model_spec"])
    bundles = {name: data.load(row["data"]) for name, row in plan["benchmarks"].items()}
    model, transform, tokenizer = models.parts(plan["model_spec"], weights=False)
    actual = preflight.header(Path(plan["model_spec"]["snapshot"])/plan["model_spec"]["weight_file"])
    coverage = preflight.key_coverage(model, actual)
    tokens = tokenizer(["A synthetic color image.", ""])
    if plan["model_key"] == "clic_clips_l14":
        if (type(tokenizer).__name__ != "CLIPS_Tokenizer" or tokens.shape != (2, 80)
                or not torch.all(tokens[:, 0] == 1) or not torch.all(tokens[:, -1] == 101)):
            raise ValueError("CLIPS actual tokenizer does not preserve source BOS/EOS/trailing class behavior")
    pixels = transform(Image.new("RGB", (96, 64), (64, 128, 192)))
    if tuple(pixels.shape) != (3, 224, 224) or not torch.isfinite(pixels).all():
        raise ValueError("Author image transform failed synthetic CPU check")
    value = {"state": "TECHNICAL_PASS", "p0_state": "NOT_RUN", "owner": "[codex]", "plan": entry(plan_path),
        "source_files": plan["source_files"], "dependencies_checked": len(dependencies["files"]), "coverage": coverage,
        "runtime": runtime(plan, gpu=False), "data": {name: row["identity"] for name, row in bundles.items()},
        "tokenizer_class": type(tokenizer).__module__+"."+type(tokenizer).__name__, "synthetic_token_ids": tokens.tolist(),
        "synthetic_pixel_shape": list(pixels.shape), "research_metrics": False}
    write(out/"cpu-audit.json", value)
    return value


R.source_gate, R.runtime, R.load_plan = source_gate, runtime, load_plan
R.smoke, R.index, R.analyze, R.audit = smoke, index, analyze, audit


if __name__ == "__main__":
    R.main()
