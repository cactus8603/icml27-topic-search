"""報告 60 §7 補量。--preflight 檢查資料及離線權重，--model 評估個別模型。"""
import argparse
import hashlib
import io
import json
import os
import sys
from pathlib import Path
from tempfile import NamedTemporaryFile

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
from common import COCO_DIR, EXP, EXTRA_SITE, HF_HOME, bootstrap_ci, cor_flags, jdump, params
from e2_analyze import delta_err
from e2_build_queries import joint_text
from headroom_models import ClipEncoder, CrossScorer, Encoder, unit_scores

# Prefer this runtime's matching torch/torchvision pair. The legacy supplement supplies OpenCLIP only.
if str(EXTRA_SITE) in sys.path:
    sys.path.remove(str(EXTRA_SITE))
    sys.path.append(str(EXTRA_SITE))


CONFIG = EXP / "configs" / "params_headroom.json"
FIELDS = ("full_truth", "anchor", "half_truth")


def snapshot(repo, revision, *, dataset=False):
    if revision is None:
        raise FileNotFoundError(f"{repo} 尚無凍結 revision")
    prefix = "datasets" if dataset else "models"
    return Path(HF_HOME) / "hub" / f"{prefix}--{repo.replace('/', '--')}" / "snapshots" / revision


def model_dir(spec):
    directory = snapshot(spec["repo"], spec["revision"])
    missing = [name for name in spec["files"] if not (directory / name).is_file()]
    if missing:
        raise FileNotFoundError(f"{spec['repo']} 缺少快取檔案：{', '.join(missing)}")
    return directory


def data_path(config, subset):
    return snapshot(config["repo"], config["revision"], dataset=True) / "data" / f"{subset}.parquet"


def load_rows(config, subset):
    import pyarrow.parquet as pq

    columns = ["id", "image_sha256", "image", *FIELDS]
    rows = pq.read_table(data_path(config, subset), columns=columns).to_pylist()
    if len(rows) != config["subset_sizes"][subset]:
        raise ValueError(f"Half-Truths {subset} 列數與凍結設定不符")
    return rows


def rgb(source):
    from PIL import Image

    with Image.open(source) as image:
        return image.convert("RGB")


def encode_items(encoder, images, texts, config):
    image_vectors = unit_scores(encoder, images, config["batch_size"]["embed_image"], images=True)
    text_vectors = unit_scores(encoder, texts, config["batch_size"]["embed_text"])
    return image_vectors, text_vectors


def cross_scores(scorer, queries, images, batch_size):
    scores = []
    for start in range(0, len(queries), batch_size):
        end = start + batch_size
        scores.extend(scorer.score([(query, image()) for query, image in zip(queries[start:end], images[start:end])]))
    return np.asarray(scores, dtype=float)


def triplet_metrics(scores, boot):
    if scores.ndim != 2 or scores.shape[1] != len(FIELDS) or not np.isfinite(scores).all():
        raise ValueError("COR 需要每列三個有限分數")
    correct = cor_flags(*scores.T)
    return {"cor": bootstrap_ci(correct, boot["n_boot"], boot["boot_seed"]),
            "pairwise_accuracy_pp": float(np.mean(scores[:, 0] > scores[:, 2]) * 100),
            "ties": int(np.sum((scores[:, 0] == scores[:, 1]) | (scores[:, 1] == scores[:, 2])))}


def half_truths(model, name, rows, subset, config):
    if name == "cs_clip" and model is None:
        path = EXP / "results" / "ht" / f"cs_clip_{subset}.npz"
        with np.load(path, allow_pickle=False) as previous:
            if (not np.array_equal(previous["ids"], [row["id"] for row in rows]) or
                    not np.array_equal(previous["sha"], [row["image_sha256"] for row in rows])):
                raise ValueError(f"{path} 與凍結資料列序不符")
            return np.stack([np.einsum("nd,nd->n", previous["img"], previous[field])
                             for field in ("full", "anchor", "half")], axis=1)
    captions = [row[field] for row in rows for field in FIELDS]
    images = [lambda row=row: rgb(io.BytesIO(row["image"]["bytes"])) for row in rows]
    if name in ("core_reranker", "vlm_fallback"):
        return cross_scores(model, captions, [image for image in images for _ in FIELDS],
                            config["batch_size"]["cross_pair"]).reshape(len(rows), len(FIELDS))
    iv, tv = encode_items(model, images, captions, config)
    return np.einsum("nd,nkd->nk", iv, tv.reshape(len(rows), len(FIELDS), -1))


def e2_stats(scores, queries, criteria, boot, recall):
    report = {}
    for k, score in scores.items():
        examples = queries["queries"][str(k)]
        ref = [q["istar"] for q in examples]
        if not np.isfinite(score).all():
            raise ValueError("E2 需要有限分數")
        if recall:
            foil = [q["j"] for q in examples]
            err = score[np.arange(len(ref)), ref] <= score[np.arange(len(ref)), foil]
            at50 = [np.isin(np.argsort(-s, kind="stable")[:50], q["rel"]).sum() / min(50, len(q["rel"]))
                    for s, q in zip(score, examples)]
        else:
            err = score[:, 0] <= score[:, 1]
            at50 = None
        report[str(k)] = {"pair_accuracy_pp": float((1 - err.mean()) * 100),
                          "pair_error": bootstrap_ci(err, boot["n_boot"], boot["boot_seed"] + k, groups=ref),
                          "err_flags": err.astype(int).tolist(), "istar": ref,
                          "recall_at_50_pp": None if at50 is None else float(np.mean(at50) * 100)}
    return {"per_k": report,
            "dilution_pp": delta_err(report, criteria["dilution_high"], criteria["dilution_low"],
                                     boot["n_boot"], boot["boot_seed"])}


def evaluate_e2(model, name, path, config, device):
    qset = json.loads(path.read_text())
    upstream = params()
    if qset["seed"] != upstream["p2"]["seed"]:
        raise ValueError("E2 必須使用原 `main` seed")
    images = [lambda i=i: rgb(COCO_DIR / "val2017" / f"{i:012d}.jpg") for i in qset["image_ids"]]
    score = {}
    cross = name in ("core_reranker", "vlm_fallback")
    cached_e2 = EXP / "results/e2/emb_cs_clip.npz"
    if name == "cs_clip" and cached_e2.is_file():
        with np.load(cached_e2, allow_pickle=False) as cached:
            if not np.array_equal(cached["image_ids"], qset["image_ids"]):
                raise ValueError("CS-CLIP E2 影像列序不符")
            iv = cached["img"]
            for k in config["e2"]["k"]:
                selection = cached["owner_main"][:, 0] == k
                order = cached["owner_main"][selection, 1]
                if not np.array_equal(order, np.arange(len(qset["queries"][str(k)]))):
                    raise ValueError("CS-CLIP E2 的既有 main 查詢列序不符")
                score[k] = cached["joint_main"][selection] @ iv.T
    else:
        if name == "cs_clip":
            model = ClipEncoder(name, model_dir(config["models"][name]), config, device)
        iv = None if cross else unit_scores(model, images, config["batch_size"]["embed_image"], images=True)
        for k in config["e2"]["k"]:
            examples = qset["queries"][str(k)]
            texts = [joint_text(q["cat_names"]) for q in examples]
            if cross:
                pairs = [images[q[key]] for q in examples for key in ("istar", "j")]
                score[k] = cross_scores(model, [t for t in texts for _ in (0, 1)], pairs,
                                        config["batch_size"]["cross_pair"]).reshape(len(examples), 2)
            else:
                score[k] = unit_scores(model, texts, config["batch_size"]["embed_text"]) @ iv.T
    return e2_stats(score, qset, config["e2"], upstream["gate_g3"], not cross)


def decide(scores, fixed_names, reranker, threshold):
    observed = {name: scores[name] for name in fixed_names if name in scores}
    best = max(observed.values(), default=None)
    missing = [name for name in fixed_names if name not in observed]
    out = {"fixed_observed_pp": observed, "missing_fixed": missing, "B_fix_pp": None if missing else best,
           "B_fix_lower_bound_pp": best, "R_pp": scores.get(reranker), "reranker": reranker}
    if best is not None and best >= threshold["fixed_fail_at"]:
        return {**out, "state": "FAIL", "reason": "固定索引已達門檻，在本 PREREG 範圍內不進候選池"}
    if missing:
        return {**out, "state": "未能測試", "reason": "必算固定索引模型未全部評估"}
    if best < threshold["fixed_pass_below"]:
        scoped = out["R_pp"] is not None and out["R_pp"] >= threshold["reranker_scope_at"]
        return {**out, "state": "PASS", "restriction": "限固定索引／單向量情境" if scoped else None,
                "scope_unmeasured": out["R_pp"] is None}
    return {**out, "state": "灰區", "reason": "交由使用者判斷"}


def aggregate(directory, config, reranker, attempt_id):
    predictions = {}
    rejected = {}
    completed = required_models(config, reranker)
    fixed = completed[:-1]
    for model in completed:
        path = directory / f"{model}.json"
        if not path.exists():
            continue
        try:
            data = json.loads(path.read_text())
            if (data["spec"] != config["models"][model] or data["dataset_revision"] != config["data"]["revision"]
                    or data.get("config_sha256") != config_digest(config) or data.get("attempt_id") != attempt_id):
                raise ValueError("來源不符本次嘗試或凍結設定")
            subset = config["primary_subset"]
            if subset in data["half_truths"]:
                stat = data["half_truths"][subset]["cor"]
                if stat["n"] != config["data"]["subset_sizes"][subset] or not 0 <= stat["point"] <= 100:
                    raise ValueError("主子集列數或 COR 不符")
                predictions[model] = stat["point"]
        except (OSError, ValueError, TypeError, KeyError) as exc:
            rejected[model] = str(exc)
    return {**decide(predictions, fixed, reranker, config["thresholds_pp"]),
            "attempt_id": attempt_id, "rejected_inputs": rejected}


def config_digest(config):
    return hashlib.sha256(json.dumps(config, sort_keys=True).encode()).hexdigest()


def required_models(config, reranker):
    optional = ["clic"] if config["models"]["clic"]["revision"] else []
    return config["fixed"] + optional + [reranker]


def stage_models(config, reranker):
    from huggingface_hub import snapshot_download

    for name in required_models(config, reranker):
        spec = config["models"][name]
        try:
            directory = model_dir(spec)
        except FileNotFoundError:
            directory = snapshot_download(repo_id=spec["repo"], revision=spec["revision"],
                                          cache_dir=Path(HF_HOME) / "hub",
                                          allow_patterns=spec["files"] + ["*.json", "merges.txt", "vocab.txt"])
            model_dir(spec)
        print(name, directory)


def preflight(config, reranker, *, report_only=False):
    import importlib.metadata as metadata

    for package in ("torch", "torchvision", "transformers", "pillow", "pyarrow", "sentencepiece"):
        try:
            print(package, metadata.version(package))
        except metadata.PackageNotFoundError:
            print(package, "missing")
    errors = []
    for subset in config["data"]["subset_sizes"]:
        path = data_path(config["data"], subset)
        if not path.is_file():
            print(f"{subset} 缺資料：{path}", file=sys.stderr)
            if subset == config["primary_subset"]:
                errors.append(f"{subset} 缺資料：{path}")
        else:
            print(subset, "ready")
        if not (EXP / "results/ht" / f"cs_clip_{subset}.npz").is_file():
            print(f"{subset} 缺凍結的 Stage 1 CS-CLIP 向量", file=sys.stderr)
            if subset == config["primary_subset"] and not report_only:
                errors.append(f"{subset} 缺凍結的 Stage 1 CS-CLIP 向量")
    for name, spec in config["models"].items():
        try:
            print(name, model_dir(spec))
        except FileNotFoundError as exc:
            print(name, str(exc))
            if not report_only and name in required_models(config, reranker):
                errors.append(str(exc))
    return errors


def e2_preflight(coco=COCO_DIR):
    errors = []
    if not (coco / "annotations_trainval2017.zip").is_file():
        errors.append("E2 缺 COCO val2017 標註 ZIP")
    if sum(1 for p in (coco / "val2017").glob("*.jpg") if p.is_file()) != 5000:
        errors.append("E2 的 COCO val2017 未備齊 5,000 張 JPEG")
    return errors


def save_model_subsets(model, name, config, path, attempt_id):
    primary = config["primary_subset"]
    subsets = [primary, *[sub for sub in config["data"]["subset_sizes"] if sub != primary]]
    boot = params()["gate_g3"]
    results = {}
    record = {"spec": config["models"][name], "dataset_revision": config["data"]["revision"],
              "config_sha256": config_digest(config), "attempt_id": attempt_id, "half_truths": results}
    for subset in subsets:
        print(f"{name} Half-Truths {subset}", flush=True)
        rows = load_rows(config["data"], subset)
        score = half_truths(model, name, rows, subset, config)
        results[subset] = triplet_metrics(score, boot)
        # A process interrupted during write leaves the last complete subset readable.
        with NamedTemporaryFile(mode="w", encoding="utf-8", dir=path.parent, prefix=f".{path.name}.", delete=False) as fh:
            draft = Path(fh.name)
            try:
                json.dump(record, fh, indent=2, ensure_ascii=False)
                fh.write("\n")
                fh.close()
                if subset == primary:
                    os.link(draft, path)
                else:
                    draft.replace(path)
            finally:
                draft.unlink(missing_ok=True)


def smoke(model, cross):
    from PIL import Image
    from time import perf_counter

    images = [Image.new("RGB", dims, color) for dims, color in (((96, 128), "red"), ((128, 96), "blue"))]
    texts = ["A red square", "A solid blue rectangle"]
    start = perf_counter()
    if cross:
        scores = model.score(list(zip(texts, images)))
        if scores.shape != (2,) or not np.isfinite(scores).all():
            raise ValueError("雙樣本 cross-scoring smoke failed")
    else:
        image_vecs = model.encode(images, images=True)
        text_vecs = model.encode(texts)
        if image_vecs.shape != text_vecs.shape or not (np.isfinite(image_vecs).all() and np.isfinite(text_vecs).all()):
            raise ValueError("雙樣本向量 smoke failed")
    return perf_counter() - start


def main():
    config = json.loads(CONFIG.read_text())
    parser = argparse.ArgumentParser(description=__doc__)
    mode = parser.add_mutually_exclusive_group(required=True)
    mode.add_argument("--preflight", action="store_true")
    mode.add_argument("--e2-preflight", action="store_true")
    mode.add_argument("--stage-models", action="store_true", help="在可連網的準備主機下載凍結 revision 的模型")
    mode.add_argument("--model", choices=list(config["models"]))
    mode.add_argument("--aggregate", action="store_true")
    mode.add_argument("--e2-model", choices=list(config["models"]))
    mode.add_argument("--smoke", choices=[name for name in config["models"] if name != "cs_clip"])
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--out-dir", type=Path, default=EXP / "results/headroom")
    parser.add_argument("--attempt-id", help="本次嘗試 ID，與輸出目錄名稱一致")
    parser.add_argument("--report-availability", action="store_true", help="正式已送出後，僅報告每個模型是否就位")
    parser.add_argument("--e2-queries", type=Path, default=EXP / "results/e2/queries_main.json")
    parser.add_argument("--reranker", choices=("core_reranker", "vlm_fallback"), default="core_reranker")
    args = parser.parse_args()
    if args.preflight:
        errors = preflight(config, args.reranker, report_only=args.report_availability)
        if errors:
            parser.exit(2, "補量離線預檢未通過：\n" + "\n".join(errors) + "\n")
        return
    if args.e2_preflight:
        errors = e2_preflight()
        if errors:
            parser.exit(2, "\n".join(errors) + "\n")
        return
    if args.stage_models:
        if not os.environ.get("HF_HOME"):
            parser.error("請先將 HF_HOME 指到共用模型快取")
        stage_models(config, args.reranker)
        return
    if not args.smoke and (not args.attempt_id or args.out_dir.name != args.attempt_id or not args.out_dir.is_dir()):
        parser.error("需要已獨佔建立的嘗試目錄，且 --attempt-id 與目錄名稱一致")
    if args.aggregate:
        if (args.out_dir / "verdict_headroom.json").exists():
            parser.error("補量判讀已存在，先人工核對執行紀錄")
        jdump(aggregate(args.out_dir, config, args.reranker, args.attempt_id), args.out_dir / "verdict_headroom.json")
        return

    name = args.model or args.e2_model or args.smoke
    spec = config["models"][name]
    if name == "clic":
        cls = ClipEncoder
    else:
        cls = CrossScorer if name in ("core_reranker", "vlm_fallback") else Encoder
    model = None if name == "cs_clip" else cls(name, model_dir(spec), config, args.device)
    if args.smoke:
        seconds = smoke(model, cls is CrossScorer)
        print(f"{name} synthetic smoke passed ({seconds:.2f}s)")
        return
    path = args.out_dir / f"{name}.json"
    if args.e2_model:
        try:
            result = evaluate_e2(model, name, args.e2_queries, config, args.device)
        except (OSError, ValueError, KeyError, RuntimeError) as exc:
            print(f"{name} E2 未能測試：{exc}", file=sys.stderr)
            result = {"state": "未能測試", "reason": str(exc)}
        jdump({**result, "attempt_id": args.attempt_id}, args.out_dir / f"e2_{name}.json")
        return
    if path.exists():
        raise FileExistsError(f"已存在 {path}，先人工核對補量執行紀錄")
    save_model_subsets(model, name, config, path, args.attempt_id)


if __name__ == "__main__":
    main()
