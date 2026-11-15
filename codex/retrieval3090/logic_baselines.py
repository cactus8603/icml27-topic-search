"""NumPy-only AND baselines; never loads images, labels, or result files.

The LCSE formula and numerical clipping follow the author's reference scorer:
https://github.com/SultanMo/factored-vlm/blob/8af1972ddfd03e8437d5f5c96d5a784e94b82d99/eval.py
Its four-template concept encoding must happen upstream. Unlike the reference's
Torch float32 power mean, these score calculations use NumPy float64.
"""

import hashlib
import json
import re

import numpy as np


LCSE_REFERENCE = {
    "paper": "https://arxiv.org/html/2607.23052",
    "code": "https://github.com/SultanMo/factored-vlm/blob/8af1972ddfd03e8437d5f5c96d5a784e94b82d99/eval.py",
    "commit": "8af1972ddfd03e8437d5f5c96d5a784e94b82d99",
    "blob": "b5aa7db8201538cc0997ed7ea4e01812006e5965",
}
CONCEPT_TEMPLATES = ("a {}", "a photo of a {}", "an image of a {}", "a {} in the image")
PAPER_CLIP_DEFAULTS = {"mu": 0.22, "beta": 30.0, "gamma": -1.0}
DEFAULT_MU_GRID = tuple(float(x) for x in np.linspace(-1.0, 1.0, 81))
DEFAULT_BETA_GRID = (1.0, 2.0, 5.0, 10.0, 20.0, 30.0, 50.0, 60.0, 100.0)
CALIBRATION_SCOPE = "coco-train2017-calibration-only"
EPS = 1e-6


def _finite_array(value, name):
    value = np.asarray(value, dtype=np.float64)
    if value.size == 0 or not np.isfinite(value).all():
        raise ValueError(f"{name} must be nonempty and finite")
    return value


def _params(mu, beta, gamma=-1.0):
    try:
        mu, beta, gamma = float(mu), float(beta), float(gamma)
    except (TypeError, ValueError) as error:
        raise ValueError("mu, beta and gamma must be numeric scalars") from error
    if not np.isfinite([mu, beta, gamma]).all() or beta <= 0 or gamma >= 0:
        raise ValueError("mu must be finite; beta > 0 and AND gamma < 0")
    return float(mu), float(beta), float(gamma)


def _inputs(holistic, condition_cosines):
    holistic = _finite_array(holistic, "holistic")
    conditions = _finite_array(condition_cosines, "condition_cosines")
    if conditions.ndim < 1 or conditions.shape[-1] == 0 or conditions.shape[:-1] != holistic.shape:
        raise ValueError("condition_cosines shape must be holistic.shape + (conditions,)")
    if np.any(np.abs(holistic) > 1 + 1e-5) or np.any(np.abs(conditions) > 1 + 1e-5):
        raise ValueError("expected raw normalized cosine similarities, not z-scores or logits")
    return holistic, conditions


def calibrated_presence(condition_cosines, *, mu=0.22, beta=30.0):
    """Sigmoid transform of RAW cosines, not empirical proof of calibration."""
    mu, beta, _ = _params(mu, beta)
    conditions = _finite_array(condition_cosines, "condition_cosines")
    return np.exp(-np.logaddexp(0.0, -beta * (conditions - mu)))


def _logit(probabilities):
    p = np.clip(probabilities, EPS, 1.0 - EPS)
    return np.log(p) - np.log1p(-p)


def _power_mean(probabilities, gamma):
    p = np.clip(probabilities, EPS, 1.0 - EPS)
    # Compute the power mean in log space, avoiding overflow for large |gamma|.
    terms = gamma * np.log(p)
    maximum = terms.max(axis=-1, keepdims=True)
    log_mean = maximum[..., 0] + np.log(np.exp(terms - maximum).mean(axis=-1))
    return np.exp(log_mean / gamma)


def lcse_and(holistic, condition_cosines, *, mu=0.22, beta=30.0, gamma=-1.0):
    """LCSE AND, including the arithmetic-mean-logit reference correction.

    Inputs have shapes (...) and (..., C), with C >= 1. For COCO's current
    object-conjunction setting C=2 and all conditions are positive. The default
    parameters are the author's CLIP defaults, not a calibration for CORE or
    CS-CLIP. Callers should provide a separate train-only calibration artifact.
    """
    mu, beta, gamma = _params(mu, beta, gamma)
    holistic, conditions = _inputs(holistic, condition_cosines)
    probs = calibrated_presence(conditions, mu=mu, beta=beta)
    # The official scorer recovers holistic cosine after clipping its sigmoid.
    base = _logit(calibrated_presence(holistic, mu=mu, beta=beta)) / beta + mu
    logic = _power_mean(probs, gamma) if conditions.shape[-1] > 1 else probs[..., 0]
    return base + (_logit(logic) - _logit(probs.mean(axis=-1))) / beta


def aggregate_scores(holistic, condition_cosines, *, mu=0.22, beta=30.0, gamma=-1.0):
    """Same-backbone controls. All outputs have holistic.shape; higher is better.

    mean_cond/min_cond aggregate raw cosines. product_cond/logsigmoid_cond use
    the same sigmoid probabilities as LCSE. Log-product is NOT an LLR. Product
    and its logarithm have identical mathematical rankings at fixed C; underflow
    can introduce product ties, so logsigmoid is the stable implementation.
    """
    mu, beta, gamma = _params(mu, beta, gamma)
    holistic, conditions = _inputs(holistic, condition_cosines)
    log_product = -np.logaddexp(0.0, -beta * (conditions - mu)).sum(axis=-1)
    return {
        "holistic": holistic.copy(),
        "mean_cond": conditions.mean(axis=-1),
        "min_cond": conditions.min(axis=-1),
        "product_cond": np.exp(log_product),
        "logsigmoid_cond": log_product,
        "lcse_and": lcse_and(holistic, conditions, mu=mu, beta=beta, gamma=gamma),
    }


def _json_sha(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True, separators=(",", ":")).encode()).hexdigest()


def _array_sha(array):
    array = np.ascontiguousarray(array)
    descriptor = json.dumps({"shape": array.shape, "dtype": array.dtype.str}, sort_keys=True).encode()
    return hashlib.sha256(descriptor + b"\n" + array.tobytes()).hexdigest()


def _image_ids(values, name):
    ids = np.asarray(values)
    if ids.ndim != 1 or ids.size == 0 or ids.dtype.kind not in "iu" or (ids <= 0).any():
        raise ValueError(f"{name} must be a nonempty vector of positive integer COCO image IDs")
    if len(np.unique(ids)) != len(ids):
        raise ValueError(f"{name} must contain unique image IDs")
    return [int(x) for x in ids]


def fit_presence_calibration(condition_cosines, condition_labels, *, image_ids,
                             excluded_image_ids, backbone, source_manifest_sha256,
                             source_split="train2017", mu_grid=DEFAULT_MU_GRID,
                             beta_grid=DEFAULT_BETA_GRID):
    """Fit ONLY independent COCO train image/concept presence labels.

    Arrays are (N images, C concepts); labels are binary COCO annotations. All
    dev/held-out/sealed pool image IDs must be passed in excluded_image_ids.
    This objective is held-out Bernoulli NLL, not the paper's multi-operator
    pairwise-accuracy calibration. It implements an explicitly documented
    train-only adaptation. It cannot establish the authenticity of caller data.
    A fixed grid is recorded; ties prefer lower mu, then lower beta.
    """
    if source_split != "train2017":
        raise ValueError("calibration source must be COCO train2017, never dev/val/sealed labels")
    if not isinstance(backbone, str) or not backbone.strip():
        raise ValueError("backbone is required")
    if not isinstance(source_manifest_sha256, str) or not re.fullmatch(r"[0-9a-f]{64}", source_manifest_sha256):
        raise ValueError("source_manifest_sha256 must be a lowercase SHA-256")
    scores = _finite_array(condition_cosines, "calibration cosines")
    labels = np.asarray(condition_labels)
    ids = _image_ids(image_ids, "image_ids")
    excluded = _image_ids(excluded_image_ids, "excluded_image_ids")
    if scores.ndim != 2 or scores.shape != labels.shape or scores.shape[0] != len(ids):
        raise ValueError("calibration arrays must be matching (images, concepts) matrices")
    if not np.isin(labels, [0, 1]).all() or np.any(np.abs(scores) > 1 + 1e-5):
        raise ValueError("calibration needs binary labels and raw cosine scores")
    if set(ids).intersection(excluded):
        raise ValueError("calibration image IDs overlap an evaluation/excluded image pool")
    if not (labels == 0).any() or not (labels == 1).any():
        raise ValueError("calibration requires both positive and negative annotations")
    mus = sorted(set(float(x) for x in mu_grid))
    betas = sorted(set(float(x) for x in beta_grid))
    if not mus or not betas:
        raise ValueError("calibration grids must be nonempty")
    for mu in mus:
        for beta in betas:
            _params(mu, beta)
    best = None
    labels = labels.astype(np.uint8)
    for mu in mus:
        for beta in betas:
            logits = beta * (scores - mu)
            nll = float(np.mean(np.where(labels == 1, np.logaddexp(0, -logits), np.logaddexp(0, logits))))
            if best is None or nll < best[0]:
                best = (nll, mu, beta)
    nll, mu, beta = best
    return {
        "schema": 1,
        "scope": CALIBRATION_SCOPE,
        "reference": LCSE_REFERENCE.copy(),
        "source": {
            "split": source_split,
            "manifest_sha256": source_manifest_sha256,
            "image_ids": ids,
            "image_ids_sha256": _json_sha(ids),
            "excluded_image_ids_sha256": _json_sha(sorted(excluded)),
            "excluded_image_count": len(excluded),
            "condition_labels_sha256": _array_sha(labels),
            "images": len(ids),
            "concepts": scores.shape[1],
            "positive_counts_per_concept": labels.sum(axis=0).astype(int).tolist(),
            "disjointness_checked": True,
        },
        "models": {backbone: {
            "mu": mu, "beta": beta, "gamma": -1.0,
            "parameter_source": "independent-train-presence-nll-grid",
            "objective": "mean Bernoulli NLL over unweighted image-concept pairs",
            "nll": nll,
            "mu_grid": mus, "beta_grid": betas,
            "score_array_sha256": _array_sha(scores),
            "concept_templates": list(CONCEPT_TEMPLATES),
        }},
    }


def validate_calibration_artifact(artifact, backbone, *, excluded_image_ids=None):
    """Validate provenance structure and return {mu, beta, gamma} for scoring.

    An artifact is a provenance record, not proof of label authenticity.
    Supply the actual evaluation IDs to independently recheck non-overlap.
    """
    if artifact.get("schema") != 1 or artifact.get("scope") != CALIBRATION_SCOPE:
        raise ValueError("not a supported independent COCO-train calibration artifact")
    source = artifact.get("source", {})
    if source.get("split") != "train2017" or source.get("disjointness_checked") is not True:
        raise ValueError("train-only source and checked exclusion are required")
    ids = _image_ids(source.get("image_ids", []), "calibration image_ids")
    if source.get("image_ids_sha256") != _json_sha(ids) or source.get("images") != len(ids):
        raise ValueError("calibration image ID fingerprint mismatch")
    for key in ("manifest_sha256", "excluded_image_ids_sha256", "condition_labels_sha256"):
        if not re.fullmatch(r"[0-9a-f]{64}", str(source.get(key, ""))):
            raise ValueError(f"missing or invalid source fingerprint: {key}")
    if not isinstance(source.get("excluded_image_count"), int) or source["excluded_image_count"] <= 0:
        raise ValueError("the excluded evaluation image pool must be nonempty")
    if excluded_image_ids is not None:
        excluded = _image_ids(excluded_image_ids, "excluded_image_ids")
        if set(ids).intersection(excluded):
            raise ValueError("calibration/evaluation image overlap")
    model = artifact.get("models", {}).get(backbone)
    if not isinstance(model, dict) or model.get("parameter_source") != "independent-train-presence-nll-grid":
        raise ValueError("missing independently calibrated backbone parameters")
    if model.get("concept_templates") != list(CONCEPT_TEMPLATES):
        raise ValueError("calibration concept template protocol mismatch")
    if not re.fullmatch(r"[0-9a-f]{64}", str(model.get("score_array_sha256", ""))):
        raise ValueError("missing calibration cosine fingerprint")
    mu, beta, gamma = _params(model.get("mu"), model.get("beta"), model.get("gamma"))
    return {"mu": mu, "beta": beta, "gamma": gamma}
