"""[codex] Google native distance ranking, without CORE/UniME metric coercion."""
from __future__ import annotations

import importlib.util

from .evidence import checked

PROFILE = {
    "id": "siglip2-google-hf-fp32-bigvision-argsort-v1",
    "embedding_dtype": "float32", "normalization": "torch.nn.functional.normalize(dim=-1) in FP32",
    "similarity_dtype": "float32", "similarity": "numpy.dot(images, texts.T); no temperature or bias",
    "orientation": "images_by_texts", "ranking": "numpy.argsort(-similarity), default quicksort",
    "ties": "author numpy.argsort default quicksort; source-order candidates; no custom stable tie break",
    "metrics": "any-positive Recall@1/5/10, separately over all text and image queries",
    "cpu_score": True,
}


def validate_embeddings(images, texts, positives):
    import numpy as np
    images, texts = np.asarray(images), np.asarray(texts)
    if (images.ndim != 2 or texts.ndim != 2 or images.shape[1] != texts.shape[1]
            or not len(images) or not len(texts) or images.dtype != np.float32 or texts.dtype != np.float32
            or len(positives) != len(texts) or not np.isfinite(images).all() or not np.isfinite(texts).all()):
        raise ValueError("Native scoring requires aligned finite FP32 full-split embeddings")
    positives = np.asarray(positives)
    if (not np.issubdtype(positives.dtype, np.integer) or np.any(positives < 0)
            or np.any(positives >= len(images)) or set(positives.tolist()) != set(range(len(images)))):
        raise ValueError("Every image must retain all its original positive captions")
    for value in (images, texts):
        if not np.allclose(np.linalg.norm(value, axis=1), 1, rtol=1e-4, atol=1e-5):
            raise ValueError("Source embeddings are not normalized in the declared FP32 profile")
    return images, texts, positives


def author_metrics(images, texts, positives, source):
    import numpy as np
    images, texts, positives = validate_embeddings(images, texts, positives)
    path = checked(source)
    spec = importlib.util.spec_from_file_location("native_p0_pinned_google_metric", path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    distance = -np.dot(images, texts.T)
    output = {}
    for direction, function in (("t2i", module.text_to_image_retrieval_eval),
                                ("i2t", module.image_to_text_retrieval_eval)):
        for name, value in function(distance, positives.tolist()).items():
            output[direction + "_R@" + name.split("@")[1]] = float(value)
    return output


def independent_metrics(images, texts, positives):
    """Separate CPU recomputation; never imports/calls the producer scorer.

    Preserve the author's full GEMM shape and default NumPy tie behavior, then
    independently derive each query's rank/event. Saving a BF16 matrix and
    rescoring it here is forbidden; inputs must be original FP32 embeddings.
    """
    import numpy as np
    images, texts, positives = validate_embeddings(images, texts, positives)
    scores = images @ texts.T
    image_rankings = (-scores).argsort(axis=0)
    text_rankings = (-scores).argsort(axis=1)
    output = {}
    for k in (1, 5, 10):
        text_hits = [(image_rankings[:k, index] == positive).any()
                     for index, positive in enumerate(positives)]
        image_hits = [(positives[text_rankings[index, :k]] == index).any()
                      for index in range(len(images))]
        output[f"t2i_R@{k}"] = float(np.mean(text_hits))
        output[f"i2t_R@{k}"] = float(np.mean(image_hits))
    return output
