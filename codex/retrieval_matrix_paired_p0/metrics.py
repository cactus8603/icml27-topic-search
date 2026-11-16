"""[codex] CLIC's paired FP32 strict comparisons, with separate CPU recomputation."""
import numpy as np

PROFILE = {"id": "clic-paired-author-fp32-strict-v1", "embedding_dtype": "float32",
    "normalization": "author load_encode_img/load_encode_text F.normalize(dim=-1), FP32",
    "similarity_dtype": "float32", "sugarcrepe_similarity": "torch (positive * image).sum(dim=-1), and negative likewise",
    "sugarcrepe_pp_similarity": "torch.nn.CosineSimilarity(dim=1, eps=1e-6) on the normalized FP32 vectors",
    "ITT": "SugarCrepe p1>negative; SugarCrepe++ image:p1>negative AND image:p2>negative",
    "TOT": "SugarCrepe++ cosine(p1,p2)>cosine(p1,negative) AND >cosine(p2,negative)",
    "ties": "strict greater-than; every equality is incorrect; no sorting or tie-break",
    "aggregation": "per-subset complete-denominator accuracy, then unweighted mean across Add/Replace/Swap subsets",
    "author_scores": "GPU FP32 six columns: image-p1,image-negative,image-p2,p1-p2,p1-negative,p2-negative; unused SugarCrepe columns zero"}


def validate_embeddings(images, texts, labels):
    images, texts = np.asarray(images), np.asarray(texts)
    if (images.ndim != 2 or texts.ndim != 2 or images.shape[1] != texts.shape[1]
            or images.dtype != np.float32 or texts.dtype != np.float32 or len(images) != len(labels)
            or not len(images) or not np.isfinite(images).all() or not np.isfinite(texts).all()):
        raise ValueError("Paired original embeddings must be aligned finite FP32 arrays")
    end = 0
    for row in labels:
        expected = 3 if row["benchmark"] == "sugarcrepe_pp" else 2
        if row["text_start"] != end or row["text_options"] != expected:
            raise ValueError("Paired roles do not cover the complete original caption order")
        end += expected
    if end != len(texts):
        raise ValueError("Paired captions were dropped or supplemented")
    for value in (images, texts):
        if not np.allclose(np.linalg.norm(value, axis=1), 1., rtol=1e-4, atol=1e-5):
            raise ValueError("Paired embeddings do not preserve the author's FP32 normalization")
    return images, texts, labels


def score_rows(images, texts, labels, *, device="cpu"):
    """Execute the exact author scalar formulas at batch1, without GEMM."""
    import torch
    images, texts, labels = validate_embeddings(images, texts, labels)
    output = []
    with torch.inference_mode():
        for index, row in enumerate(labels):
            image = torch.tensor(images[index:index+1].copy(), device=device, dtype=torch.float32)
            values = torch.tensor(texts[row["text_start"]:row["text_start"]+row["text_options"]].copy(), device=device, dtype=torch.float32)
            if row["benchmark"] == "sugarcrepe":
                output.append([(values[0:1]*image).sum(dim=-1).item(), (values[1:2]*image).sum(dim=-1).item(), 0., 0., 0., 0.])
            else:
                cosine = torch.nn.CosineSimilarity(dim=1, eps=1e-6)
                p1, p2, negative = values[0:1], values[1:2], values[2:3]
                output.append([cosine(image, p1).item(), cosine(image, negative).item(), cosine(image, p2).item(),
                    cosine(p1, p2).item(), cosine(p1, negative).item(), cosine(p2, negative).item()])
    return np.asarray(output, dtype=np.float32)


def events(scores, labels):
    scores = np.asarray(scores)
    if scores.shape != (len(labels), 6) or scores.dtype != np.float32 or not np.isfinite(scores).all():
        raise ValueError("Author paired scores must retain all six finite FP32 columns for every original row")
    itt = scores[:, 0] > scores[:, 1]
    tot = np.zeros(len(labels), dtype=bool)
    for index, row in enumerate(labels):
        if row["benchmark"] == "sugarcrepe_pp":
            itt[index] &= scores[index, 2] > scores[index, 1]
            tot[index] = scores[index, 3] > scores[index, 4] and scores[index, 3] > scores[index, 5]
    return np.stack((itt, tot), axis=1)


def from_scores(scores, labels):
    selected = events(scores, labels)
    pp = labels[0]["benchmark"] == "sugarcrepe_pp"
    output = {}
    categories = list(dict.fromkeys(row["category"] for row in labels))
    for category in categories:
        indices = [index for index, row in enumerate(labels) if row["category"] == category]
        output[category+"_ITT"] = float(np.mean(selected[indices, 0]))
        if pp:
            output[category+"_TOT"] = float(np.mean(selected[indices, 1]))
    for group in ("add", "replace", "swap"):
        subsets = [name for name in categories if name.startswith(group+"_")]
        if subsets:
            for metric in (("ITT", "TOT") if pp else ("ITT",)):
                output[group+"_"+metric] = float(np.mean([output[name+"_"+metric] for name in subsets]))
    return output


def independent_metrics(images, texts, labels):
    """Recompute source events from original vectors; never consumes GPU scores.

    Use a separate row loop and torch functional cosine instead of the
    producer's nn module. Strict events must also agree row by row at review.
    """
    import torch
    images, texts, labels = validate_embeddings(images, texts, labels)
    hits = {}
    for index, row in enumerate(labels):
        image = torch.from_numpy(images[index:index+1].copy())
        options = torch.from_numpy(texts[row["text_start"]:row["text_start"]+row["text_options"]].copy())
        if row["benchmark"] == "sugarcrepe":
            good = bool(torch.sum(image*options[0:1]) > torch.sum(image*options[1:2]))
            pair = (good, False)
        else:
            similarity = lambda a, b: torch.nn.functional.cosine_similarity(a, b, dim=1, eps=1e-6).item()
            positive1, positive2, negative = options[0:1], options[1:2], options[2:3]
            threshold = similarity(image, negative)
            pair = (similarity(image, positive1) > threshold and similarity(image, positive2) > threshold,
                similarity(positive1, positive2) > similarity(positive1, negative)
                and similarity(positive1, positive2) > similarity(positive2, negative))
        hits.setdefault(row["category"], []).append(pair)
    output = {}
    pp = labels[0]["benchmark"] == "sugarcrepe_pp"
    for category, values in hits.items():
        output[category+"_ITT"] = sum(row[0] for row in values)/len(values)
        if pp:
            output[category+"_TOT"] = sum(row[1] for row in values)/len(values)
    for group in ("add", "replace", "swap"):
        names = [name for name in hits if name.startswith(group+"_")]
        if names:
            for metric in (("ITT", "TOT") if pp else ("ITT",)):
                output[group+"_"+metric] = sum(output[name+"_"+metric] for name in names)/len(names)
    return output
