"""[codex] Build evidence-bound extra-baseline identities, never a qualification result."""
from __future__ import annotations

import hashlib
import json
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[2]
HERE = Path(__file__).resolve().parent
RUNTIME = ROOT/"dl/retrieval-matrix-extra"
sys.path.insert(0, str(ROOT))
from codex.retrieval_matrix_extra.adapters import validate_spec


def sha(path):
    with Path(path).open("rb") as stream:
        return hashlib.file_digest(stream, "sha256").hexdigest()


def source(name):
    root = RUNTIME/"sources"/name
    obj = json.loads((root/"source-receipt.json").read_text())
    files = obj["files"]
    if isinstance(files, list):
        files = {row["file"]: row["sha256"] for row in files}
    else:
        files = {key: value["sha256"] if isinstance(value, dict) else value for key, value in files.items()}
    if any(sha(root/key) != digest for key, digest in files.items()):
        raise ValueError("Pinned official source changed")
    return {"repo": obj["repo"], "revision": obj["revision"], "root": str(root), "files": files}


def targets(benchmark, table, metrics, source_path, *, exact=True, note=""):
    return {"benchmark": benchmark, "table": table, "metrics": metrics,
            "source_file": str(source_path), "source_sha256": sha(source_path),
            "reported_only": True, "checkpoint_identity_verified": exact, "note": note}


def build():
    acquisition = json.loads((RUNTIME/"asset-manifest.json").read_text())
    receipts = json.loads((RUNTIME/"public-download-receipt.json").read_text())["files"]
    supports = json.loads((RUNTIME/"support-manifest.json").read_text())
    support_receipts = json.loads((RUNTIME/"support-download-receipt.json").read_text())["files"]
    support = {}
    for name, repo in (("clip", "openai/clip-vit-base-patch32"), ("bert", "google-bert/bert-base-uncased")):
        obj = supports[repo]
        folder = Path(obj["files"][0]["path"]).parent
        support[name] = {"repo": repo, "revision": obj["revision"], "snapshot": str(folder),
                         "files": {row["file"]: support_receipts[row["path"]]["sha256"] for row in obj["files"]}}
    sources = {key: source(key) for key in ("clic", "tripletclip", "negclip_original", "unime_v2", "big_vision")}
    core = json.loads((ROOT/"codex/retrieval_matrix/official_protocols.json").read_text())
    src = core["official_code"]
    sources["core"] = {"repo": src["repo"], "revision": src["revision"], "root": str(ROOT/src["cache"]),
                       "files": {row["path"]: row["sha256"] for row in src["files"]}}
    papers = {name: RUNTIME/"sources"/(name+"-paper"+(".pdf" if name == "tripletclip" else ".html"))
              for name in ("siglip2", "clic", "unimev2", "tripletclip")}
    papers["core"] = ROOT/"dl/retrieval-matrix/sources/core-paper.html"
    models = {}
    aliases = {"unime2": "unime_v2_2b", "unime7": "unime_v2_7b"}
    for key, obj in acquisition["models"].items():
        files = {}
        for row in obj["files"]:
            receipt = receipts.get(row["path"], {})
            if receipt.get("expected_checksum") != row["expected_checksum"] or receipt.get("size") != row["expected_size"]:
                raise ValueError(f"Missing exact original checksum-verified payload: {row['path']}")
            files[row["file"]] = receipt["sha256"]
        key = aliases.get(key, key)
        models[key] = {"kind": "official_embedding", "qualification": "P0_REQUIRED", "repo": obj["repo"],
            "revision": obj["revision"], "snapshot": obj["snapshot"], "files": files, "variant": key,
            "batches": {"image": 4, "text": 32, "cross": 1}, "loader": {"torch_dtype": "float32"},
            "support": {}, "sources": {}, "native_protocol": {}, "published_targets": []}
    model = models["siglip2"]
    model.update(extra_family="siglip2", embedding_dim=1152, sources={"big_vision": sources["big_vision"]},
        native_protocol={"name": "Google SigLIP2 fixed-resolution official HF port", "image_size": 384,
            "text_lowercase": True, "padding": "max_length", "max_length": 64, "truncation": True,
            "metric": "any-positive Recall@1", "coco_split": "TFDS coco_captions/2014 test (Karpathy 5k)",
            "flickr_split": "original Karpathy test1k captions", "qualification_status": "requires exact native datasets and frozen P0",
            "disclosed_adaptation": "Official Google HF PyTorch conversion, not a JAX/TF execution. PIL/HF preprocessing is recorded; compare against BigVision targets without claiming numerical identity."},
        published_targets=[targets("COCO-Karpathy-test5k", "SigLIP2 Table1 So/14 384", {"t2i_R@1": .558, "i2t_R@1": .717}, papers["siglip2"]),
                           targets("Flickr30k-Karpathy-test1k", "SigLIP2 Table1 So/14 384", {"t2i_R@1": .857, "i2t_R@1": .949}, papers["siglip2"])])
    clic_values = {"clic_b32": ({"i2t_R@5": .759, "t2i_R@5": .600}, {"i2t_R@5": .950, "t2i_R@5": .867}, {"replace_ITT": .756, "replace_TOT": .601, "swap_ITT": .611, "swap_TOT": .279}),
                   "clic_l14": ({"i2t_R@5": .801, "t2i_R@5": .654}, None, {"replace_ITT": .790, "replace_TOT": .591, "swap_ITT": .598, "swap_TOT": .263}),
                   "clic_clips_l14": ({"i2t_R@5": .904, "t2i_R@5": .794}, None, {"replace_ITT": .843, "replace_TOT": .719, "swap_ITT": .730, "swap_TOT": .494})}
    for key, (coco, flickr, scpp) in clic_values.items():
        model = models[key]
        model.update(extra_family="clic-clips" if key == "clic_clips_l14" else "clic", embedding_dim=512 if key == "clic_b32" else 768,
            config_file="open_clip_config.json", weight_file="open_clip_model.safetensors", sources={"clic": sources["clic"]},
            native_protocol={"name": "CLIC-LAION/CogVLM author release", "text_cleaner": "clic-pre-caption-30",
                "precision": "author retrieval defaults FP32 without autocast", "metric": "any-positive Recall; numpy.argsort then last K",
                "source_gap": "Paper explicitly uses COCO Val2017; code downloads coco_karpathy_test.json but reads unavailable coco_karpathy_test_2017.json. Canonical captions_val2017 adaptation must be disclosed and cannot be exact qualification until provenance is resolved.",
                "training_family": "CLIPS" if key == "clic_clips_l14" else "OpenAI CLIP", "training_data": "1M LAION captions relabelled by CogVLM"},
            published_targets=[targets("COCO-val2017", "CLIC Table2" if key == "clic_b32" else "CLIC Table3", coco, papers["clic"], note="COCO converted caption JSON is absent from the official release"),
                targets("SugarCrepe++", "CLIC Table1" if key == "clic_b32" else "CLIC Table3", scpp, papers["clic"])])
        if flickr:
            model["published_targets"].append(targets("Flickr30k-Karpathy-test1k", "CLIC Table2", flickr, papers["clic"]))
        if key == "clic_clips_l14":
            model["support"] = {"bert": support["bert"]}
    values = {"triplet_cc12m": ((.1460,.3300,.4384,.1138,.2850,.3904),(.2800,.5590,.6570,.2528,.5238,.6332)),
              "negclip_cc12m": ((.1232,.3016,.4144,.0856,.2311,.3260),(.2470,.4660,.5820,.1866,.4170,.5342))}
    for key, (coco, flickr) in values.items():
        model = models[key]
        model.update(extra_family="triplet-separate", embedding_dim=512,
            sources={"tripletclip": sources["tripletclip"], "core": sources["core"],
                     "checkpoint_code": {"root": model["snapshot"], "revision": model["revision"], "files": {key: value for key,value in model["files"].items() if key.endswith("utils.py")}}},
            support={"clip": support["clip"]},
            native_protocol={"name": "TripletCLIP scratch-trained CC12M ViT-B/32 full encoders", "patch_size": 32,
                "model_name_warning": "ViTB12 denotes 12 transformer layers; it is not patch12", "padding": True,
                "max_length": 77, "normalization": "F.normalize full FP32", "inference": "author HF utils.py classes; CORE released i2t_emb",
                "path_redirect": "Only config.model_name is redirected from bare openai/clip-vit-base-patch32 to its exact pinned local config; no base weights or text settings change",
                "metric": "any-positive Recall@1/5/10", "coco_split": "native retrieval split must be frozen before P0",
                "distinct_from_original_negclip": key == "negclip_cc12m"},
            published_targets=[targets(name, "TripletCLIP Table13 CC12M", dict(zip([f"{d}_R@{k}" for d in ("t2i","i2t") for k in (1,5,10)], value)), papers["tripletclip"])
                               for name, value in (("MSCOCO",coco),("Flickr30k",flickr))])
    proto = core["protocols"]["core_qwen2"]
    model = models["gme7"]
    model.update(extra_family="gme-core", embedding_dim=3584, sources={"core": sources["core"]},
        loader={"torch_dtype": "bfloat16", "attn_implementation": "flash_attention_2"}, processor=proto["processor_init"],
        core_official={"verified": True, **sources["core"], "wrapper": "models/qwen2_vl/qwen2_vl_wrapper.py",
                      "model_class": proto["model_class"], "processor_class": proto["processor_class"], "embedding": "embedding.py", "model_name": "gme"},
        native_protocol={"name": "CORE released GME7 evaluation; same unmodified Qwen2 module/i2t_emb as frozen GME2", "metric": "CORE pytrec relevant-caption fraction, not conventional any-hit I2T Recall"},
        published_targets=[targets("CORE-COCO-val2017", "CORE Table5 GME7", {"t2i_R@5":.795,"t2i_R@10":.868,"i2t_fraction_R@5":.494,"i2t_fraction_R@10":.621}, papers["core"]),
                           targets("CORE-Flickr30k-HF-test1k", "CORE Table5 GME7", {"t2i_R@5":.952,"t2i_R@10":.975,"i2t_fraction_R@5":.748,"i2t_fraction_R@10":.874}, papers["core"])])
    for key, coco, flickr in (("unime_v2_2b", (.537,.651),(.798,.899)),("unime_v2_7b",(.573,.703),(.846,.935))):
        model = models[key]
        model.update(extra_family="unime-v2-native", embedding_dim=1536 if key.endswith("2b") else 3584,
            sources={"unime_v2": sources["unime_v2"]}, loader={"torch_dtype":"bfloat16","attn_implementation":"flash_attention_2"},
            native_protocol={"name": "UniME-V2 author MMEBModel.load + original Qwen2_VL_process_fn + original pooling",
                "image_instruction": "<|image_pad|> Find an image caption describing the given image.", "text_instruction": "",
                "image_resolution": "high (author process_image resize to 1344x1344)", "max_length":4096,"pooling":"eos","normalize":True,
                "metric":"any-positive Recall@1/5/10; author export field names require direction audit",
                "not_core_unime_identity": "CORE Table7 cites original UniME and does not identify an exact Qwen2 checkpoint. V2 is a separate named baseline; do not compare to CORE UniME Table2/5 rows."},
            published_targets=[targets(name,"UniME-V2 Table2",{"i2t_R@1":value[0],"t2i_R@1":value[1]},papers["unimev2"])
                               for name,value in (("MSCOCO-test5k",coco),("Flickr30k-test1k",flickr))])
    conversions = json.loads((RUNTIME/"weights/conversion-receipt.json").read_text())
    for key, filename, source_key in (("negclip_original","negclip-original.pth","negclip_original"),("triplet_openclip_finetune","triplet-openclip-finetune.pt","tripletclip")):
        row = conversions[filename]
        path = Path(row["path"])
        if not row["every_tensor_equal"] or sha(path) != row["sha256"]:
            raise ValueError("Author tensor-only conversion is not verified")
        vals = ((.836,.722),(.964,.917)) if key == "negclip_original" else ((.556,.5332),(.826,.8088))
        paper = papers["clic"] if key == "negclip_original" else papers["tripletclip"]
        models[key] = {"kind":"official_embedding","qualification":"P0_REQUIRED","extra_family":"author-openclip",
            "repo":sources[source_key]["repo"],"revision":sources[source_key]["revision"],"snapshot":str(path.parent),
            "variant":key,"files":{path.name:row["sha256"]},"weight_file":path.name,"embedding_dim":512,
            "sources":{source_key:sources[source_key]},"support":{},"loader":{"torch_dtype":"float32"},
            "batches":{"image":4,"text":32,"cross":1},"conversion_receipt":row,
            "native_protocol":{"name":"author released OpenCLIP ViT-B/32 full weights","text_cleaner":"clic-pre-caption-30" if key == "negclip_original" else "raw OpenCLIP tokenizer",
                "release_revision":"Drive file IDs have no immutable public history; source commit + exact original SHA + tensor conversion receipt identify this copy",
                "metric":"any-positive retrieval recall", "warning":"These finetune weights are separate from scratch CC12M checkpoints",
                "qualification_source_gap":"NegCLIP targets here are CLIC author's published reevaluation, with missing COCO2017 conversion provenance" if key == "negclip_original" else "Author finetune Table16; do not compare with CLIC's differently reported TripletCLIP row"},
            "published_targets":[targets(name,"CLIC Table2 NegCLIP" if key == "negclip_original" else "TripletCLIP Table16",{"i2t_R@5":value[0],"t2i_R@5":value[1]},paper)
                                  for name,value in (("MSCOCO",vals[0]),("Flickr30k",vals[1]))]}
    cs = ROOT/"hf_home/hub/models--kbora--CS-CLIP/snapshots/afd6e90f39fe49137fe15d6a462d994eae4811f5"
    cs_files = {name:sha(cs/name) for name in ("README.md","open_clip_config.json","open_clip_pytorch_model.bin")}
    models["cs_clip"] = {"kind":"official_embedding","qualification":"P0_REQUIRED","extra_family":"author-openclip",
        "repo":"kbora/CS-CLIP","revision":cs.name,"snapshot":str(cs),"files":cs_files,"variant":"cs_clip",
        "config_file":"open_clip_config.json","weight_file":"open_clip_pytorch_model.bin","embedding_dim":512,
        "sources":{"modelcard":{"root":str(cs),"revision":cs.name,"files":{name:cs_files[name] for name in ("README.md","open_clip_config.json")}}},
        "support":{},"loader":{"torch_dtype":"float32"},"batches":{"image":4,"text":32,"cross":1},
        "native_protocol":{"name":"author standard OpenCLIP dual encoder","max_length":77,"image_size":224,
            "metric":"Half-Truth accuracy: strict score(correct shorter caption) > score(foil extension)","weights_reused":True,
            "note":"Existing author's checkpoint is reused without downloading duplicate weights. Half-Truth sets are qualification datasets only, never panel method-development data."},
        "published_targets":[targets(name,"pinned CS-CLIP modelcard",{"half_truth_accuracy":value},cs/"README.md")
            for name,value in (("Half-Truth-COCO-Qwen",.764),("Half-Truth-COCO-Mistral",.672),("Half-Truth-CC3M-Qwen",.662))]}
    for model in models.values():
        validate_spec(model)
    output = {"schema":1,"owner":"[codex]","state":"PREPARATION_ONLY","models":models,
        "acquisition_bytes":acquisition["bytes"]+sum(row["bytes"] for row in conversions.values()),
        "sources":sources,"papers":{key:{"path":str(path),"sha256":sha(path)} for key,path in papers.items()},
        "unresolved":["Original UniME Qwen2 2B/7B exact released checkpoint identity is unresolved; do not substitute UniME-V2 or original Phi/LLaVA variants.",
            "CLIC COCO2017 converted caption JSON provenance is missing from the official release.",
            "All baselines require frozen native P0 datasets/protocol, two published benchmark checks and independent review before P2; no research GPU inference has run."]}
    (HERE/"registry.json").write_text(json.dumps(output,indent=2,sort_keys=True)+"\n")
    print("Prepared identities",len(models),"bytes",output["acquisition_bytes"])


if __name__ == "__main__":
    build()
