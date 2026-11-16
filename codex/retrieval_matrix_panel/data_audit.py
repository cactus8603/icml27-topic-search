"""[codex] Bind independent development audit to all current frozen inputs."""
from codex.retrieval_matrix_panel import qualification as Q


def validate(audit, manifest, images, *, manifest_sha, source_sha, images_sha):
    if (audit.get("state")!="PASS" or audit.get("independent_recomputation") is not True
            or not audit.get("auditor") or audit.get("model_results_read") is not False
            or audit.get("original_test_queries_opened") is not False
            or audit.get("manifest_sha256")!=manifest_sha
            or audit.get("source_receipt_sha256")!=source_sha
            or manifest.get("source_receipt_sha256")!=source_sha
            or audit.get("freeze_commit")!=manifest.get("freeze_commit")
            or len(audit.get("freeze_commit",""))!=40
            or audit.get("gallery")!=20000 or manifest.get("gallery_size")!=20000
            or audit.get("exclusions")!=manifest.get("exclusions")
            or set(manifest.get("exclusions",{}))!={"original_test","cola_paco","karpathy_test","half_truths"}
            or any(row.get("overlap_with_new_dev")!=0 for row in manifest["exclusions"].values())):
        raise ValueError("Independent development audit is not bound to this isolated frozen version")
    checked_images=audit.get("images",{})
    if (checked_images.get("all_payloads_hashed_decoded") is not True
            or checked_images.get("verified")!=20000
            or checked_images.get("receipt_sha256")!=images_sha
            or images.get("receipt_sha256")!=images_sha
            or images.get("expected")!=20000 or images.get("ok")!=20000 or images.get("failures")!=[]
            or checked_images.get("bytes")!=images.get("bytes")):
        raise ValueError("Independent complete image audit does not bind current receipt bytes")
    for name in ("conjunction","negation"):
        actual=audit.get("queries",{}).get(name,{})
        expected=manifest[name]
        if (actual.get("queries")!=expected["queries"] or actual.get("dev_small")!=expected["dev_small_queries"]
                or actual.get("full_metadata_and_text_equal") is not True
                or actual.get("hash_round_robin_rebuilt") is not True
                or actual.get("three_value_truth_rebuilt_all_images") is not True
                or actual.get("positive_disjoint") is not True):
            raise ValueError("Independent full-query reconstruction does not cover this panel")
    return audit


def verify(audit_path,manifest_path,source_path,summary_path,receipt_path):
    return validate(Q.read(audit_path),Q.read(manifest_path),Q.read(summary_path),
        manifest_sha=Q.sha(manifest_path),source_sha=Q.sha(source_path),images_sha=Q.sha(receipt_path))


def bind_datasets(plan,manifest):
    """A valid v2 audit cannot authorize a v1/different dataset entry."""
    if set(plan["datasets"])!={"conjunction-dev","negation-dev","cola-gqa","cola-paco"}:
        raise ValueError("P2 requires the fixed complete panel/COLA dataset set")
    for dataset,panel in (("conjunction-dev","conjunction"),("negation-dev","negation")):
        spec=plan["datasets"][dataset]
        expected=manifest[panel]
        if (spec.get("format")!="conjunction" or spec.get("split")!="dev" or "subset" in spec
                or spec.get("expected_gallery")!=20000
                or spec.get("expected_queries")!=expected["queries"]
                or spec.get("expected_eligible")!=expected["queries"]
                or spec["gallery"]["sha256"]!=expected["outputs"]["gallery.json"]
                or spec["queries"]["sha256"]!=expected["outputs"]["queries.jsonl"]
                or spec["images_receipt"]!=plan["development_images_receipt"]):
            raise ValueError("Executing P2 dataset is not the independently audited v2 panel")
