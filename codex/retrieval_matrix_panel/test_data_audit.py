from copy import deepcopy
import unittest
from codex.retrieval_matrix_panel.data_audit import validate,bind_datasets


class IndependentAuditBinding(unittest.TestCase):
    def fixture(self):
        exclusions={name:{"overlap_with_new_dev":0} for name in ("original_test","cola_paco","karpathy_test","half_truths")}
        manifest={"source_receipt_sha256":"s","freeze_commit":"a"*40,"gallery_size":20000,"exclusions":exclusions,
            **{name:{"queries":10,"dev_small_queries":2} for name in ("conjunction","negation")}}
        audit={"state":"PASS","independent_recomputation":True,"auditor":"separate-agent","model_results_read":False,
            "original_test_queries_opened":False,"manifest_sha256":"m","source_receipt_sha256":"s",
            "freeze_commit":"a"*40,"gallery":20000,"exclusions":deepcopy(exclusions),
            "images":{"all_payloads_hashed_decoded":True,"verified":20000,"receipt_sha256":"i","bytes":200},
            "queries":{name:{"queries":10,"dev_small":2,"full_metadata_and_text_equal":True,
                "hash_round_robin_rebuilt":True,"three_value_truth_rebuilt_all_images":True,"positive_disjoint":True}
                for name in ("conjunction","negation")}}
        images={"receipt_sha256":"i","expected":20000,"ok":20000,"failures":[],"bytes":200}
        return audit,manifest,images

    def test_matching_complete_audit_passes(self):
        validate(*self.fixture(),manifest_sha="m",source_sha="s",images_sha="i")

    def test_rejects_other_manifest_source_or_image_receipt(self):
        for field in ("manifest_sha","source_sha","images_sha"):
            args={"manifest_sha":"m","source_sha":"s","images_sha":"i"};args[field]="wrong"
            with self.assertRaises(ValueError):validate(*self.fixture(),**args)

    def test_pass_flag_without_independent_image_and_query_rebuild_is_insufficient(self):
        for mutate in (lambda a:a.update(independent_recomputation=False),
                       lambda a:a["images"].update(verified=19999),
                       lambda a:a["queries"]["conjunction"].update(full_metadata_and_text_equal=False)):
            a,m,i=self.fixture();mutate(a)
            with self.assertRaises(ValueError):validate(a,m,i,manifest_sha="m",source_sha="s",images_sha="i")

    def test_correct_v2_audit_cannot_authorize_wrong_executing_query_entries(self):
        manifest={name:{"queries":10,"outputs":{"gallery.json":"g","queries.jsonl":name}}
                  for name in ("conjunction","negation")}
        receipt={"path":"receipt","sha256":"i"}
        datasets={key:{"format":"conjunction","split":"dev","expected_gallery":20000,
            "expected_queries":10,"expected_eligible":10,"gallery":{"sha256":"g"},
            "queries":{"sha256":panel},"images_receipt":receipt}
            for key,panel in (("conjunction-dev","conjunction"),("negation-dev","negation"))}
        plan={"datasets":{**datasets,"cola-gqa":{},"cola-paco":{}},"development_images_receipt":receipt}
        bind_datasets(plan,manifest)
        plan["datasets"]["conjunction-dev"]["queries"]["sha256"]="v1-queries"
        with self.assertRaises(ValueError):bind_datasets(plan,manifest)


if __name__=="__main__":unittest.main()
