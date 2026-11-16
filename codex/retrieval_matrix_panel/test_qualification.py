"""[codex] Reject wrong metrics, duplicated benchmarks, or self-review."""
from copy import deepcopy
import unittest
from codex.retrieval_matrix_panel.qualification import assess


class Qualification(unittest.TestCase):
    def fixture(self):
        model = {"repo":"test/model", "revision":"a"*40,
                 "published_targets":{"coco_t2i_recall_5":.8,"flickr_t2i_recall_5":.9,"negbench_coco_recall_5":.7}}
        ds={name:{"expected_gallery":5,"expected_queries":10,"expected_eligible":10} for name in ("coco","flickr","negbench")}
        plan={"models":{"m":model},"datasets":ds,"settings":{"main_k":100}}
        reports={}
        for name,value in (("coco",.8),("flickr",.9),("negbench",.6)):
            reports[name]={"model":"m","dataset":name,"queries":10,"raw_queries":10,"gallery":5,"excluded":[],
                "official":{"R@5":value} if name=="negbench" else {"t2i":{"recall_5":value}},
                "provenance":{"index":{"identity":{"model":"m","dataset":name,"plan_sha256":"p",
                    "model_spec":model,"settings":plan["settings"]}}}}
        review={"schema":1,"state":"PASS","scope":"official-native-p0","reviewer":"agent-b","producer":"agent-a",
                "independent_recomputation":True,"audited_arrays":True,"model":"m","plan_sha256":"p"}
        return plan,reports,review

    def test_requires_two_datasets_and_retains_failed_third(self):
        p,r,v=self.fixture(); x=assess(p,"p","m",r,v)
        self.assertEqual(x["state"],"PASS")
        self.assertFalse(x["datasets"]["negbench"]["within_1pp"])
        r["flickr"]["official"]["t2i"]["recall_5"]=.88
        self.assertEqual(assess(p,"p","m",r,v)["state"],"FAILED")

    def test_no_secondary_metric_substitution(self):
        p,r,v=self.fixture(); r["coco"]["official"]=None
        with self.assertRaises(ValueError): assess(p,"p","m",r,v)

    def test_no_wrong_model_dataset_or_denominator(self):
        for key,value in (("model","other"),("dataset","flickr"),("queries",9)):
            p,r,v=self.fixture(); r["coco"][key]=value
            with self.assertRaises(ValueError): assess(p,"p","m",r,v)

    def test_no_self_review(self):
        p,r,v=self.fixture(); v["reviewer"]=v["producer"]
        with self.assertRaises(ValueError): assess(p,"p","m",r,v)

    def test_no_missing_or_duplicated_dataset(self):
        p,r,v=self.fixture(); del r["negbench"]
        with self.assertRaises(ValueError): assess(p,"p","m",r,v)


if __name__ == "__main__": unittest.main()
