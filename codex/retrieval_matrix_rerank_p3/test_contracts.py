"""[codex] Synthetic coverage/admission/batching checks; no real model or effect."""
from types import SimpleNamespace
import unittest
import numpy as np
from codex.retrieval_matrix_rerank_p3 import contracts as C


class Contracts(unittest.TestCase):
    def fixture(self):
        gallery=[{"id":str(i),"path":f"image-{i}","sha256":"a"*64} for i in range(210)]
        return {"gallery":gallery,"queries":[{"id":"q","text":"synthetic query","positives":["209"],"ignore":[str(i) for i in range(200)]}]},np.arange(210).reshape(1,-1)

    def test_labels_never_filter_or_refill_original_admission(self):
        bundle,order=self.fixture();pairs=C.admit_pairs(bundle,order,k=200,dataset="dev")
        self.assertEqual([p["candidate_id"] for p in pairs],list(map(str,range(200))))
        self.assertFalse(any({"positives","ignore","hard_list"}&p.keys() for p in pairs))
        self.assertEqual(len({p["id"] for p in pairs}),200)

    def test_invalid_or_partial_order_is_rejected(self):
        bundle,order=self.fixture()
        for bad in (order[:,:200],np.zeros((1,210),dtype=int)):
            with self.assertRaises(ValueError):C.admit_pairs(bundle,bad,k=200,dataset="dev")

    def test_shared_prefix_ties_keep_embedding_order_and_tail(self):
        _,order=self.fixture();values=np.zeros((1,200))
        np.testing.assert_array_equal(C.reranked(order,values,100),order)
        values[0,1]=1;ranked=C.reranked(order,values,50)
        self.assertEqual(list(ranked[0,:3]),[1,0,2]);np.testing.assert_array_equal(ranked[:,50:],order[:,50:])

    def test_storage_chunks_preserve_pairbatch4_and_original_raw(self):
        pairs=[{"id":str(i)} for i in range(130)];calls=[]
        def score(group,**kwargs):
            calls.append([p["id"] for p in group])
            return [{"id":p["id"],"score":.25,"official_raw":[.25]} for p in group]
        backend=SimpleNamespace(score_pairs=score)
        actual=[]
        for start in range(0,len(pairs),64):actual+=C.score_groups(backend,pairs[start:start+64],instruction="fixed")
        self.assertEqual(calls,[[str(i) for i in range(s,min(s+4,130))] for s in range(0,130,4)])
        self.assertEqual([r["official_raw"] for r in actual],[[.25]]*130)

    def test_wrong_response_order_or_missing_pair_is_rejected(self):
        backend=SimpleNamespace(score_pairs=lambda p,**k:[{"id":"wrong","score":1}])
        with self.assertRaises(ValueError):C.score_groups(backend,[{"id":"a"}],instruction="fixed")




class EvidenceAndResume(unittest.TestCase):
    def fixture_report(self):
        from copy import deepcopy
        targets={"cola_text_choice":.7,"scpp":.8,"negbench_mcq":.6}
        plan={"counts":deepcopy(C.COUNTS),"acceptance":{"core2":targets,"absolute_tolerance":.01}}
        report={"rows":15912,"subsets":{},"benchmarks":{}}
        for task,names in C.SUBSETS.items():
            for name in names:report["subsets"][name]={"rows":C.COUNTS[name],"accuracy":targets[task]}
            report["benchmarks"][task]={"subsets":sorted(names),"rows":sum(C.COUNTS[n] for n in names),"macro_accuracy":targets[task]}
        return plan,report

    def test_mcq_requires_three_named_complete_denominators(self):
        from copy import deepcopy
        plan,report=self.fixture_report()
        self.assertEqual(C.mcq_report(plan,"core2",report),3)
        for mutate in (lambda p,r:r["subsets"]["voc_mcq"].update(rows=4951),
                       lambda p,r:r["benchmarks"]["negbench_mcq"].update(macro_accuracy=.4),
                       lambda p,r:r["benchmarks"].pop("scpp"),
                       lambda p,r:p["acceptance"]["core2"].update(scpp=float("nan"))):
            p,r=deepcopy(plan),deepcopy(report);mutate(p,r)
            with self.assertRaises(ValueError):C.mcq_report(p,"core2",r)

    def test_mcq_third_failure_remains_visible_and_two_match(self):
        plan,report=self.fixture_report()
        report["benchmarks"]["cola_text_choice"]["macro_accuracy"]=.5
        report["subsets"]["cola"]["accuracy"]=.5
        self.assertEqual(C.mcq_report(plan,"core2",report),2)
        for name in C.SUBSETS["scpp"]:report["subsets"][name]["accuracy"]=.5
        report["benchmarks"]["scpp"]["macro_accuracy"]=.5
        with self.assertRaises(ValueError):C.mcq_report(plan,"core2",report)

    def test_mcq_self_review_and_changed_model_fail_before_arrays(self):
        import tempfile
        from pathlib import Path
        from copy import deepcopy
        with tempfile.TemporaryDirectory(dir=C.ROOT/"hf_home/tmp",prefix="p3-mcq-test-") as folder:
            folder=Path(folder);plan,report=self.fixture_report();spec={"backend":"swift_released","revision":"a"*40}
            plan.update(state="FROZEN",models={"core2":spec},pair_batch=4)
            C.R.atomic_json(folder/"plan.json",plan)
            identity={"plan_sha256":C.R.sha(folder/"plan.json"),"model":"core2","model_spec":spec}
            C.R.atomic_json(folder/"complete.json",{"state":"COMPLETE","identity":identity})
            C.R.atomic_json(folder/"report.json",report)
            review={"state":"PASS","scope":"official-reranker-mcq","independent_recomputation":True,
                "audited_arrays":True,"reviewer":"same","producer":"same","plan_sha256":identity["plan_sha256"],
                "model":"core2","complete_sha256":C.R.sha(folder/"complete.json"),"report_sha256":C.R.sha(folder/"report.json")}
            C.R.atomic_json(folder/"review.json",review)
            proof={key:{"path":str(folder/(key+".json")),"sha256":C.R.sha(folder/(key+".json"))} for key in ("plan","complete","report","review")};proof["model"]="core2"
            with self.assertRaises(ValueError):C.qualify_reranker(spec,proof)
            with self.assertRaises(ValueError):C.qualify_reranker({**spec,"revision":"b"*40},proof)
            (folder/"report.json").write_text("{}")
            with self.assertRaises(ValueError):C.qualify_reranker(spec,proof)

    def test_raw_transaction_recovers_after_state_commit_interruption_without_model_replay(self):
        import tempfile
        from pathlib import Path
        from unittest.mock import patch
        from codex.retrieval_matrix_rerank_p3 import runner as P
        pairs=[{"id":str(i),"position":i} for i in range(137)]
        calls=[];identity={"dataset":"synthetic","model":"core2","plan_sha256":"b"*64};plan={"instruction":"fixed"}
        def score(group,**kwargs):
            calls.append([p["id"] for p in group])
            return [{"id":p["id"],"score":float(int(p["id"]))/1000,"official_raw":[float(int(p["id"]))/1000]} for p in group]
        engine=SimpleNamespace(score_pairs=score)
        with tempfile.TemporaryDirectory(dir=C.ROOT/"hf_home/tmp",prefix="p3-resume-test-") as folder:
            out=Path(folder);C.R.atomic_json(out/"backend.json",{"backend":"synthetic"})
            def compute(group):
                responses=[]
                for start in range(0,len(group),4):responses+=P.raw_group(out,group[start:start+4],identity,plan,lambda:engine)
                return np.array([[r["score"]] for r in responses],dtype=np.float64)
            original=C.R.C.atomic_json
            def crash(path,value):
                if Path(path).name=="score.state.json":raise RuntimeError("synthetic crash after raw, before state")
                return original(path,value)
            with patch.object(C.R.C,"atomic_json",side_effect=crash):
                with self.assertRaisesRegex(RuntimeError,"synthetic crash"):
                    C.R.checkpoint(out/"score.npy",pairs,64,compute,identity)
            self.assertEqual(len(calls),16)
            values=C.R.checkpoint(out/"score.npy",pairs,64,compute,identity)
            self.assertEqual(calls,[[str(i) for i in range(s,min(s+4,137))] for s in range(0,137,4)])
            np.testing.assert_array_equal(values[:,0],np.arange(137)/1000)
            before=len(calls);C.R.checkpoint(out/"score.npy",pairs,64,compute,identity);self.assertEqual(len(calls),before)
            with self.assertRaises(ValueError):C.R.checkpoint(out/"score.npy",pairs,64,compute,{**identity,"model":"changed"})
            raw=out/"raw/000000000-000000004.json";saved=C.R.read(raw)
            saved["responses"][0]["official_raw"]=[1];C.R.atomic_json(raw,saved)
            with self.assertRaises(ValueError):P.raw_group(out,pairs[:4],identity,plan,lambda:self.fail("must reject before model"))

    def test_failed_partial_storage_chunk_records_and_reuses_successful_raw_pairs(self):
        import tempfile
        from pathlib import Path
        from codex.retrieval_matrix_rerank_p3 import runner as P
        pairs=[{"id":str(i),"position":i} for i in range(12)];calls=[];failed=[False]
        identity={"dataset":"synthetic"};plan={"instruction":"fixed"}
        def score(group,**kwargs):
            calls.append(group[0]["id"])
            if group[0]["id"]=="4" and not failed[0]:failed[0]=True;raise RuntimeError("synthetic model interruption")
            return [{"id":p["id"],"score":.5,"official_raw":[.5]} for p in group]
        engine=SimpleNamespace(score_pairs=score)
        with tempfile.TemporaryDirectory(dir=C.ROOT/"hf_home/tmp",prefix="p3-partial-test-") as folder:
            out=Path(folder);C.R.atomic_json(out/"backend.json",{"backend":"synthetic"})
            with self.assertRaises(RuntimeError):
                for start in range(0,12,4):P.raw_group(out,pairs[start:start+4],identity,plan,lambda:engine)
            for start in range(0,12,4):P.raw_group(out,pairs[start:start+4],identity,plan,lambda:engine)
            self.assertEqual(calls,["0","4","4","8"])
            self.assertIn('FAILED_OR_INTERRUPTED',(out/"calls.jsonl").read_text())

    def test_synthetic_smoke_evidence_rejects_missing_nan_and_wrong_identity(self):
        from copy import deepcopy
        from codex.retrieval_matrix_rerank_p3 import runner as P
        identity={k:k for k in ("plan_sha256","freeze_commit","model","model_spec","environment")}
        proof={"state":"PASS","technical_only":True,"identity":identity,"batch1_vs4_max_abs":.005}
        plan={"settings":{"smoke_max_abs":.01}};P.require_smoke(proof,identity,plan)
        for bad in (float("nan"),.011,-.1,None):
            p=deepcopy(proof);p["batch1_vs4_max_abs"]=bad
            with self.assertRaises(ValueError):P.require_smoke(p,identity,plan)
        with self.assertRaises(ValueError):P.require_smoke(proof,{**identity,"model":"changed"},plan)


class FamilyAndFreeze(unittest.TestCase):
    def test_central_gpu_guard_rejects_unassigned_before_importing_engine(self):
        from unittest.mock import patch
        from codex.retrieval_matrix_rerank_p3 import runner as P
        for env in ({},{"RETRIEVAL_CENTRAL_OWNER":"[codex]"},
                    {"RETRIEVAL_CENTRAL_OWNER":"other","RETRIEVAL_CENTRAL_TASK":"x","CUDA_VISIBLE_DEVICES":"0"}):
            with patch.dict(P.os.environ,env,clear=True):
                with self.assertRaisesRegex(ValueError,"central dispatcher"):P.environment({})

    def test_producing_git_blobs_bind_plan_and_sources_without_requiring_current_p2_source(self):
        import tempfile
        from pathlib import Path
        from copy import deepcopy
        from unittest.mock import patch
        with tempfile.TemporaryDirectory(dir=C.ROOT/"hf_home/tmp",prefix="p3-source-test-") as folder:
            plan_path=Path(folder)/"plan.json";source="codex/synthetic-source.py";prereg="codex/synthetic-prereg.md"
            blobs={source:b"old frozen source",prereg:b"old fixed rule"}
            sha=lambda b:C.hashlib.sha256(b).hexdigest()
            plan={"source_files":{k:sha(v) for k,v in blobs.items()},"prereg":prereg}
            C.R.atomic_json(plan_path,plan);relative=str(plan_path.relative_to(C.ROOT));blobs[relative]=plan_path.read_bytes()
            identity={"freeze_commit":"a"*40,"source_files":{k:sha(v) for k,v in blobs.items()}}
            with patch.object(C.R.C,"git_bytes",side_effect=lambda commit,path:blobs[path]):
                C.frozen_origin(plan_path,plan,identity,include_plan=True)
                bad=deepcopy(identity);bad["source_files"][source]="b"*64
                with self.assertRaises(ValueError):C.frozen_origin(plan_path,plan,bad,include_plan=True)
                bad=deepcopy(identity);bad["source_files"].pop(relative)
                with self.assertRaises(ValueError):C.frozen_origin(plan_path,plan,bad,include_plan=True)
                with self.assertRaises(ValueError):C.frozen_origin(plan_path,plan,{**identity,"freeze_commit":"a"*8})
                plan_path.write_text("{}")
                with self.assertRaises(ValueError):C.frozen_origin(plan_path,plan,identity,include_plan=True)

    def test_clip_phase_uses_native_admit_and_never_six_mllm_validator(self):
        import tempfile
        from pathlib import Path
        from unittest.mock import patch
        with tempfile.TemporaryDirectory(dir=C.ROOT/"hf_home/tmp",prefix="p3-family-test-") as folder:
            folder=Path(folder);adapter={"path":str(C.ROOT/C.NATIVE_VALIDATOR),"sha256":C.R.sha(C.ROOT/C.NATIVE_VALIDATOR)}
            spec={"family":"CLIP","variant":"synthetic"};settings={"top_ks":[50,100,200],"main_k":100}
            plan={"schema":1,"state":"FROZEN","phase":C.PHASES["clip"],"retriever_family":"CLIP",
                "models":{"clip":spec},"settings":settings,"qualification":{"clip":{"path":"unused"}},"qualification_adapter":adapter}
            C.R.atomic_json(folder/"plan.json",plan)
            identity={"model":"clip","dataset":"conjunction-dev","plan_sha256":C.R.sha(folder/"plan.json"),"model_spec":spec,"settings":settings}
            C.R.atomic_json(folder/"provenance.json",identity)
            C.R.atomic_json(folder/"index-meta.json",{"state":"COMPLETE","identity":identity})
            entry=lambda name:{"path":str(folder/name),"sha256":C.R.sha(folder/name)}
            ref={"family":"clip","model":"clip","plan":entry("plan.json"),"qualification_adapter":adapter,
                "indexes":{"conjunction-dev":{"meta":entry("index-meta.json"),"identity":entry("provenance.json")}}}
            def execute(module):
                module.admit=lambda receipt,model:{"state":"PASS","family":"CLIP","model":"clip","model_spec_sha256":C.Q.digest(model)}
            loader=SimpleNamespace(exec_module=execute)
            fake=SimpleNamespace(loader=loader)
            with patch.object(C,"frozen_origin"),patch.object(C.importlib.util,"spec_from_file_location",return_value=fake),\
                    patch.object(C.importlib.util,"module_from_spec",return_value=SimpleNamespace()),\
                    patch.object(C.R.models,"validate_spec",side_effect=AssertionError("six MLLM validator must not run")),\
                    patch.object(C.Q,"admit",side_effect=AssertionError("six MLLM admit must not run")):
                actual,_,_=C.first_stage_header(ref,"conjunction-dev");self.assertEqual(actual["phase"],C.PHASES["clip"])
                C.R.atomic_json(folder/"index-meta.json",{"state":"RUNNING","identity":identity});ref["indexes"]["conjunction-dev"]["meta"]=entry("index-meta.json")
                with self.assertRaises(ValueError):C.first_stage_header(ref,"conjunction-dev")
                plan["phase"]=C.PHASES["mllm"];C.R.atomic_json(folder/"plan.json",plan);ref["plan"]=entry("plan.json")
                with self.assertRaises(ValueError):C.first_stage_header(ref,"conjunction-dev")


if __name__=="__main__":unittest.main()
