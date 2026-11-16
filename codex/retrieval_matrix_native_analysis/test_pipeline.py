"""[codex] Disposable end-to-end wrapper/reading integration.

Only source-freeze lookup, prospective synthetic path mapping, the CPU-module
probe (separately exercised against all real frozen environments), and the
mechanical subprocess transport are replaced. Original native numerical
review/make/admit functions, checkpoint checks and durable reading are real.
All artifacts, including apparent PASS records, are private temporary fixtures.
"""
from argparse import Namespace
from copy import deepcopy
from pathlib import Path
import json,unittest
from unittest.mock import patch
from codex.retrieval_matrix_native_analysis import contracts as C,workflow as W,launcher as L,synthetic as S

class Pipeline(unittest.TestCase):
    def test_three_family_wrappers_preserve_true_native_review_make_admit(self):
        for family in ('siglip2','paired','reranker'):
            with self.subTest(family=family),S.fixture(family,variant='qwen2' if family=='reranker' else None) as f:
                control=f.root/'analysis';control.mkdir();prereg=control/'PREREG.md'
                prereg.write_bytes(b'Explicit disposable SYNTHETIC integration\n'+C.MARKER)
                plan_path=control/'SYNTHETIC-analysis-plan.json';plan_path.write_text('{"SYNTHETIC":true}\n')
                if family=='reranker':
                    from codex.retrieval_matrix_rerank_native_p0 import contracts as N,runner as NR
                    producer=deepcopy(S.E.read(f.plan_path));variant=producer['variants'][f.key]
                    rows=N.load_examples(producer,variant);scores=NR.verify_complete(f.out,S.E.read(f.out/'identity.json'),rows,variant)
                    _,detail=N.evaluate(rows,scores,variant);S.overwrite(f.out/'results/per-example.json',detail)
                    result_paths=[f.out/'results/report.json',f.out/'results/per-example.json']
                    smoke=f.root/'smoke/smoke.json';smoke.parent.mkdir()
                    S.overwrite(smoke,{'state':'PASS','technical_only':True,'identity':S.E.read(f.out/'identity.json'),'max_abs':0.})
                    completes=[f.out/'complete.json']
                else:
                    result_paths=list(f.reports.values());completes=[Path(S.E.read(p)['index_receipt']['path']) for p in result_paths]
                payloads={str(p):p.read_bytes() for p in result_paths}
                raw=[str(p) for p in f.root.rglob('*') if p.is_file() and p not in result_paths and not p.is_relative_to(control)]
                for p in result_paths:p.unlink()
                profile={'id':'SYNTHETIC-'+family,'family':family,'model':f.key or S.E.read(f.plan_path)['model_key'],
                    'producer_plan':C.entry(f.plan_path),'producer_commit':S.REFS[family],
                    'output_root':str(f.root),'score_root':str(f.out) if f.out else None,
                    'benchmarks':['COLA','SugarCrepe++','NegBench'] if family=='reranker' else list(f.reports),
                    'qualification_module':C.entry(f.Q.__file__),'qualification_scope':C.SCOPES[family],
                    'model_spec_sha256':C.digest(f.model_spec)}
                job={'id':profile['id'],'profile':profile,'producer_plan':profile['producer_plan'],
                    'producer_commit':profile['producer_commit'],'model_spec_sha256':profile['model_spec_sha256'],
                    'cpu':{'SYNTHETIC':'environment check has separate actual-family CPU probes'},
                    'roles':{'known_implementation_authors':[],'analysis_authors':W.ANALYSIS_AUTHORS,'excluded_reviewers':W.EXCLUDED},
                    'outputs':{name:str(control/(name+'.json')) for name in ('mechanical','independent-review','independent-envelope','qualification','admission')},
                    'stage_attempts':{stage:str(control/(stage+'-attempt.json')) for stage in W.STAGES}}
                manifest={'immutable_raw':raw,'completion':list(map(str,completes)),
                    'mechanical_results':list(map(str,result_paths)),'producer_mutable_analysis_ledger':[],
                    'cost':{'SYNTHETIC':True}}
                mp=control/'manifest.json';S.overwrite(mp,manifest);job['manifest']=C.entry(mp)
                plan={'jobs':{job['id']:job}}
                runtime={'environment_recipe':job['cpu'],'cuda_initialized':False,'synthetic_probe_boundary':True}
                job['cpu_runtime']=runtime
                def command(p,b=None):return ['SYNTHETIC-original-mechanical',p['family'],str(b)]
                def transport(argv,**kwargs):
                    if argv[:1]!=['SYNTHETIC-original-mechanical']:raise AssertionError('No real subprocess/GPU invocation permitted')
                    for name,body in payloads.items():
                        path=Path(name)
                        if not path.exists():path.write_bytes(body)
                with patch.object(W,'HERE',control),patch.object(W,'ANCHORS',control/'anchor'),\
                        patch.object(W,'source_gate',return_value=plan),patch.object(W,'cpu_environment',return_value=runtime),\
                        patch.object(W,'reports',return_value=f.reports),patch.object(W,'facade',return_value=f.Q),\
                        patch.object(C,'mechanical_argv',side_effect=command),patch.object(L,'run_original',side_effect=transport):
                    a=Namespace(plan=plan_path,freeze_commit='b'*40,job=job['id'],actor='SYNTHETIC-mechanical-actor',
                        stage='mechanical',nonimplementer=False,accept_disclosed_protocol=True)
                    L.run_worker(a)
                    a.stage='independent';a.actor='SYNTHETIC-fresh-nonimplementer';a.nonimplementer=True
                    L.run_worker(a)
                    envelope=W.read(job['outputs']['independent-envelope']);self.assertEqual(envelope['state'],'PASS')
                    a.stage='qualify';a.actor='SYNTHETIC-qualification-operator'
                    L.run_worker(a)
                    qualified=W.read(job['outputs']['qualification']);admission=W.read(job['outputs']['admission'])
                    self.assertEqual(qualified['scope'],C.SCOPES[family]);self.assertEqual(qualified['model_spec_sha256'],C.digest(f.model_spec))
                    self.assertTrue(admission['downstream_admitted']);self.assertEqual(admission['user_decision'],'PENDING')
                    rows=W.frozen_reader().records(prereg,expected_plan=C.sha(plan_path),expected_commit='b'*40,anchor_root=control/'anchor')
                    self.assertEqual([r['event'] for r in rows],['READ_STARTED','READ_COMPLETED']*3)
                    with self.assertRaises(ValueError):L.run_worker(a)
                self.assertTrue(all(Path(name).read_bytes()==body for name,body in payloads.items()))
            self.assertFalse(f.root.exists())

if __name__=='__main__':unittest.main()
