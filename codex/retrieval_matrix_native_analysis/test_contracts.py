"""[codex] Bounded source/CPU checks; no real producer output is consumed."""
from copy import deepcopy
import json,os,tempfile,unittest
from pathlib import Path
import numpy as np
from codex.retrieval_matrix_native_analysis import contracts as C


class NativePreparation(unittest.TestCase):
    @classmethod
    def setUpClass(cls):cls.inventory=C.profiles()

    def test_exact_seven_source_profiles_keep_native_denominators_and_scopes(self):
        p=self.inventory['profiles'];self.assertEqual(len(p),7)
        self.assertFalse(self.inventory['execution_plan']);self.assertFalse(self.inventory['qualification_emitted'])
        sig=p['native-siglip2-einops-v1']
        self.assertEqual(sig['counts'],{'coco':{'images':5000,'texts':25010},'flickr':{'images':1000,'texts':5000}})
        self.assertEqual(sig['qualification_scope'],C.SCOPES['siglip2'])
        for key in ('clic_b32','clic_l14','clic_clips_l14'):
            row=p['native-paired-'+key+'-einops-v1']
            self.assertEqual(row['counts'],{'sugarcrepe':{'images':7511,'texts':15022},'sugarcrepe_pp':{'images':4757,'texts':14271}})
            self.assertNotIn('add_ITT',row['primary_metric_names']['sugarcrepe_pp'])
            self.assertEqual(row['qualification_scope'],C.SCOPES['paired'])
        self.assertEqual(p['native-rerank-jina_m0']['cola_rule'],'cola_image_choice')
        self.assertEqual(p['native-rerank-qwen2']['cola_rule'],'cola_text_choice')
        self.assertEqual(p['native-rerank-qwen8']['counts'],{'examples':15912,'pairs':58891,'subsets':8})

    def test_original_mechanical_command_apis_do_not_produce_embeddings(self):
        p=self.inventory['profiles']
        for key,benchmark in [('native-siglip2-einops-v1','coco'),('native-paired-clic_b32-einops-v1','sugarcrepe_pp'),('native-rerank-qwen2',None)]:
            command=C.mechanical_argv(p[key],benchmark)
            self.assertIn('analyze',command);self.assertNotIn('index',command);self.assertNotIn('score',command)
            self.assertEqual(command[command.index('--freeze-commit')+1],p[key]['producer_commit'])
        with self.assertRaises(ValueError):C.mechanical_argv(p['native-rerank-jina_m0'],'COLA')
        with self.assertRaises(ValueError):C.mechanical_argv(p['native-siglip2-einops-v1'],'negbench')

    def test_qualification_routes_use_original_new_facades_and_explicit_acceptance(self):
        p=self.inventory['profiles'];output=C.ROOT/'hf_home/tmp/synthetic-native-review.json'
        sig=p['native-siglip2-einops-v1']
        with self.assertRaisesRegex(ValueError,'explicit independent acceptance'):C.qualification_argv(sig,'review',output,reviewer='synthetic')
        command=C.qualification_argv(sig,'review',output,reviewer='synthetic',accept_protocol=True)
        self.assertIn('--reports-root',command);self.assertIn('--accept-disclosed-protocol',command)
        paired=p['native-paired-clic_l14-einops-v1']
        self.assertIn('retrieval_matrix_paired_runtime_fix/qualification.py',C.qualification_argv(paired,'review',output,reviewer='synthetic',accept_protocol=True)[2])
        native=p['native-rerank-jina_m0'];command=C.qualification_argv(native,'review',output,reviewer='synthetic')
        self.assertIn('--run',command);self.assertNotIn('--reports-root',command)
        self.assertTrue(command[command.index('--run')+1].endswith('/scores'))
        with self.assertRaises(ValueError):C.qualification_argv(native,'qualify',output)

    def test_changed_model_ref_or_scope_cannot_be_adopted(self):
        row=self.inventory['profiles']['native-rerank-qwen2']
        for mutate in (lambda p:p.update(producer_commit='f'*40),lambda p:p.update(model='qwen_native2'),
                lambda p:p.update(qualification_scope='official-reranker-mcq')):
            changed=deepcopy(row);mutate(changed)
            with self.assertRaisesRegex(ValueError,'exact frozen'):C.mechanical_argv(changed)

    def test_siglip_google_fp32_full_shape_ties_any_positive_recall(self):
        from codex.retrieval_matrix_native_p0 import metrics
        pp=Path(self.inventory['profiles']['native-siglip2-einops-v1']['producer_plan']['path']);plan=json.loads(pp.read_text())
        images=np.tile(np.array([[1,0]],dtype=np.float32),(3,1));texts=np.tile(images[:1],(6,1));positive=[0,0,1,1,2,2]
        author=metrics.author_metrics(images,texts,positive,plan['external_sources']['google_metric'])
        independent=metrics.independent_metrics(images,texts,positive)
        self.assertEqual(author,independent);self.assertAlmostEqual(author['i2t_R@1'],1/3)
        self.assertEqual(author['i2t_R@5'],1.)
        with self.assertRaises(ValueError):metrics.independent_metrics(images.astype(np.float64),texts,positive)

    def test_paired_scalar_directions_ties_and_unweighted_subset_macros(self):
        from codex.retrieval_matrix_paired_p0 import metrics
        images=np.tile(np.array([[1,0]],dtype=np.float32),(4,1));texts=[];labels=[]
        for i in range(4):
            labels.append({'id':str(i),'benchmark':'sugarcrepe_pp','category':'replace_att' if i==0 else 'replace_obj','text_start':i*3,'text_options':3})
            texts.extend([[1,0],[1,0],[-1,0]] if i==0 else [[1,0],[1,0],[1,0]])
        texts=np.array(texts,dtype=np.float32)
        scores=metrics.score_rows(images,texts,labels,device='cpu')
        self.assertEqual(scores.dtype,np.float32);self.assertEqual(scores.shape,(4,6))
        self.assertEqual(metrics.from_scores(scores,labels),metrics.independent_metrics(images,texts,labels))
        self.assertEqual(metrics.from_scores(scores,labels)['replace_ITT'],.5)
        self.assertFalse(metrics.events(scores,labels)[1:].any())
        image=np.array([[-1,0]],dtype=np.float32);options=np.array([[1,0],[1,0],[-1,0]],dtype=np.float32)
        different=metrics.score_rows(image,options,[{**labels[0],'text_start':0}],device='cpu')
        self.assertEqual(metrics.events(different,[labels[0]]).tolist(),[[False,True]])

    def test_native_mcq_axes_and_missing_published_targets_remain_distinct(self):
        from codex.retrieval_matrix_rerank import metrics
        from codex.retrieval_matrix_rerank_native_p0 import contracts as N
        values=[.9,.7,.2,.6]
        self.assertEqual(metrics.hit('cola_text_choice',values),1);self.assertEqual(metrics.hit('cola_image_choice',values),0)
        self.assertEqual(metrics.hit('scpp',[.5,.5,.5]),0)
        pp=self.inventory['profiles']['native-rerank-qwen2']['producer_plan']['path'];plan=json.loads(Path(pp).read_text())
        subsets={name:{'rows':count,'accuracy':.5} for name,count in N.COUNTS.items()}
        names={'cola_text_choice':['cola'],'scpp':list(N.SCPP),'negbench_mcq':['coco_mcq','voc_mcq']}
        report={'rows':15912,'subsets':subsets,'benchmarks':{k:{'rows':sum(N.COUNTS[n] for n in v),'subsets':sorted(v),'macro_accuracy':.5} for k,v in names.items()}}
        for key in self.inventory['excluded_missing_targets']:
            self.assertEqual(N.assess(report,plan['variants'][key])['state'],'TARGETS_INCOMPLETE')

    def test_shared_frozen_reader_uses_only_isolated_synthetic_prereg_and_anchor(self):
        with tempfile.TemporaryDirectory(dir=C.ROOT/'hf_home/tmp',prefix='native-analysis-reader-') as d:
            root=Path(d);prereg=root/'PREREG.md';prereg.write_bytes(b'synthetic criteria\n'+C.MARKER)
            plan=root/'synthetic-plan.json';plan.write_text('{"synthetic":true}\n');artifact=root/'synthetic-scalar.npy';artifact.write_bytes(b'synthetic')
            anchor=root/'anchor'
            with C.synthetic_read_session(prereg,anchor,plan,[artifact]):
                self.assertIn(b'READ_STARTED',prereg.read_bytes());artifact.read_bytes()
            self.assertIn(b'READ_COMPLETED',prereg.read_bytes());self.assertTrue(list(anchor.rglob('journal.jsonl')))
            with self.assertRaisesRegex(ValueError,'isolated synthetic'):
                C.synthetic_read_session(prereg,anchor,plan,[C.ROOT/'dl/forbidden-production-array.npy'])


if __name__=='__main__':unittest.main()
