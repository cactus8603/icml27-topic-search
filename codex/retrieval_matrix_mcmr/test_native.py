"""[codex] CPU integrity tests for complete-product native MCMR execution."""
import json
from pathlib import Path
from types import SimpleNamespace
import tempfile
import unittest
from unittest.mock import patch

import numpy as np

from codex.retrieval_matrix_mcmr import runner as N
from codex.retrieval_matrix_mcmr.adapters import NativeOfficialModel


class NativeTests(unittest.TestCase):
    def setUp(self):
        scratch = N.ROOT / 'hf_home/tmp'
        scratch.mkdir(exist_ok=True, parents=True)
        self.temp = tempfile.TemporaryDirectory(dir=scratch)
        self.root = Path(self.temp.name)

    def tearDown(self):
        self.temp.cleanup()

    def entry(self, path):
        return {'path': str(path), 'sha256': N.R.sha(path)}

    def fixture(self):
        from PIL import Image
        image = self.root / 'images/shared.jpg'
        image.parent.mkdir()
        Image.new('RGB', (8, 8), 'blue').save(image)
        receipt = self.root / 'images-receipt.jsonl'
        receipt.write_text(json.dumps({'file': image.name, 'sha256': N.R.sha(image), 'bytes': image.stat().st_size}) + '\n')
        corpus = self.root / 'corpus.jsonl'
        products = [{'id': 'product-a', 'image': str(image), 'text': 'title: first\u0085product'},
                    {'id': 'product-b', 'image': str(image), 'text': 'title: different metadata'}]
        with corpus.open('w') as stream:
            for product in products:
                stream.write(json.dumps(product, ensure_ascii=False) + '\n')
        queries = self.root / 'queries.jsonl'
        queries.write_text(json.dumps({'id': 'q1', 'text': 'Find the first product.'}) + '\n')
        labels = self.root / 'instances.jsonl'
        labels.write_text('UNREADABLE LABELS MUST NOT ENTER INFERENCE\n')
        manifest = self.root / 'manifest.json'
        manifest.write_text(json.dumps({'adapter_version': 'mcmr-mmeb-v3-native-v1',
            'corpus_admission_uses_qrel_or_labels': False,
            'full_image_receipt_sha256': N.R.sha(receipt),
            'outputs': {'corpus.jsonl': N.R.sha(corpus), 'queries.jsonl': N.R.sha(queries)}}))
        return {'native_manifest': self.entry(manifest), 'corpus': self.entry(corpus),
                'queries': self.entry(queries), 'instances': self.entry(labels),
                'images_receipt': self.entry(receipt), 'images_dir': str(image.parent),
                'expected_gallery': 2, 'expected_queries': 1}

    def test_shared_image_preserves_distinct_metadata_and_unicode_records(self):
        bundle = N.load_data(self.fixture(), verify_images=True)
        self.assertEqual(len(bundle['gallery']), 2)
        left, right = bundle['gallery']
        self.assertEqual(left['path'], right['path'])
        self.assertNotEqual(left['id'], right['id'])
        self.assertNotEqual(left['text'], right['text'])
        self.assertIn('\u0085', left['text'])

    def test_qrel_is_not_opened_by_inference_loader(self):
        spec = self.fixture()
        Path(spec['instances']['path']).unlink()
        # Even a nonexistent qrel does not affect candidate admission.
        self.assertEqual(len(N.load_data(spec)['gallery']), 2)

    def test_changed_metadata_invalidates_checkpoint_identity(self):
        path = self.root / 'resume.npy'
        items = [{'id': 'a', 'path': 'same.jpg', 'text': 'first'},
                 {'id': 'b', 'path': 'same.jpg', 'text': 'second'}]
        calls = []
        def interrupt(batch):
            calls.extend(batch)
            if batch[0]['id'] == 'b':
                raise RuntimeError('simulated interruption')
            return np.array([[1., 0.]], dtype=np.float32)
        with self.assertRaises(RuntimeError):
            N.R.checkpoint(path, items, 1, interrupt, {'run': 'fixed'})
        changed = [{**items[0], 'text': 'changed first metadata'}, items[1]]
        with self.assertRaises(ValueError):
            N.R.checkpoint(path, changed, 1, lambda batch: np.array([[1., 0.]], dtype=np.float32), {'run': 'fixed'})
        self.assertEqual(len(calls), 2)

    def test_candidate_encoder_passes_both_distinct_texts_and_same_image(self):
        import torch
        captured = {}
        def fused(name, models, *, texts, images):
            captured.update(name=name, texts=texts, images=images)
            return torch.tensor([[3., 4.], [0., 2.]], dtype=torch.bfloat16)
        model = object.__new__(NativeOfficialModel)
        model.spec = {'variant': 'vlm2vec-v2-dora', 'official': {'model_name': 'vlm2vec'}}
        model.backend = SimpleNamespace(embedding=SimpleNamespace(fused_emb=fused), model=object(),
                                        processor=object(), dtype=torch.bfloat16)
        result = model.encode_candidates([{'id': 'a', 'path': 'same.jpg', 'text': 'first'},
                                          {'id': 'b', 'path': 'same.jpg', 'text': 'second'}])
        self.assertEqual(captured, {'name': 'gme', 'texts': ['first', 'second'], 'images': ['same.jpg', 'same.jpg']})
        self.assertEqual(result.dtype, np.float32)
        np.testing.assert_allclose(np.linalg.norm(result, axis=1), 1, atol=.01)

    def test_native_dtype_mismatch_fails(self):
        import torch
        model = object.__new__(NativeOfficialModel)
        model.spec = {'official': {'model_name': 'vlembed'}}
        model.backend = SimpleNamespace(embedding=SimpleNamespace(fused_emb=lambda *a, **k: torch.ones((1, 2))),
                                        model=object(), processor=object(), dtype=torch.bfloat16)
        with self.assertRaisesRegex(ValueError, 'dtype'):
            model.encode_candidates([{'id': 'a', 'path': 'same.jpg', 'text': 'first'}])

    def test_mrr10_and_fractional_recall_have_complete_denominators(self):
        ids = [f'p{i}' for i in range(100)]
        row = np.r_[np.arange(100), np.arange(100, 0, -1)].astype(np.float32)
        metrics, details = N.ordinal_metrics(np.stack([row, row, row]), ids,
            [{'id': 'q1', 'positives': ['p0', 'p1']}, {'id': 'q2', 'positives': ['p9']},
             {'id': 'q3', 'positives': ['p10']}])
        self.assertAlmostEqual(metrics['Hit@1'], 1/3)
        self.assertAlmostEqual(metrics['Recall@1'], .5/3)
        self.assertAlmostEqual(metrics['MRR@10'], 1.1/3)
        self.assertEqual(details[2]['MRR@10'], 0)

    def test_product_byte_change_fails_image_receipt(self):
        spec = self.fixture()
        Path(spec['images_dir'], 'shared.jpg').write_bytes(b'changed')
        with self.assertRaisesRegex(ValueError, 'image differs'):
            N.load_data(spec, verify_images=True)

    def test_storage_chunk_keeps_separate_batch1_forward_and_product_metadata(self):
        calls=[]
        def encode(batch):
            calls.append(batch)
            return np.array([[len(batch[0]['text']),1]],dtype=np.float32)
        products=[{'id':'a','path':'same.jpg','text':'first'},
                  {'id':'b','path':'same.jpg','text':'different metadata'}]
        values=N.encode_product_chunk(SimpleNamespace(encode_candidates=encode),products)
        self.assertEqual(calls,[[products[0]],[products[1]]])
        np.testing.assert_array_equal(values[:,0],[5,18])

    def test_topk_uses_original_one_dimensional_call_shape(self):
        import torch
        queries=torch.tensor([[1.,0.],[0.,1.]],dtype=torch.bfloat16)
        corpus=torch.tensor([[1.,0.],[0.,1.],[1.,1.]],dtype=torch.bfloat16)
        original=torch.topk
        dimensions=[]
        def topk(value,*args,**kwargs):
            dimensions.append(value.ndim)
            return original(value,*args,**kwargs)
        with patch.object(torch,'topk',side_effect=topk):
            result=N.top100_encode(queries,corpus,[0,1],torch.bfloat16)
        self.assertEqual(dimensions,[1,1])
        self.assertEqual(result.shape,(2,6))


if __name__ == '__main__':
    unittest.main()
