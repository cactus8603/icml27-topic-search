"""[codex] Native image+metadata encoding through the pinned CORE fused path."""
from __future__ import annotations

from codex.retrieval_matrix import models as M


class NativeOfficialModel:
    def __init__(self, spec, *, device='cuda'):
        if spec.get('variant') in {'gme-qwen2', 'vlm2vec-v2-dora'}:
            from codex.retrieval_matrix_qwen2.adapters import Qwen2OfficialModel
            self.backend = Qwen2OfficialModel(spec, device=device)
        else:
            self.backend = M.OfficialModel(spec, device=device)
        self.spec = spec

    @property
    def fused_dispatch(self):
        name = self.spec['official']['model_name']
        # CORE i2t_emb names VLM2Vec, while its otherwise identical generic
        # Qwen2 fused_emb dispatch omits that name. Keep the loaded full DoRA
        # model and processor; explicitly select the existing generic branch.
        return 'gme' if self.spec.get('variant') == 'vlm2vec-v2-dora' else name

    def encode_queries(self, items):
        return self.backend.encode([row['text'] for row in items])

    def encode_candidates(self, items):
        import torch
        if not items or any(not all(key in row for key in ('id', 'path', 'text')) for row in items):
            raise ValueError('Every fused item must carry its complete product id/path/text identity')
        with torch.inference_mode():
            value = self.backend.embedding.fused_emb(
                self.fused_dispatch, [self.backend.model, self.backend.processor],
                texts=[row['text'] for row in items], images=[row['path'] for row in items])
            if value.dtype != self.backend.dtype:
                raise ValueError('Released fused encoder returned a dtype different from the frozen native model dtype')
            value = torch.nn.functional.normalize(value, dim=-1)
        return value.float().cpu().numpy()

    def signature(self):
        return {**self.backend.signature(), 'native_input': 'each product image plus its own metadata',
                'fused_dispatch': self.fused_dispatch,
                'dispatch_disclosure': ('CORE fused_emb generic Qwen2 branch selected as gme for the fully '
                    'loaded/merged VLM2Vec-V2 DoRA; release omitted vlm2vec in this dispatch condition')
                    if self.spec.get('variant') == 'vlm2vec-v2-dora' else None,
                'fused_max_length': 4096, 'fused_truncation': False,
                'normalization': 'torch.nn.functional.normalize in original native model dtype before float32 checkpoint'}
