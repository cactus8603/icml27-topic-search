"""[codex] Exact original class/encode; stabilize only signature repr addresses."""
from copy import deepcopy
import re
from types import MethodType
from codex.retrieval_matrix_paired_p0 import models as ORIGINAL


def stable_signature(signature):
    value=deepcopy(signature)
    # Keep function name, transform class/parameters and every other byte.
    value['transform']=re.sub(r'(<function [^<>]+) at 0x[0-9a-fA-F]+>',r'\1>',value['transform'])
    return value


def prepare(model):
    original=model.signature
    def signature(self):return stable_signature(original())
    model.signature=MethodType(signature,model)
    return model


def create(spec,device='cuda'):
    model=ORIGINAL.PairedModel(spec,device=device)
    if type(model) is not ORIGINAL.PairedModel:raise ValueError('Keep the exact original paired wrapper class')
    return prepare(model)
