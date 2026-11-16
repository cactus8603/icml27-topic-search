"""[codex] Three new unreviewed paired runtime plans; original science unchanged."""
from copy import deepcopy
from pathlib import Path
import argparse
import sys
ROOT=Path(__file__).resolve().parents[2];sys.path.insert(0,str(ROOT))
from codex.retrieval_matrix_paired_runtime_fix import contracts as C


def build(key):
    C.write(C.HERE/'dependencies.json',C.dependency_spec())
    result=deepcopy(C.parent(key));result.update(state='REVIEW_REQUIRED',reviewed=False,runtime_batch=C.BATCH,
        dependencies=C.entry(C.HERE/'dependencies.json'))
    result['runtime_fix']=C.runtime_identity(result)
    result['source_files'].update({str((C.HERE/name).relative_to(ROOT)):C.sha(C.HERE/name) for name in C.OWN})
    result['source_files'][C.OVERLAY_REL]=C.sha(ROOT/C.OVERLAY_REL)
    C.validate(result);return result


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--model',choices=C.MODELS,required=True);p.add_argument('--out',type=Path,required=True);a=p.parse_args()
    if a.out.exists():raise ValueError('Never overwrite a planned execution')
    value=build(a.model);C.write(a.out,value);print({'model':a.model,'state':value['state'],'sha256':C.sha(a.out)})
