"""[codex] Three literal Triplet author variants; unresolved protocol is not PASS."""
from copy import deepcopy
from pathlib import Path
import subprocess
from codex.retrieval_matrix_native_p0.evidence import checked,digest,entry,read,sha,write

ROOT=Path(__file__).resolve().parents[2];HERE=Path(__file__).resolve().parent
EXTRA_REF='e374bfc783a1d88244b7dfb5d5ddfc0de4969d0f'
OVERLAY_REF='99431594893bba94337b3befd10a7a483ba7374f'
NATIVE_REF='f3f787eeb193aad8dcc6f2ee43e9609ddf5b47eb'
BENCHMARK_REF='a23028288ed914efb31c5ff32a1e702123cbd0ff'
VARIANTS=('triplet_cc12m','triplet_openclip_finetune','negclip_cc12m')
SOURCES=ROOT/'dl/retrieval-matrix-triplet-native-p0/sources'
DATA=ROOT/'dl/retrieval-matrix-triplet-native-p0/data'
BENCHMARK=SOURCES/'clip_benchmark'
OWN=('contracts.py','build_plans.py','acquire_audit.py','metrics.py','preflight.py','qualification.py',
    'runner.py','test_triplet.py','central_task.sh','PREREG.md','README.md')
TABLE5={
    'triplet_cc12m':{'COCO':{'i2t_R@5':.33,'t2i_R@5':.285},'Flickr30k':{'i2t_R@5':.559,'t2i_R@5':.5238}},
    'negclip_cc12m':{'COCO':{'i2t_R@5':.3016,'t2i_R@5':.2311},'Flickr30k':{'i2t_R@5':.466,'t2i_R@5':.417}},
}


def frozen(relative,ref):
    path=(ROOT/relative).resolve()
    if not path.is_relative_to(ROOT):raise ValueError('Frozen source path escapes repository')
    if subprocess.check_output(['git','-C',str(ROOT),'show',ref+':'+str(path.relative_to(ROOT))])!=path.read_bytes():
        raise ValueError('Referenced frozen source changed: '+relative)
    return path


def registry():return read(frozen('codex/retrieval_matrix_extra/registry.json',EXTRA_REF))


def model(key):
    if key not in VARIANTS:raise ValueError('Only the three literal Triplet author variants, never original NegCLIP')
    value=deepcopy(registry()['models'][key])
    for name in ('adapters.py','preflight.py','dependencies.json'):
        frozen('codex/retrieval_matrix_extra/'+name,EXTRA_REF)
    return value


def source_manifest():
    receipt=read(BENCHMARK/'source-receipt.json')
    if receipt['commit']!=BENCHMARK_REF:raise ValueError('The disclosed pre-paper LAION reference changed')
    sources={}
    for relative,value in receipt['files'].items():
        path=BENCHMARK/relative
        if sha(path)!=value['sha256']:raise ValueError('Fixed upstream LAION source bytes changed')
        sources['laion/'+relative]=entry(path)
    for path in (BENCHMARK/'source-receipt.json',BENCHMARK/'commit-before-paper.json',
            SOURCES/'annotation-release-1.0.0.json',ROOT/'dl/retrieval-matrix-extra/sources/tripletclip-paper.pdf',
            ROOT/'dl/retrieval-matrix-extra/sources/tripletclip-paper.txt'):
        sources[str(path.relative_to(ROOT))]=entry(path)
    return sources


def dependency_reference():
    # Reuse the reviewed einops overlay, never alter the original extra deps.
    path=frozen('codex/retrieval_matrix_native_runtime_fix/dependencies.json',NATIVE_REF)
    frozen('codex/retrieval_matrix_qwen2_runtime_fix/overlay.json',OVERLAY_REF)
    return entry(path)


def gaps(key):
    result=[
        {'id':'author-eval-command','status':'UNRESOLVED','detail':'Author paper cites CLIP_benchmark but releases no retrieval command, exact version, AMP/noAMP, batch size or image/text batching for these checkpoints.'},
        {'id':'laion-version-proxy','status':'DISCLOSED_PROXY_NOT_AUTHOR_PIN','detail':'LAION commit '+BENCHMARK_REF+' is the latest commit before 2024-11-01, not a Triplet author pin.'},
        {'id':'annotation-payloads','status':'PENDING_ROOT_ACQUIRE_AUDIT','detail':'Use exact CLIP_benchmark release1.0.0 annotations; raw Karpathy25010 text/order cannot silently substitute.'},
        {'id':'precision-and-batches','status':'UNRESOLVED','detail':'FP32 image4/text32 are extra preparation settings. Official LAION default uses AMP and image batch64 with all captions in that image batch; neither is proven to be the author run.'},
        {'id':'dependency-versions','status':'DISCLOSED_ADAPTATION_REQUIRES_REVIEW','detail':'Author OpenCLIP training requirements pin open_clip_torch2.26.1/torch2.4.0; reviewed extra runtime uses2.31.0/torch2.6.0. No native equivalence is asserted.'},
    ]
    if key in TABLE5:
        result.append({'id':'table5-table13-direction','status':'UNRESOLVED','detail':'Identical R@5 values have opposite I2T/T2I headings in Table5 and Table13. Preserve both printed targets; no direction chosen using model outcomes.'})
        result.append({'id':'scratch-inference-port','status':'DISCLOSED_ADAPTATION_REQUIRES_REVIEW','detail':'Author HF separate encoder classes are available; frozen extra delegates processor/i2t to CORE. This is not evidence of the original Triplet retrieval invocation.'})
    return result


def require_executable(plan):
    if plan.get('model_key') not in VARIANTS:raise ValueError('Unsupported Triplet checkpoint identity')
    # A source change plus fresh independent review is required to resolve
    # these gaps. Editing a draft plan field cannot enable the launcher.
    raise ValueError('Triplet native P0 source review required; unresolved author protocol: '+', '.join(x['id'] for x in gaps(plan['model_key'])))


def qualification_contract():
    return {'scope':'official-triplet-native-p0-qualified','minimum_distinct_benchmarks':2,
        'canonical_benchmarks':['COCO','Flickr30k'],'subsets_or_directions_do_not_count_twice':True,
        'implementation_authors':['/root/matrix_runner'],
        'requires':['resolved independently frozen protocol and target direction','complete original annotation order/captions and all images',
            'exact variant/revision/weightSHA/model-spec/settings/sourceGit/ref/actual-module-runtime identity',
            'original complete embedding arrays with all checkpoint batch receipts and index receipt',
            'producer and different non-implementation numeric reviewer','independent original-array score/topk recomputation',
            'separate paper-target qualification and userdecision; raw recomputation PASS is insufficient'],
        'current_admission':'REFUSE_ALL_UNTIL_NEW_SOURCE_REVIEW'}
