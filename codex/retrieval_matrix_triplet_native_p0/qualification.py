"""[codex] Fail closed: source preparation cannot create native qualifications."""
from codex.retrieval_matrix_triplet_native_p0 import contracts as C,build_plans

SCOPE='official-triplet-native-p0-qualified'


def entry():return C.qualification_contract()


def admit(receipt_entry,model_spec):
    # Do not even open a claimed PASS or its arrays before the protocol/source
    # gate is resolved. No old SigLIP/paired/CORE receipt can fill this gap.
    key=model_spec.get('variant')
    if key not in C.VARIANTS or C.digest(model_spec)!=C.digest(C.model(key)):
        raise ValueError('Triplet native admission requires its exact independent checkpoint variant')
    C.require_executable(build_plans.build(key))


def review(*args,**kwargs):
    raise ValueError('No Triplet native independent numerical review before author protocol/source freeze')
