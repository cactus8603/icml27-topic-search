"""[codex] CPU reference only; preserve upstream names and torch.topk policy."""
import importlib.util
from codex.retrieval_matrix_triplet_native_p0 import contracts as C


def author_module():
    C.source_manifest()
    path=C.BENCHMARK/'clip_benchmark/metrics/zeroshot_retrieval.py'
    spec=importlib.util.spec_from_file_location('triplet_laion_pinned_cpu_reference',path)
    module=importlib.util.module_from_spec(spec);spec.loader.exec_module(module)
    return module


def cpu_reference(images,texts,positive_image_indices,*,batch_size=64,ks=(1,5,10)):
    """Not a GPU tie-equivalence claim and not an admitted native benchmark."""
    import torch
    image=torch.as_tensor(images);text=torch.as_tensor(texts)
    positive=torch.as_tensor(positive_image_indices,dtype=torch.long)
    if (image.dtype!=torch.float32 or text.dtype!=torch.float32 or image.ndim!=2 or text.ndim!=2
            or image.shape[1]!=text.shape[1] or len(positive)!=len(text)
            or not torch.isfinite(image).all() or not torch.isfinite(text).all()
            or positive.min()<0 or positive.max()>=len(image) or set(positive.tolist())!=set(range(len(image)))):
        raise ValueError('Complete normalized FP32 image/text arrays and positive mapping required')
    if any(not torch.allclose(torch.linalg.vector_norm(value,dim=-1),torch.ones(len(value)),atol=1e-5) for value in (image,text)):
        raise ValueError('Reference embeddings must retain declared normalization')
    if type(batch_size)is not int or batch_size<1:raise ValueError('Explicit positive batch size required')
    if any(type(k)is not int or k<1 or k>min(len(image),len(text)) for k in ks):raise ValueError('Recall K exceeds original candidate universe')
    scores=text@image.T;pairs=torch.zeros_like(scores,dtype=torch.bool)
    pairs[torch.arange(len(text)),positive]=True;official=author_module();output={}
    for k in ks:
        output['image_retrieval_recall@'+str(k)]=(official.batchify(official.recall_at_k,scores,pairs,batch_size,'cpu',k=k)>0).float().mean().item()
        output['text_retrieval_recall@'+str(k)]=(official.batchify(official.recall_at_k,scores.T,pairs.T,batch_size,'cpu',k=k)>0).float().mean().item()
    return output
