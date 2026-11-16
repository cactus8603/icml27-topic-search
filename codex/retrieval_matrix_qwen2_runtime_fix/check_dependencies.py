"""[codex] Verify original dependencies plus the isolated missing package."""
from pathlib import Path
import argparse
import sys
ROOT=Path(__file__).resolve().parents[2]
sys.path.insert(0,str(ROOT))
from codex.retrieval_matrix_qwen2_runtime_fix import contracts as C

def main():
    p=argparse.ArgumentParser(description=__doc__);p.add_argument("--plan",type=Path,required=True);a=p.parse_args()
    plan=C.read(a.plan);manifest=C.validate(plan)
    from transformers import BloomPreTrainedModel,Qwen2VLForConditionalGeneration
    import torch
    print({"owner":"[codex]","state":"PASS","original_dependencies":len(plan["dependencies"]["files"]),
        "overlay_files":len(manifest["files"]),"actual_imports":len(manifest["modules"]),
        "bloom_and_qwen2_classes_imported":True,"cuda_initialized":torch.cuda.is_initialized()})

if __name__=="__main__":main()
