"""[codex] Build P2 only after genuine completed, independently reviewed P0."""
import argparse
from pathlib import Path
import json
import sys

ROOT=Path(__file__).resolve().parents[2]
HERE=Path(__file__).resolve().parent
sys.path.insert(0,str(ROOT))
from codex.retrieval_matrix_panel import qualification as Q
from codex.retrieval_matrix_panel import data_audit


def build(base_path,models,receipts,audit_path):
    base=Q.read(base_path)
    if base.get("state")!="FROZEN" or base.get("phase") not in {"P0","P0-Qwen2"}:
        raise ValueError("P2 must extend a frozen official P0 plan")
    if set(models)!=set(receipts) or not models:
        raise ValueError("Each model requires its own qualified receipt")
    qualified={key:Q.entry(receipts[key]) for key in models}
    for key in models:
        Q.admit(qualified[key],base["models"][key])
    development=ROOT/"dl/retrieval-matrix/data/coco-development-v2"
    manifest=Q.read(development/"manifest.json")
    images=Q.read(development/"images-summary.json")
    data_audit.verify(audit_path,development/"manifest.json",HERE/"development-sources.json",
                      development/"images-summary.json",development/"dev/images-receipt.jsonl")
    if manifest["conjunction"]["queries"]<1000:
        raise ValueError("Formal conjunction development requires at least 1000 queries")
    datasets={}
    for name,folder,meta in (("conjunction-dev","dev","conjunction"),("negation-dev","negation","negation")):
        path=development/folder
        for file,expected in manifest[meta]["outputs"].items():
            if Q.sha(path/file)!=expected: raise ValueError("Development metadata changed")
        datasets[name]={"format":"conjunction","split":"dev","gallery":Q.entry(path/"gallery.json"),
            "queries":Q.entry(path/"queries.jsonl"),"images_receipt":Q.entry(development/"dev/images-receipt.jsonl"),
            "images_dir":str(development/"dev/images"),"expected_gallery":20000,
            "expected_queries":manifest[meta]["queries"],"expected_eligible":manifest[meta]["queries"],
            "scope":"formal conjunction development" if folder=="dev" else "negation diagnostic only; insufficient 1000-query power"}
    cola=ROOT/"dl/retrieval-matrix/data/cola"
    for subset,gallery,queries in (("gqa",2589,320),("paco",7921,400)):
        datasets["cola-"+subset]={"format":"cola_single","split":"public","subset":subset,
            "parquet":Q.entry(cola/"data"/f"singleobjects_{subset}.parquet"),
            "vocabulary":Q.entry(cola/"data"/f"singleobjects_{subset}_labels.json"),
            "images_receipt":Q.entry(cola/"file_receipt.jsonl"),"image_root":str(cola),
            "allowed_shared_image_roots":[str(ROOT/"dl/retrieval3090/data/coco2017/val2017"),
                                          str(ROOT/"dl/claude-panel/conj-panel-v1/dev/images")],
            "expected_gallery":gallery,"expected_queries":queries,"expected_eligible":queries,
            "use":"fixed baseline test only; no fitting or method selection"}
    sources={}
    for relative,expected in base["source_files"].items():
        actual=Q.sha(ROOT/relative)
        # The P0 preregistration gains mandatory read-time entries after P0.
        # Executing code/configuration must retain the qualified P0 bytes.
        if actual!=expected and not relative.endswith("/PREREG.md"):
            raise ValueError("Qualified P0 source implementation changed: "+relative)
        sources[relative]=actual
    for path in sorted(HERE.iterdir()):
        if path.suffix in {".py",".md",".sh"}: sources[str(path.relative_to(ROOT))]=Q.sha(path)
    sources["codex/retrieval_matrix_qwen2/adapters.py"]=Q.sha(ROOT/"codex/retrieval_matrix_qwen2/adapters.py")
    return {**base,"phase":"P2-qualified-fixed-baselines","scope":"qualified published fixed baselines on isolated development and COLA test",
        "models":{key:base["models"][key] for key in models},"datasets":datasets,
        "source_files":sources,"prereg":"codex/retrieval_matrix_panel/PREREG.md",
        "qualification":qualified,"development_audit":Q.entry(audit_path),"fixed_baseline_test":False,
        "development_manifest":Q.entry(development/"manifest.json"),
        "development_sources":Q.entry(HERE/"development-sources.json"),
        "development_images_summary":Q.entry(development/"images-summary.json"),
        "development_images_receipt":Q.entry(development/"dev/images-receipt.jsonl"),
        "parent_p0_plan":Q.entry(base_path),"not_covered":["MCMR separate fused metadata runner","custom methods","P3 rerank"],
        "disclosure":"COCO dev v2 excludes published PACO/Karpathy/Half-Truths and v1 test image IDs; historical v1 exposure preserved. COLA captions are a fixed disclosed adaptation, not an exact original model reproduction."}


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument("--base-plan",required=True,type=Path)
    p.add_argument("--qualification",action="append",required=True,help="MODEL=completed qualification JSON")
    p.add_argument("--development-audit",required=True,type=Path)
    p.add_argument("--out",required=True,type=Path)
    a=p.parse_args()
    if a.out.exists(): raise ValueError("Never overwrite an existing P2 plan")
    receipts=dict(row.split("=",1) for row in a.qualification)
    if len(receipts)!=len(a.qualification): raise ValueError("Each qualified model must be declared once")
    result=build(a.base_plan,list(receipts),receipts,a.development_audit)
    a.out.write_text(json.dumps(result,sort_keys=True,indent=2)+"\n")
    print(json.dumps({"state":"PREPARED_REQUIRES_COMMIT","path":str(a.out),"sha256":Q.sha(a.out)}))


if __name__=="__main__": main()
