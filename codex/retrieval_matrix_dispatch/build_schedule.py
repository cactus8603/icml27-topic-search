"""[codex] Preserve eight original queue commands; root may append MCMR before freeze."""
from pathlib import Path
import json
import sys

ROOT=Path(__file__).resolve().parents[2]
sys.path.insert(0,str(ROOT))
from codex.retrieval_matrix_dispatch.dispatch import sha,validate


def build():
    tasks=[]
    queues=[]
    groups=[("embedding","dl/retrieval-matrix/runs/p0-qwen3-batch1-20261116",["core2","qwen2","core8","qwen8"]),
            ("embedding","dl/retrieval-matrix/runs/p0-qwen2-batch1-20261116",["gme2","vlm2vec2"]),
            ("mcq","dl/retrieval-matrix-rerank/runs/p0-core-mcq-batch1-20261116",["core2","core8"])]
    for group,folder,keys in groups:
        for key in keys:
            path=ROOT/folder/(key+"-queue.json")
            queue=json.loads(path.read_text())
            command=queue["command"]
            launcher,commit,plan,model,output=command[1:]
            completion=([str(Path(output)/part/"index-meta.json") for part in ("coco","negbench","flickr")]
                        if group=="embedding" else [str(Path(output)/"main/complete.json")])
            tasks.append({"id":f"{group}-{key}","model":model,"command":command,"cwd":queue["cwd"],
                "launcher":launcher,"launcher_sha256":sha(launcher),"freeze_commit":commit,"plan":plan,
                "plan_sha256":sha(plan),"output":output,"completion":completion,
                "min_free_mib":20480 if key.endswith("8") else 8192,"original_queue":str(path),
                "original_queue_pid":queue["queue_pid"],"priority":len(tasks)})
            queues.append(str(path))
    # Prefer all requested 2B capacity classes, preserving their internal source
    # order. No task/model command or original protocol is rewritten.
    tasks.sort(key=lambda task:(task["min_free_mib"],task["priority"]))
    schedule={"schema":1,"owner":"[codex]","state":"DRAFT","max_live":2,
        "confirmation_seconds":5,"max_existing_compute_pids":1,"utilization_limit":None,
        "gpu_lock_dir":str(ROOT/"dl/retrieval3090/runs/.gpu-locks"),
        "shared_log":str(ROOT/"killtests/SUBMITTED.log"),"legacy_queues":queues,"tasks":tasks,
        "extension":"Root may append fixed MCMR tasks before first freeze; all tasks share this same cap2 and lock directory",
        "policy":"No automatic retry of failed/cancelled/uncertain launches; no changes to frozen model parameters; no foreign process cancellation"}
    validate(schedule)
    path=Path(__file__).with_name("schedule.json")
    path.write_text(json.dumps(schedule,indent=2,ensure_ascii=False)+"\n")
    print("Draft",len(tasks),"unchanged frozen commands",path)


if __name__=="__main__":
    build()
