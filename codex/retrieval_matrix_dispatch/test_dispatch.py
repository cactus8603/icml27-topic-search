"""[codex] Stdlib synthetic admission/lifecycle tests, never a real GPU."""
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

from codex.retrieval_matrix_dispatch import dispatch as D


def gpu(index,free=9000,pids=(100,)):
    return {"index":index,"uuid":f"GPU-synthetic-{index}","free_mib":free,"utilization_percent":100,"compute_pids":list(pids)}


def schedule(root,count=3):
    tasks=[{"id":f"task-{i}","model":"core2","min_free_mib":8192,"command":["literal","arg;not-a-shell"],
            "freeze_commit":"a"*40,"output":str(root/f"out-{i}"),"cwd":str(root),"completion":[str(root/f"complete-{i}.json")],"plan_sha256":"b"*64}
           for i in range(count)]
    return {"schema":1,"owner":"[codex]","max_live":2,"confirmation_seconds":5,"max_existing_compute_pids":1,
            "gpu_lock_dir":str(root/"locks"),"shared_log":str(root/"submitted.log"),"tasks":tasks}


class FakeChild:
    pid=999999999
    def wait(self):return 0


class DispatchTests(unittest.TestCase):
    def setUp(self):
        tmp=D.ROOT/"hf_home/tmp"
        tmp.mkdir(parents=True,exist_ok=True)
        self.temp=tempfile.TemporaryDirectory(prefix="dispatcher-synthetic-",dir=tmp)
        self.root=Path(self.temp.name)
        self.instances=[]

    def tearDown(self):
        for obj in self.instances:
            for lock in obj.leases.values():lock.__exit__(None,None,None)
        self.temp.cleanup()

    def dispatcher(self,rows,count=3,query=None):
        self.time=0
        self.launched=[]
        def spawn(cmd,**kwargs):
            self.launched.append((cmd,kwargs))
            return FakeChild()
        obj=D.Dispatcher(schedule(self.root,count),"c"*64,self.root/"runtime",query=query or (lambda:rows),
                         clock=lambda:self.time,popen=spawn,complete=lambda task:False)
        self.instances.append(obj)
        return obj

    def test_capacity_and_existing_pid_count_ignore_utilization(self):
        task={"min_free_mib":8192}
        self.assertTrue(D.eligible(gpu(0,8192,pids=(1,)),task))
        self.assertFalse(D.eligible(gpu(0,8191,pids=()),task))
        self.assertFalse(D.eligible(gpu(0,24000,pids=(1,2)),task))
        self.assertFalse(D.eligible(gpu(0,20479,pids=()),{"min_free_mib":20480}))
        self.assertEqual(D.parse_processes("GPU-test, 1\nGPU-test, 1\nGPU-test, 2"),{"GPU-test":[1,2]})
        with self.assertRaises(ValueError):D.parse_processes("GPU-test, N/A")

    def test_two_five_second_samples_and_global_cap_two(self):
        obj=self.dispatcher([gpu(i) for i in range(4)])
        obj.tick();self.assertEqual(len(self.launched),0)
        self.time=4.99;obj.tick();self.assertEqual(len(self.launched),0)
        self.time=5;obj.tick();self.assertEqual(len(self.launched),2)
        self.time=100;obj.tick();self.assertEqual(len(self.launched),2)
        self.assertEqual(obj.live_count(),2)
        self.assertNotEqual(obj.state["tasks"]["task-0"]["gpu"]["uuid"],obj.state["tasks"]["task-1"]["gpu"]["uuid"])
        self.assertTrue(all(item[1]["shell"] is False and item[1]["pass_fds"] for item in self.launched))

    def test_gpu_lock_respects_another_cooperating_job(self):
        obj=self.dispatcher([gpu(0),gpu(1)],count=1)
        with D.FileLock(self.root/"locks/GPU-synthetic-0.lock"):
            obj.tick();self.time=5;obj.tick()
        self.assertEqual(obj.state["tasks"]["task-0"]["gpu"]["uuid"],"GPU-synthetic-1")

    def test_last_second_pid_or_capacity_change_restarts_confirmation(self):
        samples=iter([[gpu(0)],[gpu(0)],[gpu(0,pids=(1,2))],[gpu(0)],[gpu(0)],[gpu(0)]])
        obj=self.dispatcher([],count=1,query=lambda:next(samples))
        obj.tick();self.time=5;obj.tick();self.assertEqual(len(self.launched),0)
        self.time=10;obj.tick();self.assertEqual(len(self.launched),0)
        self.time=15;obj.tick();self.assertEqual(len(self.launched),1)

    def test_failed_task_is_never_retried_even_after_restart(self):
        obj=self.dispatcher([gpu(0)],count=1)
        obj.tick();self.time=5;obj.tick()
        row=obj.state["tasks"]["task-0"]
        Path(row["receipt"]).write_text(json.dumps({"task_id":"task-0","schedule_sha256":"c"*64,"token":row["token"],"status":"failed","returncode":1}))
        obj.reconcile();self.assertEqual(row["status"],"failed")
        self.time=10;obj.tick();self.assertEqual(len(self.launched),1)
        again=D.Dispatcher(obj.schedule,"c"*64,self.root/"runtime",query=lambda:[gpu(0)],clock=lambda:100,
                           popen=lambda *_a,**_k:self.fail("failed task retried"),complete=lambda _:False)
        self.instances.append(again);again.tick()

    def test_complete_outputs_skip_without_model_launch_and_identity_mismatch_fails(self):
        task=schedule(self.root,1)["tasks"][0]
        Path(task["completion"][0]).write_text(json.dumps({"state":"COMPLETE","identity":{"model":"core2","freeze_commit":"a"*40,"plan_sha256":"b"*64}}))
        self.assertTrue(D.completed_output(task))
        task["plan_sha256"]="d"*64
        with self.assertRaises(ValueError):D.completed_output(task)

    def test_scheduler_singleton_and_pid_reuse_protection(self):
        with D.FileLock(self.root/"central.lock"):
            with self.assertRaises(BlockingIOError):
                with D.FileLock(self.root/"central.lock"):pass
        token=D.process_token(__import__("os").getpid())
        self.assertTrue(D.alive(token))
        self.assertFalse(D.alive({**token,"start_ticks":token["start_ticks"]+1}))

    def test_best_fit_preserves_the_only_large_gpu_for_eight_b(self):
        obj=self.dispatcher([gpu(0,24000),gpu(1,9000)],count=1)
        obj.tick();self.time=5;obj.tick()
        self.assertEqual(obj.state["tasks"]["task-0"]["gpu"]["index"],1)

    def test_state_error_after_popen_keeps_live_lease_and_never_replays_attempt(self):
        obj=self.dispatcher([gpu(0)],count=1)
        obj.tick();self.time=5
        original=obj.save
        failed=[]
        def save():
            if self.launched and not failed:
                failed.append(True)
                raise OSError("synthetic state write error after successful Popen")
            original()
        obj.save=save
        with self.assertRaises(OSError):obj.tick()
        self.assertEqual(obj.state["tasks"]["task-0"]["status"],"uncertain")
        with self.assertRaises(BlockingIOError):
            with D.FileLock(self.root/"locks/GPU-synthetic-0.lock"):pass
        self.time=10;obj.tick();self.assertEqual(len(self.launched),1)

    def test_worker_freeze_failure_is_terminal_without_launch(self):
        task=schedule(self.root,1)["tasks"][0]
        receipt=self.root/"attempt.json"
        config=self.root/"worker.json"
        config.write_text(json.dumps({"task":task,"gpu":gpu(0),"receipt":str(receipt),"schedule_sha256":"c"*64,
            "token":"synthetic-token","journal":str(self.root/"events.jsonl")}))
        fd=__import__("os").open(self.root/"fake-fd",__import__("os").O_RDWR|__import__("os").O_CREAT,0o600)
        with patch.object(D,"worker_gate",side_effect=ValueError("synthetic freeze mismatch")),patch.object(D.subprocess,"Popen",side_effect=AssertionError("must not launch")):
            self.assertEqual(D.worker(config,fd),1)
        self.assertEqual(json.loads(receipt.read_text())["status"],"failed")


if __name__=="__main__":unittest.main()
