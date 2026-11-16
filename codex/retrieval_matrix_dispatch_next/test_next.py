"""[codex] CPU-only synthetic adoption, immutable additions and parent signals."""
import copy
import fcntl
import json
import os
from pathlib import Path
import signal
from types import SimpleNamespace
import tempfile
import unittest
from unittest.mock import patch

from codex.retrieval_matrix_dispatch import dispatch as canonical
from codex.retrieval_matrix_dispatch_next import controller as C, build_schedule as S, handoff as H
B = C.B
ROOT = C.ROOT


def dump(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value))


class Fixture(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory(dir=ROOT/"hf_home/tmp", prefix="next-central-test-")
        self.addCleanup(self.tmp.cleanup)
        self.root = Path(self.tmp.name)
        self.old, self.earlier, self.new = [self.root/key for key in ("old", "earlier", "new")]
        self.old.mkdir(); self.earlier.mkdir()
        self.tasks = [self.task(key) for key in ("mcq-core2", "fixed-core2", "already-failed", "mcmr-core2")]
        self.old_path = self.old/"schedule.json"
        dump(self.old_path, {"schema": 1, "max_live": 2, "tasks": self.tasks})
        self.schedule = {"schema": 1, "owner": "[codex]", "state": "FROZEN", "max_live": 4,
            "confirmation_seconds": 5, "max_existing_compute_pids": 1,
            "controller": "immutable-additive-next-v1", "tasks": copy.deepcopy(self.tasks),
            "gpu_lock_dir": str(self.root/"locks"), "shared_log": str(self.root/"submitted.log"),
            "adoption": {"schedule": C.entry(self.old_path), "freeze_commit": "a"*40,
                "runtime": str(self.old), "retire_pending": []}, "additions": {}, "resource_exceptions": {}}
        self.new_path = self.root/"new-schedule.json"
        self.parent = {"pid": 10, "start_ticks": 10, "boot_id": "synthetic"}
        dump(self.new_path, self.schedule)
        self.config = {"schema": 1, "owner": "[codex]", "old": {
            "schedule": C.entry(self.old_path), "freeze_commit": "a"*40, "runtime": str(self.old),
            "controller": str(C.V3.HERE/"controller.py"), "python": "python3", "expected_token": self.parent},
            "new": {"schedule": C.entry(self.new_path), "freeze_commit": "b"*40, "runtime": str(self.new),
                "controller": str(C.HERE/"controller.py"), "python": "python3"}}
        self.previous = {"schema": 1, "scheduler": self.parent, "schedule_sha256": B.sha(self.old_path), "tasks": {
            "mcq-core2": self.running(self.tasks[0], self.earlier, 100, 101, "GPU-old"),
            "fixed-core2": self.running(self.tasks[1], self.old, 200, 201, "GPU-new"),
            "already-failed": {"status": "failed", "error": "preserved failure", "attempts": [1]},
            "mcmr-core2": {"status": "pending"}}}
        dump(self.old/"state.json", self.previous)

    def task(self, key):
        launcher, plan = self.root/(key+".sh"), self.root/(key+"-plan.json")
        launcher.write_text("synthetic launcher only")
        dump(plan, {"synthetic": True})
        output = self.root/(key+"-output")
        return {"id": key, "model": "core2", "freeze_commit": "a"*40, "launcher": str(launcher),
            "launcher_sha256": B.sha(launcher), "plan": str(plan), "plan_sha256": B.sha(plan),
            "command": ["bash", str(launcher), "a"*40, str(plan), "core2", str(output)],
            "output": str(output), "completion": [str(output/"complete.json")], "min_free_mib": 8192,
            "priority": 20, "cwd": str(ROOT)}

    def running(self, task, origin, worker, child, gpu):
        old_origin = origin == self.earlier
        if old_origin:
            dump(origin/"schedule.json", {"tasks": [task]})
        schedule_path = origin/"schedule.json"
        row = {"status": "running", "token": "token-"+str(worker), "command": task["command"],
            "worker": {"pid": worker, "start_ticks": worker, "boot_id": "synthetic"},
            "child": {"pid": child, "start_ticks": child, "boot_id": "synthetic"}, "gpu": {"uuid": gpu},
            "receipt": str(origin/(task["id"]+"-attempt.json"))}
        if old_origin:
            row.update(adopted_runtime=str(origin), adopted_schedule_sha256=B.sha(schedule_path))
        receipt = {**{k: row[k] for k in ("status", "token", "worker", "child", "gpu")},
            "task_id": task["id"], "schedule_sha256": B.sha(schedule_path)}
        dump(row["receipt"], receipt)
        dump(origin/(task["id"]+"-worker.json"), {"task": task, "token": row["token"], "gpu": row["gpu"],
            "receipt": row["receipt"], "schedule_sha256": B.sha(schedule_path),
            "execution_proof": {"schedule": str(schedule_path), "freeze_commit": "a"*40}})
        return row

    def dispatcher(self, query=lambda: [], **kwargs):
        self.new.mkdir(exist_ok=True)
        dump(self.new/"previous-state.json", self.previous)
        dump(self.new/"migration.json", {"new_schedule_sha256": "d"*64,
            "previous_state_sha256": B.sha(self.new/"previous-state.json"), "old_parent_retired": True})
        return C.Dispatcher(self.schedule, "d"*64, self.new, query=query, **kwargs)


class Additive(Fixture):
    def test_seed_preserves_failures_and_earliest_origin(self):
        state = C.seed(self.schedule, "d"*64, self.previous)
        self.assertEqual(state["tasks"]["already-failed"], self.previous["tasks"]["already-failed"])
        row = state["tasks"]["mcq-core2"]
        self.assertEqual(row["receipt"], self.previous["tasks"]["mcq-core2"]["receipt"])
        self.assertEqual(row["adopted_schedule_sha256"], B.sha(self.earlier/"schedule.json"))
        self.assertEqual(row["adopted_runtime"], str(self.earlier))

    def test_reject_any_inherited_dictionary_or_order_change(self):
        for field, value in (("min_free_mib", 20480), ("command", ["changed"]), ("priority", 0)):
            bad = copy.deepcopy(self.schedule)
            bad["tasks"][0][field] = value
            with self.assertRaisesRegex(ValueError, "preserve"):
                C.validate(bad)
        bad = copy.deepcopy(self.schedule)
        bad["tasks"].reverse()
        with self.assertRaises(ValueError):
            C.validate(bad)

    def test_adopted_count_and_failed_never_retried(self):
        dispatcher = self.dispatcher()
        with patch.object(C.V3.B, "alive", return_value=True):
            dispatcher.tick()
        self.assertEqual(dispatcher.live_count(), 2)
        self.assertEqual(dispatcher.state["tasks"]["already-failed"]["status"], "failed")
        self.assertEqual({r["gpu"]["uuid"] for r in dispatcher.state["tasks"].values() if r["status"] in B.LIVE}, {"GPU-old", "GPU-new"})

    def add_new_task(self, key):
        row = self.task(key)
        self.schedule["tasks"].append(row)
        self.schedule["additions"][key] = C.digest(row)
        return row

    def third_running(self, dispatcher):
        path = self.new/"third-attempt.json"
        value = {"task_id": "third", "token": "third-token", "schedule_sha256": "d"*64,
            "status": "running", "worker": {"pid": 300}, "child": {"pid": 301}, "gpu": {"uuid": "GPU-third"}}
        dump(path, value)
        dispatcher.state["tasks"]["third"] = {**value, "receipt": str(path)}

    def test_three_adopted_or_new_live_can_start_fourth_then_four_reject_fifth(self):
        self.add_new_task("third"); self.add_new_task("fifth")
        timer = [0.]
        gpu = {"uuid": "GPU-fourth", "index": 3, "free_mib": 9000, "compute_pids": [987], "utilization": 100}
        launches = []
        def popen(argv, **options):
            state = json.loads((self.new/"state.json").read_text())
            self.assertEqual(state["tasks"]["mcmr-core2"]["status"], "starting")
            self.assertEqual(argv[2], str(C.HERE/"controller.py"))
            launches.append(argv)
            return SimpleNamespace(pid=400)
        dispatcher = self.dispatcher(query=lambda: [gpu], clock=lambda: timer[0], popen=popen, complete=lambda task: False)
        self.addCleanup(lambda: [lease.__exit__(None, None, None) for lease in dispatcher.leases.values()])
        self.third_running(dispatcher)
        with patch.object(C.V3.B, "alive", return_value=True), patch.object(B, "process_token", side_effect=lambda pid: {"pid": pid}):
            dispatcher.tick()
            self.assertEqual(dispatcher.live_count(), 3)
            self.assertFalse(launches)
            timer[0] = 5.
            dispatcher.tick()
            self.assertEqual(dispatcher.live_count(), 4)
            self.assertEqual(len(launches), 1)
            dispatcher.query = lambda: self.fail("A fifth admission query is forbidden at cap4")
            dispatcher.tick()
            self.assertEqual(dispatcher.live_count(), 4)
            self.assertEqual(dispatcher.state["tasks"]["fifth"]["status"], "pending")
            self.assertEqual(len(launches), 1)

    def test_room_below_cap4_still_rejects_an_adopted_occupied_gpu(self):
        self.add_new_task("third")
        gpu = {"uuid": "GPU-old", "index": 0, "free_mib": 24000, "compute_pids": [101]}
        def forbidden(*args, **kwargs):
            self.fail("Cannot start on an adopted worker's occupied GPU")
        dispatcher = self.dispatcher(query=lambda: [gpu], clock=lambda: 10., popen=forbidden, complete=lambda task: False)
        self.third_running(dispatcher)
        with patch.object(C.V3.B, "alive", return_value=True):
            dispatcher.tick(); dispatcher.tick()
        self.assertEqual(dispatcher.live_count(), 3)

    def test_new_schedule_requires_exact_cap4(self):
        for limit in (2, 3, 5):
            bad = copy.deepcopy(self.schedule); bad["max_live"] = limit
            with self.assertRaisesRegex(ValueError, "cap4"):
                C.validate(bad)

    def test_no_canonical_validator_mutation_and_own_stop_only(self):
        original = canonical.validate
        dispatcher = self.dispatcher()
        self.assertIs(canonical.validate, original)
        self.assertIs(C.V3.B.validate, C.V3.ORIGINAL_VALIDATE)
        self.assertIsNot(B, canonical)
        dispatcher.children["fixed-core2"] = SimpleNamespace(pid=200)
        with patch.object(dispatcher, "reconcile"), patch.object(C.V3.B, "alive", return_value=True), patch.object(C.V3.B.os, "kill") as kill:
            dispatcher.stop_own()
            kill.assert_called_once_with(200, signal.SIGTERM)

    def test_exact_siglip_12g_plan_only(self):
        plan = ROOT/"codex/retrieval_matrix_native_p0/p0-siglip2.json"
        row = self.task("new-siglip2")
        row.update(model="siglip2", plan=str(plan), plan_sha256=B.sha(plan), min_free_mib=12288)
        self.schedule["tasks"].append(row)
        self.schedule["additions"][row["id"]] = C.digest(row)
        self.schedule["resource_exceptions"][row["id"]] = C.siglip_exception(row)
        C.validate(self.schedule)
        bad = copy.deepcopy(self.schedule)
        bad["resource_exceptions"][row["id"]]["model_spec_sha256"] = "0"*64
        with self.assertRaisesRegex(ValueError, "exceptions"):
            C.validate(bad)
        row["model"] = "core2"
        with self.assertRaisesRegex(ValueError, "12GiB"):
            C.siglip_exception(row)

    def test_builder_in_memory_synthetic_sources_and_insertion(self):
        out = ROOT/"dl/synthetic-next-test-never-executed"
        request = {"id": "synthetic-addition", "model": "core2", "freeze_commit": "a"*40,
            "launcher": self.tasks[0]["launcher"], "plan": self.tasks[0]["plan"],
            "output": str(out), "completion": [str(out/name/"index/final-receipt.json") for name in ("sugarcrepe", "sugarcrepe_pp")], "min_free_mib": 8192}
        additions = self.root/"requests.json"
        dump(additions, {"schema": 1, "owner": "[codex]", "tasks": [request]})
        dump(self.old_path, self.schedule | {"tasks": self.tasks, "max_live": 2})
        with patch.object(S, "frozen", side_effect=lambda path, commit: Path(path)):
            value = S.build(self.old_path, "a"*40, self.old, additions)
        self.assertEqual(len(value["tasks"]), 5)
        self.assertEqual(value["max_live"], 4)
        old = self.tasks
        self.assertEqual([row for row in value["tasks"] if row["id"] != request["id"]], old)
        ids = [row["id"] for row in value["tasks"]]
        self.assertLess(ids.index(request["id"]), ids.index("mcmr-core2"))

    def test_frozen_source_checks_git_bytes_without_running_launcher(self):
        path = ROOT/"codex/retrieval_matrix_dispatch_batches/central-v3.json"
        self.assertEqual(S.frozen(path, "665a5743c6c1308ae84d25dd450a625cb98fa3b4"), path)
        with patch.object(S.subprocess, "check_output", return_value=b"different committed payload"):
            with self.assertRaisesRegex(ValueError, "fixed commit"):
                S.frozen(path, "a"*40)

    def test_real_twenty_seven_task_source_preserved_without_runtime_read(self):
        old_path = ROOT/"codex/retrieval_matrix_dispatch_batches/central-v3.json"
        old = json.loads(old_path.read_text())
        value = copy.deepcopy(old)
        value.update(controller="immutable-additive-next-v1", max_live=4, additions={}, resource_exceptions={},
            adoption={"schedule": C.entry(old_path), "freeze_commit": "665a5743c6c1308ae84d25dd450a625cb98fa3b4",
                "runtime": str(self.old), "retire_pending": []})
        C.validate(value)
        previous = {"schedule_sha256": B.sha(old_path), "tasks": {row["id"]: {"status": "failed"} for row in old["tasks"]}}
        state = C.seed(value, "f"*64, previous)
        self.assertEqual(len(state["tasks"]), 27)
        self.assertTrue(all(row["status"] == "failed" for row in state["tasks"].values()))

    def test_native_completion_exact_variant_no_metrics(self):
        plan_path = ROOT/"codex/retrieval_matrix_rerank_native_p0/plan.native-mcq.approved.v1.json"
        row = self.task("native-qwen2")
        row.update(model="qwen2", plan=str(plan_path), plan_sha256=B.sha(plan_path), completion_profile="native-reranker-v1", min_free_mib=20480)
        path = Path(row["output"])/"scores/complete.json"
        row["completion"] = [str(path)]
        self.assertFalse(C.completed(row))
        variant = C.require_reranker(row)
        value = {"state": "COMPLETE", "identity": {"variant": "qwen2", "variant_spec": variant,
            "plan_sha256": row["plan_sha256"], "freeze_commit": row["freeze_commit"]},
            "examples": 15912, "distinct_pairs": 58891, "original_images": 9425}
        dump(path, value)
        self.assertTrue(C.completed(row))
        value["identity"]["variant"] = "qwen8"
        dump(path, value)
        with self.assertRaisesRegex(ValueError, "variant"):
            C.completed(row)
        row["model"] = "qwen_native8"
        with self.assertRaisesRegex(ValueError, "approved"):
            C.require_reranker(row)


class FakeHost:
    def __init__(self, fixture):
        self.fixture, self.signals, self.spawns, self.logs = fixture, [], [], []
        self.tokens = {10: fixture.parent}
        self.commands = {10: H.parent_command(fixture.config["old"])}
        for key, row in fixture.previous["tasks"].items():
            if row["status"] != "running":
                continue
            self.tokens.update({row[name]["pid"]: row[name] for name in ("worker", "child")})
            origin = Path(row.get("adopted_runtime", fixture.old))
            script = canonical.HERE/"dispatch.py" if origin == fixture.earlier else Path(fixture.config["old"]["controller"])
            self.commands[row["worker"]["pid"]] = ["/usr/bin/python3", "-B", str(script), "worker", "--config", str(origin/(key+"-worker.json")), "--gpu-fd", "7"]
            self.commands[row["child"]["pid"]] = row["command"]
        self.bad_lease = False
        self.on_retire = lambda: None
        self.fail_singleton = False
        self.fail_spawn = False

    def alive(self, token):
        return bool(token) and self.tokens.get(token["pid"]) == token

    def command(self, token):
        if not self.alive(token):
            raise ValueError("Synthetic process disappeared")
        return self.commands[token["pid"]]

    def freeze(self, path, ref):
        pass

    def signal_parent(self, token, command, signum):
        if token != self.fixture.parent or self.command(token) != command:
            raise AssertionError("Cannot signal a model or an unverified parent")
        self.signals.append((token["pid"], signum))
        if signum == signal.SIGKILL:
            del self.tokens[token["pid"]]
            self.on_retire()

    def wait_stopped(self, token):
        pass

    def wait_retired(self, token):
        if self.alive(token):
            raise AssertionError("Parent remains alive")

    def lease(self, worker, descriptor, path):
        if self.bad_lease:
            raise ValueError("Synthetic wrong GPU lease inode")
        return {"device": 1, "inode": worker["pid"], "held": True}

    def singleton_free(self):
        if self.fail_singleton:
            raise ValueError("Synthetic singleton is occupied")

    def spawn(self, command, runtime):
        self.spawns.append(command)
        if self.fail_spawn:
            raise OSError("Synthetic Popen uncertainty")
        return {"pid": 999, "token": {"pid": 999, "start_ticks": 999, "boot_id": "synthetic"}, "command": command}

    def log(self, *args):
        self.logs.append(args)


class FrozenLauncherExpansion(unittest.TestCase):
    """Only frozen source metadata is read; no original output or process."""
    def tasks(self):
        source = ROOT/"codex/retrieval_matrix_dispatch_batches/central-v3.json"
        return [task for task in json.loads(source.read_text())["tasks"]
            if task["id"] in {"fixed-"+mode+"-"+model for mode in ("embedding", "mcmr")
                for model in ("gme2", "vlm2vec2")}]

    def expanded(self, task):
        mode = "mcmr" if task["id"].startswith("fixed-mcmr-") else "embedding"
        return ["bash", str(H.QWEN2_FIX/"run.sh"), mode, *task["command"][2:]]

    def test_all_four_exact_frozen_wrapper_execs_and_source_evidence(self):
        self.assertEqual(len(self.tasks()), 4)
        for task in self.tasks():
            with self.subTest(task=task["id"]):
                host = H.Host()
                with patch.object(host, "freeze", wraps=S.frozen) as freeze:
                    actual = self.expanded(task)
                    proof = H.child_command_proof(task, actual, host)
                self.assertEqual(proof["dispatch_command"], task["command"])
                self.assertEqual(proof["child_command"], actual)
                self.assertEqual(proof["launcher_expansion"]["model"], task["model"])
                self.assertEqual(proof["launcher_expansion"]["freeze_commit"], task["freeze_commit"])
                self.assertEqual([call.args for call in freeze.call_args_list], [
                    (H.QWEN2_FIX/"central_task.sh", H.QWEN2_FIX_REF),
                    (H.QWEN2_FIX/"run.sh", H.QWEN2_FIX_REF),
                    (Path(task["plan"]), H.QWEN2_FIX_REF)])

    def test_ordinary_exact_command_keeps_original_rule_without_expansion(self):
        task = {"command": ["bash", "/synthetic/ordinary.sh", "unchanged"]}
        host = SimpleNamespace(freeze=lambda *args: self.fail("No wrapper expansion for an exact command"))
        proof = H.child_command_proof(task, task["command"], host)
        self.assertIsNone(proof["launcher_expansion"])
        with self.assertRaisesRegex(ValueError, "child command"):
            H.child_command_proof(task, ["bash", "/synthetic/run.sh", "unchanged"], host)

    def test_wrong_expanded_mode_model_ref_plan_output_or_script_is_rejected(self):
        for task in self.tasks():
            actual = self.expanded(task)
            changes = {1: "/another/run.sh", 2: "mcmr" if actual[2] == "embedding" else "embedding",
                3: "0"*40, 4: "/another/plan.json", 5: "core2", 6: task["output"]+"-other"}
            for index, value in changes.items():
                with self.subTest(task=task["id"], index=index):
                    bad = actual.copy(); bad[index] = value
                    with self.assertRaisesRegex(ValueError, "child command"):
                        H.child_command_proof(task, bad, H.Host())
            for bad in (actual+["extra"], ["/bin/bash", *actual[1:]]):
                with self.assertRaisesRegex(ValueError, "child command"):
                    H.child_command_proof(task, bad, H.Host())

    def test_mutating_dispatch_identity_never_creates_a_new_allowlisted_exec(self):
        task = self.tasks()[0]
        changes = {"id": "another-task", "model": "core2", "freeze_commit": "0"*40,
            "plan": "/another/plan.json", "output": task["output"]+"-other",
            "launcher": "/another/central_task.sh", "plan_sha256": "0"*64,
            "launcher_sha256": "0"*64, "cwd": "/another"}
        for key, value in changes.items():
            with self.subTest(field=key):
                bad = copy.deepcopy(task); bad[key] = value
                bad["command"] = ["bash", bad["launcher"], bad["freeze_commit"], bad["plan"], bad["model"], bad["output"]]
                with self.assertRaisesRegex(ValueError, "child command"):
                    H.child_command_proof(bad, self.expanded(bad), H.Host())

    def test_changed_run_script_git_bytes_are_rejected_without_editing_source(self):
        real = S.subprocess.check_output
        wanted = H.QWEN2_FIX_REF+":codex/retrieval_matrix_qwen2_runtime_fix/run.sh"
        def different_run(command, *args, **kwargs):
            return b"different committed run script" if command[-1] == wanted else real(command, *args, **kwargs)
        task = self.tasks()[0]
        with patch.object(S.subprocess, "check_output", side_effect=different_run):
            with self.assertRaisesRegex(ValueError, "fixed commit"):
                H.child_command_proof(task, self.expanded(task), H.Host())

    def test_phase_mode_and_complete_model_spec_remain_bound(self):
        task = self.tasks()[0]
        original = json.loads(Path(task["plan"]).read_text())
        variants = []
        for key, value in (("phase", "MCMR-native-MMEB-v2"), ("state", "REVIEW_REQUIRED"), ("reviewed", False)):
            bad = copy.deepcopy(original); bad[key] = value; variants.append(bad)
        bad = copy.deepcopy(original); bad["runtime_fix"]["mode"] = "mcmr"; variants.append(bad)
        bad = copy.deepcopy(original); bad["models"][task["model"]]["synthetic_change"] = True; variants.append(bad)
        for bad in variants:
            with patch.object(H.json, "loads", return_value=bad):
                with self.assertRaisesRegex(ValueError, "phase/mode or complete model"):
                    H.child_command_proof(task, self.expanded(task), H.Host())


class Handoff(Fixture):
    def test_frozen_wrapper_expansion_keeps_child_identity_and_handoff_evidence(self):
        task = next(task for task in FrozenLauncherExpansion().tasks() if task["id"] == "fixed-embedding-gme2")
        self.tasks[1] = task
        dump(self.old_path, {"schema": 1, "max_live": 2, "tasks": self.tasks})
        self.previous["tasks"].pop("fixed-core2")
        self.previous["tasks"][task["id"]] = self.running(task, self.old, 200, 201, "GPU-new")
        self.previous["schedule_sha256"] = B.sha(self.old_path)
        self.schedule["tasks"] = copy.deepcopy(self.tasks)
        self.schedule["adoption"]["schedule"] = C.entry(self.old_path)
        self.config["old"]["schedule"] = C.entry(self.old_path)
        dump(self.new_path, self.schedule)
        self.config["new"]["schedule"] = C.entry(self.new_path)
        dump(self.old/"state.json", self.previous)
        host = FakeHost(self)
        host.commands[201] = FrozenLauncherExpansion().expanded(task)
        host.freeze = lambda path, ref: S.frozen(path, ref) if Path(path).parent == H.QWEN2_FIX else None
        result = H.execute(self.config, host=host, gate=False)
        proof = result["workers"][task["id"]]
        self.assertEqual(proof["child"], {"pid": 201, "start_ticks": 201, "boot_id": "synthetic"})
        self.assertEqual(proof["dispatch_command"], task["command"])
        self.assertEqual(proof["child_command"], host.commands[201])
        self.assertEqual(proof["launcher_expansion"]["run_script"]["sha256"], H.QWEN2_RUN_SHA)
        self.assertEqual(host.signals, [(10, signal.SIGSTOP), (10, signal.SIGKILL)])
        self.assertEqual(len(host.spawns), 1)

    def prepare_four_live_origin(self, old_limit):
        for key in ("third-old", "fourth-old"):
            self.tasks.append(self.task(key))
        dump(self.old_path, {"schema": 1, "max_live": old_limit, "tasks": self.tasks})
        old_sha = B.sha(self.old_path)
        self.previous["schedule_sha256"] = old_sha
        row = self.previous["tasks"]["fixed-core2"]
        for path in (Path(row["receipt"]), self.old/"fixed-core2-worker.json"):
            value = json.loads(path.read_text()); value["schedule_sha256"] = old_sha; dump(path, value)
        for index, key in enumerate(("third-old", "fourth-old"), 3):
            task = next(task for task in self.tasks if task["id"] == key)
            self.previous["tasks"][key] = self.running(task, self.old, index*100, index*100+1, "GPU-"+str(index))
        self.schedule["tasks"] = copy.deepcopy(self.tasks)
        self.schedule["adoption"]["schedule"] = C.entry(self.old_path)
        self.config["old"].update(schedule=C.entry(self.old_path), controller=str(C.HERE/"controller.py") if old_limit == 4 else str(C.V3.HERE/"controller.py"))
        dump(self.new_path, self.schedule)
        self.config["new"]["schedule"] = C.entry(self.new_path)
        dump(self.old/"state.json", self.previous)

    def test_future_cap4_parent_handoff_adopts_four_live_with_earliest_origin(self):
        self.prepare_four_live_origin(4)
        host = FakeHost(self)
        result = H.execute(self.config, host=host, gate=False)
        self.assertEqual(len(result["workers"]), 4)
        self.assertEqual(result["workers"]["mcq-core2"]["receipt"], str(self.earlier/"mcq-core2-attempt.json"))
        self.assertEqual(host.signals, [(10, signal.SIGSTOP), (10, signal.SIGKILL)])
        self.assertEqual(len(host.spawns), 1)

    def test_old_v3_cap2_snapshot_cannot_adopt_four_even_when_new_cap4(self):
        self.prepare_four_live_origin(2)
        host = FakeHost(self)
        with self.assertRaisesRegex(ValueError, "frozen cap2/cap4"):
            H.execute(self.config, host=host, gate=False)
        self.assertEqual(host.signals, [(10, signal.SIGSTOP), (10, signal.SIGCONT)])
        self.assertFalse(host.spawns)

    def test_real_signal_interface_pins_pidfd_and_rechecks_command_without_sending(self):
        host = H.Host()
        command = H.parent_command(self.config["old"])
        with patch.object(H.os, "pidfd_open", return_value=123) as opened, \
                patch.object(H.os, "close") as closed, \
                patch.object(host, "command", return_value=command), \
                patch.object(H.signal, "pidfd_send_signal") as sent:
            host.signal_parent(self.parent, command, signal.SIGSTOP)
            opened.assert_called_once_with(10)
            sent.assert_called_once_with(123, signal.SIGSTOP)
            closed.assert_called_once_with(123)
        with patch.object(H.os, "pidfd_open", return_value=123), \
                patch.object(H.os, "close"), patch.object(host, "command", return_value=["changed"]), \
                patch.object(H.signal, "pidfd_send_signal") as sent:
            with self.assertRaisesRegex(ValueError, "command changed"):
                host.signal_parent(self.parent, command, signal.SIGKILL)
            sent.assert_not_called()

    def test_wrong_parent_exact_command_performs_no_signal_or_runtime_write(self):
        host = FakeHost(self)
        host.commands[10] = ["another-controller"]
        with self.assertRaisesRegex(ValueError, "exact command"):
            H.execute(self.config, host=host, gate=False)
        self.assertFalse(host.signals)
        self.assertFalse(self.new.exists())

    def test_success_signals_only_old_parent_and_adopts_two_origins(self):
        host = FakeHost(self)
        result = H.execute(self.config, host=host, gate=False)
        self.assertEqual(host.signals, [(10, signal.SIGSTOP), (10, signal.SIGKILL)])
        self.assertEqual(len(host.spawns), 1)
        self.assertEqual(set(result["workers"]), {"mcq-core2", "fixed-core2"})
        self.assertEqual(result["workers"]["mcq-core2"]["receipt"], self.previous["tasks"]["mcq-core2"]["receipt"])
        self.assertTrue(result["old_parent_retired"])
        self.assertEqual(json.loads((self.new/"previous-state.json").read_text())["tasks"]["already-failed"], self.previous["tasks"]["already-failed"])

    def test_bad_lease_resumes_same_parent_no_spawn(self):
        host = FakeHost(self); host.bad_lease = True
        with self.assertRaisesRegex(ValueError, "lease"):
            H.execute(self.config, host=host, gate=False)
        self.assertEqual(host.signals, [(10, signal.SIGSTOP), (10, signal.SIGCONT)])
        self.assertFalse(host.spawns)

    def test_wrong_child_command_or_receipt_token_fail_before_retirement(self):
        host = FakeHost(self)
        host.commands[101] = ["wrong"]
        with self.assertRaisesRegex(ValueError, "child command"):
            H.execute(self.config, host=host, gate=False)
        self.assertNotIn((10, signal.SIGKILL), host.signals)
        self.assertEqual(host.signals[-1], (10, signal.SIGCONT))

    def test_wrong_original_receipt_token_never_retires_parent(self):
        row = self.previous["tasks"]["mcq-core2"]
        value = json.loads(Path(row["receipt"]).read_text())
        value["token"] = "another-attempt"
        dump(row["receipt"], value)
        host = FakeHost(self)
        with self.assertRaisesRegex(ValueError, "token/task/schedule"):
            H.execute(self.config, host=host, gate=False)
        self.assertEqual(host.signals, [(10, signal.SIGSTOP), (10, signal.SIGCONT)])
        self.assertFalse(host.spawns)

    def test_ambiguous_starting_is_never_replayed(self):
        previous = copy.deepcopy(self.previous)
        previous["tasks"]["fixed-core2"]["status"] = "starting"
        dump(self.old/"state.json", previous)
        host = FakeHost(self)
        with self.assertRaisesRegex(ValueError, "Ambiguous"):
            H.execute(self.config, host=host, gate=False)
        self.assertFalse(host.spawns)
        self.assertEqual(host.signals[-1], (10, signal.SIGCONT))

    def test_original_two_live_workers_cannot_share_gpu_at_handoff(self):
        previous = copy.deepcopy(self.previous)
        previous["tasks"]["fixed-core2"]["gpu"] = previous["tasks"]["mcq-core2"]["gpu"]
        dump(self.old/"state.json", previous)
        host = FakeHost(self)
        with self.assertRaisesRegex(ValueError, "perGPU1"):
            H.execute(self.config, host=host, gate=False)
        self.assertEqual(host.signals[-1], (10, signal.SIGCONT))
        self.assertFalse(host.spawns)

    def finish(self, host, key="fixed-core2"):
        row = self.previous["tasks"][key]
        host.tokens.pop(row["child"]["pid"], None)
        host.tokens.pop(row["worker"]["pid"], None)
        path = Path(row["receipt"])
        value = json.loads(path.read_text()); value.update(status="completed", returncode=0)
        dump(path, value)

    def test_terminal_receipt_race_preserves_terminal_and_evidence(self):
        host = FakeHost(self)
        host.on_retire = lambda: self.finish(host)
        result = H.execute(self.config, host=host, gate=False)
        self.assertEqual(result["workers"]["fixed-core2"]["status"], "completed")
        saved = json.loads((self.new/"previous-state.json").read_text())
        self.assertEqual(saved["tasks"]["fixed-core2"]["status"], "completed")
        self.assertEqual(len(host.spawns), 1)

    def test_after_retirement_race_fail_closed_then_explicit_recovery(self):
        host = FakeHost(self)
        host.on_retire = lambda: host.tokens.pop(201)
        with self.assertRaisesRegex(ValueError, "disappeared"):
            H.execute(self.config, host=host, gate=False)
        self.assertFalse(host.spawns)
        self.assertEqual(host.signals, [(10, signal.SIGSTOP), (10, signal.SIGKILL)])
        self.finish(host)
        result = H.recover(self.config, host=host, gate=False)
        self.assertEqual(result["phase"], "controller_start_attempted")
        self.assertEqual(len(host.spawns), 1)
        with self.assertRaisesRegex(ValueError, "Recovery"):
            H.recover(self.config, host=host, gate=False)

    def test_spawn_uncertainty_is_recorded_before_popen_never_auto_recovered(self):
        host = FakeHost(self); host.fail_spawn = True
        with self.assertRaisesRegex(OSError, "Popen"):
            H.execute(self.config, host=host, gate=False)
        proof = json.loads((self.new/"migration.json").read_text())
        self.assertTrue(proof["replacement_spawn_attempted"])
        with self.assertRaisesRegex(ValueError, "Recovery"):
            H.recover(self.config, host=host, gate=False)
        self.assertEqual(len(host.spawns), 1)

    def test_real_cpu_fd_inode_and_flock_proof(self):
        path, wrong = self.root/"own-test.lock", self.root/"other-test.lock"
        with path.open("w+") as a, wrong.open("w+") as b:
            fcntl.flock(a.fileno(), fcntl.LOCK_EX|fcntl.LOCK_NB)
            token = canonical.process_token(os.getpid())
            actual = H.Host().lease(token, str(a.fileno()), path)
            self.assertTrue(actual["held"])
            with self.assertRaisesRegex(ValueError, "inode"):
                H.Host().lease(token, str(b.fileno()), path)


if __name__ == "__main__":
    unittest.main()
