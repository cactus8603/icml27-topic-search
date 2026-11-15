"""CPU checks using synthetic losses and in-repository temporary files."""
import csv
import importlib.util
import json
import math
import os
from pathlib import Path
import shutil
import subprocess
import sys
import tempfile
from types import SimpleNamespace
import unittest
from unittest.mock import Mock, patch

import torch
import torch.nn.functional as F
import yaml

TOPIC = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(TOPIC))
from phasea_v2.estimator import Coverage, VERSION, combine_fold_dots, coverage_record, file_hash, item_token_counts, prompt_fold_weights, validate_gate1_coverage
from phasea_v2.inputs import OPTIONAL_MODELS, REQUIRED_MODEL, inputs_for_config, prepare, read_manifest, snapshot_evidence


def load_script(source, name):
    spec = importlib.util.spec_from_file_location(name, source / f"{name}.py")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


class TinyLM(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.weight = torch.nn.Parameter(torch.arange(9).reshape(3, 3).float() / 8)

    def forward(self, input_ids, attention_mask):
        return SimpleNamespace(logits=self.weight[input_ids])


def fake_snapshot(root, model="synthetic"):
    cache = root / ("models--" + model.replace("/", "--"))
    snapshot = cache / "snapshots" / ("2" * 40)
    snapshot.mkdir(parents=True)
    (cache / "blobs").mkdir()
    weight = cache / "blobs" / ("1" * 64)
    weight.write_text("synthetic weights")
    (snapshot / "model.safetensors").symlink_to(weight)
    for name in ("config.json", "tokenizer.json", "tokenizer_config.json"):
        (snapshot / name).write_text("{}")
    return snapshot


def manifest_for_config(root, config, model, snapshot):
    manifest = root / "inputs_v2.json"
    manifest.write_text(json.dumps({"version": 1, "models": {model: snapshot_evidence(snapshot)},
                                   "configs": {config.name: {"config_sha256": file_hash(config), "model": model, "vectors": {}}}}))
    return manifest


class PhaseATests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        torch.set_num_threads(1)
        override = os.environ.get("LAMBDA_V2_SCRIPTS")
        if override:
            source = Path(override)
        else:
            staging = tempfile.TemporaryDirectory(dir=TOPIC / "phasea_v2")
            cls.addClassCleanup(staging.cleanup)
            root = Path(staging.name)
            source = root / "killtests" / TOPIC.name / "scripts"
            source.mkdir(parents=True)
            for script in ("compute_lambda.py", "aggregate.py"):
                shutil.copy(TOPIC / "scripts" / script, source / script)
            with (TOPIC / "phasea_v2" / "integration.patch").open() as diff:
                subprocess.run(["patch", "--batch", "--forward", "--fuzz=0", "-d", str(root), "-p1"], stdin=diff, check=True)
        cls.compute = load_script(source, "compute_lambda")
        cls.aggregate = load_script(source, "aggregate")

    def test_unequal_prompt_and_fold_counts_match_direct_autograd_and_summary(self):
        model = TinyLM()
        params = [model.weight]
        # Folds have unequal token counts and each prompt changes the relative fold mass.
        seq = lambda tokens: (tokens, tokens)
        folds = [[[seq([0, 1, 0, 1])], [seq([0, 2])]],
                 [[seq([1, 0])], [seq([0, 2, 0, 2, 0, 2])]]]
        counts = [[sum(item_token_counts(items)) for items in fold] for fold in folds]
        weights = prompt_fold_weights(counts)
        self.assertEqual(counts, [[3, 1], [1, 5]])
        self.assertAlmostEqual(sum(weights[0]), 11 / 24)
        response = [[2, 1, 0], [1, 0, 1]]
        heldout = F.cross_entropy(model.weight[torch.tensor(response)[:, :-1]].reshape(-1, 3),
                                  torch.tensor(response)[:, 1:].reshape(-1))
        g_tau, = torch.autograd.grad(heldout, params)
        direct_prompt_losses = []
        for prompt in range(len(folds[0])):
            all_items = [item for fold in folds for item in fold[prompt]]
            logits = torch.cat([model.weight[torch.tensor(ids[:-1])] for ids, _ in all_items])
            labels = torch.cat([torch.tensor(labels[1:]) for _, labels in all_items])
            direct_prompt_losses.append(F.cross_entropy(logits, labels))
        direct_gradient, = torch.autograd.grad(torch.stack(direct_prompt_losses).mean(), params)
        direct = float((direct_gradient * g_tau).sum())
        pooled_gradient, _ = self.compute.collect_grad(model, params, [item for fold in folds for prompt in fold for item in prompt], 1, 0, "cpu")
        self.assertGreater(abs(direct - float((pooled_gradient[0] * g_tau).sum())), 0.005)

        rows, components = [], []
        for fold, tokens, prompt_weights in zip(folds, counts, weights, strict=True):
            weight = math.fsum(prompt_weights)
            prompt_blocks = [(items, n, w / weight) for items, n, w in zip(fold, tokens, prompt_weights, strict=True)]
            gradient, loss = self.compute.collect_grad(model, params, None, 2, 0, "cpu", prompt_blocks=prompt_blocks)
            dot = float((gradient[0] * g_tau).sum())
            components.append(gradient[0] * weight)
            rows.append(dict(defense="bank", tau_set="test", group="all", ctx="none", seed=0, dot=dot,
                             n_train_tokens=sum(tokens), train_fold_weight=weight, cos=0, gL_norm=0, gT_norm=0, train_loss=loss))
        torch.testing.assert_close(sum(components), direct_gradient)
        self.assertAlmostEqual(combine_fold_dots(rows), direct, places=6)
        with tempfile.TemporaryDirectory(dir=TOPIC / "phasea_v2") as tmp:
            target = Path(tmp) / "summary.csv"
            self.compute.write_summary(rows, target)
            with target.open() as stream:
                self.assertAlmostEqual(float(next(csv.DictReader(stream))["lam"]), direct, places=6)

    def test_finite_difference_uses_exact_averaged_direction_and_restores_origin(self):
        parameter = torch.nn.Parameter(torch.zeros(2))
        row = dict(seed=0, fold=0, defense="none", ctx="none", tau_set="heldout", group="all", dot=2.0, gL_norm=2.0)
        gradient = [torch.tensor([2.0, 0.0])]
        seen = []

        def nll(*_args):
            seen.append(parameter.detach().clone())
            return float(parameter.detach()[0])

        cache = {"_ctxs": [{"name": "none"}]}
        gate = {"fd_target_dtau": 0.01, "fd_ctx": "none", "fd_max_step": 0.005}
        with patch.object(self.compute, "nll_nograd", nll):
            self.compute.fd_apply([row], gradient, {("heldout", "none"): None}, {("heldout", "none"): 0.0}, cache,
                                  None, [parameter], 1, 0, "cpu", lambda *_: None, {}, gate, None, "none", 0, 0, exact_origin=True)
        torch.testing.assert_close(seen[0], -seen[1], rtol=0, atol=0)
        torch.testing.assert_close(parameter.detach(), torch.zeros(2), rtol=0, atol=0)
        self.assertAlmostEqual(row["dtau_central"], row["dtau_pred"], places=7)
        self.assertAlmostEqual(float(seen[1].norm()), gate["fd_max_step"], places=7)
        row.pop("dtau_true")
        row["dot"] = 0.0
        with self.assertRaisesRegex(ValueError, "UNTESTABLE"):
            self.compute.fd_apply([row], gradient, {("heldout", "none"): None}, {}, cache,
                                  None, [parameter], 1, 0, "cpu", None, {}, gate, None, "none", 0, 0, exact_origin=True)

    def test_exact_gate_requires_source_and_token_receipts(self):
        with tempfile.TemporaryDirectory(dir=TOPIC / "phasea_v2") as tmp:
            root = Path(tmp)
            records = [{"messages": [{"role": "user", "content": str(i)}, {"role": "assistant", "content": "ok"}]} for i in range(4)]
            data = root / "records.jsonl"
            data.write_text("".join(json.dumps(record) + "\n" for record in records))
            prompt = root / "p.json"
            prompt.write_text(json.dumps({"prompts": ["a", "b"]}))
            spec = lambda offset: {"file": data.name, "n": 2, "offset": offset}
            cfg = dict(data_root=str(root), train=spec(0), tau_sets={"heldout": spec(2)}, max_len=10,
                       defenses=[{"name": "bank", "kind": "ip", "prompt_file": prompt.name}],
                       contexts=[{"name": "none"}, {"name": "alt"}], dtype="float32", n_folds=1, microbatch=2,
                       groups=[{"name": "all"}], lora={"seeds": [0]}, model="synthetic")
            receipt = Coverage(train=coverage_record(data, records, [0, 1]),
                               tau={"heldout": {**coverage_record(data, records, [2, 3]), "contexts": {"none": [2, 3], "alt": [3, 3]}}},
                               defenses={"bank": {"prompt_file_sha256": file_hash(prompt), "loss_tokens_by_fold_prompt": [[[3, 6], [4, 2]]]}})
            row = dict(defense="bank", ctx="none", tau_set="heldout", n_prefixes=2, n_train_sequences=2, n_train_items=4,
                       n_train_tokens=15, n_tau_items=2, n_tau_tokens=5, train_fold_weight=1.0,
                       dtau_pred=.01, dtau_true=.01, dtau_central=.01, seed=0, group="all", fold=0)
            alt = {k: v for k, v in row.items() if not k.startswith("dtau")}
            alt.update(ctx="alt", n_tau_tokens=6)
            config_dir = root / "configs"
            config_dir.mkdir()
            config_path = config_dir / "grant_synthetic_qwen7b.yaml"
            config_path.write_text(yaml.safe_dump(cfg))
            manifest = manifest_for_config(root, config_path, cfg["model"], fake_snapshot(root))
            _, _, evidence = inputs_for_config(read_manifest(manifest), manifest, config_path, cfg["model"])
            raw_dir = root / "grant_synthetic_qwen7b_fd"
            raw_dir.mkdir()
            raw = raw_dir / "lambda_raw.json"
            synthetic_gate = dict(fd_ctx="none", fd_dtype="float32", fd_n_lora_seeds=1)
            values = dict(config=cfg, model=cfg["model"], job_id="100", estimator=VERSION,
                          coverage=receipt.to_dict(), rows=[row, alt], inputs={**evidence, "manifest_sha256": file_hash(manifest)})
            validate_gate1_coverage(values["coverage"], cfg, values["rows"])
            with patch.multiple(self.aggregate, ROOT=str(root), FD_TRAITS=("synthetic",)):
                raw.write_text(json.dumps(values))
                self.assertEqual(len(self.aggregate.gate1_inputs(root, synthetic_gate, "100", v2=True)), 1)
                raw.write_text(json.dumps({**values, "inputs": {**evidence, "manifest_sha256": "different"}}))
                with self.assertRaisesRegex(ValueError, "different input manifest"):
                    self.aggregate.gate1_inputs(root, synthetic_gate, "100", v2=True)
                raw.write_text(json.dumps(values))
                with self.assertRaises(ValueError):
                    self.aggregate.gate1_inputs(root, synthetic_gate, "101", v2=True)
                raw.write_text(json.dumps({**values, "rows": [row]}))
                with self.assertRaisesRegex(ValueError, "defense-by-context"):
                    self.aggregate.gate1_inputs(root, synthetic_gate, "100", v2=True)
                values["coverage"]["tau"]["heldout"]["record_indices"][:] = [3, 2]
                raw.write_text(json.dumps(values))
                status = self.aggregate.gate1(root, {"gate1": synthetic_gate}, False, v2=True, origin_job="100")
                self.assertEqual(status["status"], "UNTESTABLE")

    def test_cli_preserves_failed_gate_receipt_and_has_no_v2_phase_b(self):
        with tempfile.TemporaryDirectory(dir=TOPIC / "phasea_v2") as tmp:
            root = Path(tmp)
            snapshot = root / "source"
            scripts = snapshot / "scripts"
            scripts.mkdir(parents=True)
            shutil.copy(self.aggregate.__file__, scripts / "aggregate.py")
            shutil.copytree(TOPIC / "configs", snapshot / "configs")
            (snapshot / "phasea_v2").mkdir()
            shutil.copy(TOPIC / "phasea_v2" / "estimator.py", snapshot / "phasea_v2" / "estimator.py")
            shutil.copy(TOPIC / "phasea_v2" / "inputs.py", snapshot / "phasea_v2" / "inputs.py")
            output = root / "isolated-run"
            output.mkdir()
            (output / "run_v2.json").write_text(json.dumps({"gate_job": "123"}))
            command = [sys.executable, "-BI", str(scripts / "aggregate.py"), "--phase-a-v2", "--results-dir", str(output)]
            done = subprocess.run([*command, "--phase", "a"], capture_output=True, text=True)
            self.assertEqual(done.returncode, 1, done.stderr)
            report = output / "aggregate_phaseA_v2_gate.json"
            self.assertTrue(report.is_file(), done.stderr)
            saved = report.read_bytes()
            self.assertEqual(json.loads(saved)["gate1"]["status"], "UNTESTABLE")
            self.assertEqual(json.loads(saved)["origin_gate_job"], "123")
            retry = subprocess.run([*command, "--phase", "a"], capture_output=True, text=True)
            self.assertNotEqual(retry.returncode, 0)
            self.assertEqual(report.read_bytes(), saved)
            old_phase = subprocess.run([*command, "--phase", "b"], capture_output=True, text=True)
            self.assertEqual(old_phase.returncode, 2)
            self.assertNotIn("RESULT_PRIMARY", old_phase.stdout)

    def test_compute_runs_v2_mode_and_labels_pg_social_coverage(self):
        with tempfile.TemporaryDirectory(dir=TOPIC / "phasea_v2") as tmp:
            root = Path(tmp)
            dataset = root / "records.jsonl"
            records = [dict(messages=[{"role": "user", "content": "x"}, {"role": "assistant", "content": "y"}])] * 88
            dataset.write_text("".join(json.dumps(record) + "\n" for record in records))
            prompt = root / "prompts.json"
            prompt.write_text(json.dumps({"prompts": ["a", "abc"]}))
            cfg = dict(name="pg_synthetic", model="synthetic", data_root=str(root), param_mode="full_layers",
                       loss_mask="all", n_folds=2, max_len=32, stream_tau=True, dtype="float32",
                       train=dict(file=dataset.name, n=2), tau_sets=dict(OOD_social=dict(file=dataset.name, offset=2, n=128)),
                       contexts=[dict(name="plain")], defenses=[dict(name="bank", kind="ip", prompt_file=prompt.name)])
            config = root / "synthetic.yaml"
            config.write_text(yaml.safe_dump(cfg))

            class Tokenizer:
                pad_token_id = 0

                def apply_chat_template(self, messages, tokenize, add_generation_prompt=False):
                    text = "".join("r" + msg["content"] for msg in messages)
                    return text + ("r" if add_generation_prompt else "")

                def __call__(self, text, add_special_tokens, truncation=False, max_length=None):
                    return {"input_ids": [ord(char) % 3 for char in text[:max_length]]}

            class Model(TinyLM):
                def __init__(self):
                    super().__init__()
                    del self.weight
                    self.model = torch.nn.Module()
                    self.model.layers = torch.nn.ModuleList([torch.nn.Embedding(3, 3)])

                def forward(self, input_ids, attention_mask):
                    return SimpleNamespace(logits=self.model.layers[0](input_ids))

            snapshot = fake_snapshot(root)
            manifest = manifest_for_config(root, config, cfg["model"], snapshot)
            load_tokenizer = Mock(return_value=Tokenizer())
            load_model = Mock(side_effect=lambda *_, **__: Model())
            transformers = SimpleNamespace(AutoTokenizer=SimpleNamespace(from_pretrained=load_tokenizer),
                                           AutoModelForCausalLM=SimpleNamespace(from_pretrained=load_model))
            output = root / "run"
            argv = [self.compute.__file__, "--phase-a-v2", "--config", str(config), "--out", str(output), "--device", "cpu",
                    "--input-manifest", str(manifest)]
            with patch.dict(sys.modules, {"transformers": transformers}), patch.object(sys, "argv", argv), \
                    patch.object(self.compute, "__file__", str(TOPIC / "scripts" / "compute_lambda.py")), \
                    patch.dict(os.environ, {"SLURM_JOB_ID": "synthetic"}):
                self.compute.main()
                with self.assertRaises(FileExistsError):
                    self.compute.main()
            data = json.loads((output / "lambda_raw.json").read_text())
            self.assertEqual(data["config"]["tau_sets"]["OOD_social"]["n"], 85)
            self.assertEqual(len(data["coverage"]["tau"]["OOD_social"]["contexts"]["plain"]), 85)
            self.assertEqual(data["estimator"], VERSION)
            self.assertEqual(data["inputs"]["manifest_sha256"], file_hash(manifest))
            load_tokenizer.assert_called_once_with(str(snapshot), local_files_only=True)
            load_model.assert_called_once_with(str(snapshot), local_files_only=True, use_safetensors=True, torch_dtype=torch.float32)

    def test_shared_inputs_pin_cache_paths_copy_unique_pps_and_reject_changes(self):
        with tempfile.TemporaryDirectory(dir=TOPIC / "phasea_v2") as tmp:
            root = Path(tmp)
            config_dir = root / "configs"
            config_dir.mkdir()
            vector = root / "original.pt"
            vector.write_bytes(b"frozen vector")
            grant = config_dir / "grant_synthetic.yaml"
            grant.write_text(yaml.safe_dump(dict(model=REQUIRED_MODEL, data_root=str(root), defenses=[
                dict(name="pps1", kind="pps", vector=vector.name), dict(name="pps2", kind="pps", vector=vector.name)])))
            snapshots = {model: fake_snapshot(root, model) for model in {REQUIRED_MODEL, *OPTIONAL_MODELS}}
            for i, model in enumerate(sorted(OPTIONAL_MODELS)):
                (config_dir / f"pg_synthetic_{i}.yaml").write_text(yaml.safe_dump(dict(model=model, defenses=[])))
            absent, = [model for model in OPTIONAL_MODELS if model.startswith("allenai/")]

            def resolve(model):
                if model == absent:
                    raise FileNotFoundError(model)
                return snapshot_evidence(snapshots[model])

            manifest = root / "inputs_v2.json"
            with patch("phasea_v2.inputs.resolve_snapshot", side_effect=resolve) as resolve_call:
                prepare(config_dir, manifest)
                with self.assertRaises(FileExistsError):
                    prepare(config_dir, manifest)
                self.assertEqual(resolve_call.call_count, 3)
            (root / "pps_v2").chmod(0o755)
            document = read_manifest(manifest)
            self.assertIn("unavailable", document["models"][absent])
            vector.write_bytes(b"external source changed")
            snapshot, vectors, evidence = inputs_for_config(document, manifest, grant, REQUIRED_MODEL, verify=True)
            copied = vectors["pps1"]
            self.assertEqual(copied, vectors["pps2"])
            self.assertEqual(copied.read_bytes(), b"frozen vector")
            self.assertEqual(evidence["tokenizer_snapshot"], str(snapshot))
            weight = snapshot / "model.safetensors"
            other = weight.resolve().with_name("f" * 64)
            other.write_bytes(weight.read_bytes())
            original = weight.readlink()
            weight.unlink()
            weight.symlink_to(other)
            with self.assertRaisesRegex(ValueError, "model or tokenizer snapshot differs"):
                inputs_for_config(document, manifest, grant, REQUIRED_MODEL, verify=True)
            weight.unlink()
            weight.symlink_to(original)
            tokenizer = snapshot / "tokenizer.json"
            tokenizer.write_text('{"changed": true}')
            with self.assertRaisesRegex(ValueError, "model or tokenizer snapshot differs"):
                inputs_for_config(document, manifest, grant, REQUIRED_MODEL, verify=True)
            tokenizer.write_text("{}")
            copied.chmod(0o644)
            copied.write_bytes(b"copied vector changed")
            with self.assertRaisesRegex(ValueError, "PPS vector differs"):
                inputs_for_config(document, manifest, grant, REQUIRED_MODEL, verify=True)


if __name__ == "__main__":
    unittest.main()
