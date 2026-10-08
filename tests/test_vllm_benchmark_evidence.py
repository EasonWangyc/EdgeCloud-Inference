"""Benchmark configuration drift and existing evidence must be explicit."""

import copy
import importlib.util
import io
import json
import tempfile
import unittest
from contextlib import redirect_stderr, redirect_stdout
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch


ROOT = Path(__file__).resolve().parents[1]
SPEC = importlib.util.spec_from_file_location("benchmark_vllm_http", ROOT / "scripts/benchmark_vllm_http.py")
assert SPEC is not None and SPEC.loader is not None
MODULE = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(MODULE)


class BenchmarkEvidenceTests(unittest.TestCase):
    def setUp(self):
        self.payload = {"vllm_config": {
            "model_config": {"revision": "fixed", "enforce_eager": True},
            "cache_config": {"enable_prefix_caching": False},
            "scheduler_config": {"max_num_seqs": 4},
            "compilation_config": {}, "parallel_config": {},
        }}
        self.report = SimpleNamespace(
            records=[object()], failure_summary={},
            performance_metrics=SimpleNamespace(to_mapping=lambda: {}, backend_completed_sample_count=1,
                                                token_counts={"output_tokens": 2}),
        )

    def arguments(self, output):
        return ["--config", str(ROOT / "configs/studies/wsl_vllm_workload_resize_ps20.json"),
                "--output", str(output), "--server-info-url", "http://localhost/server_info?config_format=json"]

    def run_with_snapshots(self, output, before, after):
        responses = [io.BytesIO(json.dumps(payload).encode()) for payload in (before, after)]
        with patch.object(MODULE, "urlopen", side_effect=responses), \
                patch.object(MODULE, "run_configured_study", return_value=self.report), \
                redirect_stdout(io.StringIO()):
            return MODULE.main(self.arguments(output))

    def test_unchanged_effective_settings_are_saved_as_evidence(self):
        with tempfile.TemporaryDirectory() as directory:
            output = Path(directory) / "run.json"
            self.assertEqual(self.run_with_snapshots(output, self.payload, self.payload), 0)
            execution = json.loads(output.with_suffix(".execution.json").read_text())
            self.assertTrue(execution["server_config_unchanged"])
            self.assertEqual(execution["status"], "succeeded")
            before = output.with_suffix(".server_config.before.json").read_text()
            self.assertEqual(before, output.with_suffix(".server_config.after.json").read_text())

    def test_changed_cache_settings_fail_a_successful_inference_run(self):
        after = copy.deepcopy(self.payload)
        after["vllm_config"]["cache_config"]["enable_prefix_caching"] = True
        with tempfile.TemporaryDirectory() as directory:
            output = Path(directory) / "run.json"
            self.assertEqual(self.run_with_snapshots(output, self.payload, after), 2)
            execution = json.loads(output.with_suffix(".execution.json").read_text())
            self.assertFalse(execution["server_config_unchanged"])
            self.assertEqual(execution["status"], "failed")

    def test_existing_configuration_sidecar_is_not_overwritten(self):
        with tempfile.TemporaryDirectory() as directory:
            output = Path(directory) / "run.json"
            evidence = output.with_suffix(".server_config.before.json")
            evidence.write_text("original evidence")
            with patch.object(MODULE, "run_configured_study") as run, \
                    redirect_stderr(io.StringIO()), self.assertRaises(SystemExit) as error:
                MODULE.main(self.arguments(output))
            self.assertEqual(error.exception.code, 2)
            run.assert_not_called()
            self.assertEqual(evidence.read_text(), "original evidence")

    def test_server_counter_mismatch_or_reset_fails_the_window(self):
        before = "vllm:request_success_total 10\nvllm:generation_tokens_total 20\n"
        for requests, tokens, expected_code in ((11, 22, 0), (12, 22, 2), (11, 23, 2), (1, 2, 2)):
            with self.subTest(requests=requests, tokens=tokens), tempfile.TemporaryDirectory() as directory:
                output = Path(directory) / "run.json"
                after = f"vllm:request_success_total {requests}\nvllm:generation_tokens_total {tokens}\n"
                args = self.arguments(output)[:-2] + ["--metrics-url", "http://localhost/metrics"]
                responses = [io.BytesIO(text.encode()) for text in (before, after)]
                with patch.object(MODULE, "urlopen", side_effect=responses), \
                        patch.object(MODULE, "run_configured_study", return_value=self.report), \
                        redirect_stdout(io.StringIO()):
                    self.assertEqual(MODULE.main(args), expected_code)
                execution = json.loads(output.with_suffix(".execution.json").read_text())
                self.assertEqual(execution["server_metrics_consistent"], expected_code == 0)
                self.assertEqual(execution["status"], "succeeded" if expected_code == 0 else "failed")
