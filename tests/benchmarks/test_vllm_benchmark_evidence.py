"""Benchmark configuration drift and existing evidence must be explicit."""

import copy
import importlib.util
import io
import json
import tempfile
import unittest
from contextlib import redirect_stderr, redirect_stdout
from datetime import datetime, timedelta, timezone
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch


ROOT = Path(__file__).resolve().parents[2]
SPEC = importlib.util.spec_from_file_location("benchmark_vllm_http", ROOT / "scripts/benchmark_vllm_http.py")
assert SPEC is not None and SPEC.loader is not None
MODULE = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(MODULE)


class BenchmarkEvidenceTests(unittest.TestCase):
    def setUp(self):
        self.payload = {"vllm_config": {
            "model_config": {
                "revision": "89644892e4d85e24eaac8bacfd4f463576704203",
                "dtype": "torch.bfloat16", "enforce_eager": True,
            },
            "cache_config": {"enable_prefix_caching": False},
            "scheduler_config": {"max_num_seqs": 4},
            "compilation_config": {}, "parallel_config": {},
        }}
        self.report = SimpleNamespace(
            records=[object()], failure_summary={},
            performance_metrics=SimpleNamespace(to_mapping=lambda: {}, backend_completed_sample_count=1,
                                                token_counts={"output_tokens": 2, "output_usage_record_count": 1}),
        )

    def arguments(self, output):
        return ["--config", str(ROOT / "configs/studies/wsl/wsl_vllm_workload_resize_ps20.json"),
                "--output", str(output), "--server-info-url", "http://127.0.0.1:8000/server_info?config_format=json"]

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
            self.assertTrue(execution["server_identity_verified"])
            self.assertEqual(execution["server_identity_before"], {
                "model_revision_matches_client": True,
                "precision_matches_client": True,
            })
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
            self.assertFalse(execution["server_identity_verified"])
            self.assertEqual(execution["status"], "failed")

    def test_server_model_change_after_measurement_fails_the_window(self):
        after = copy.deepcopy(self.payload)
        after["vllm_config"]["model_config"]["revision"] = "other-model"
        with tempfile.TemporaryDirectory() as directory:
            output = Path(directory) / "run.json"
            self.assertEqual(self.run_with_snapshots(output, self.payload, after), 2)
            execution = json.loads(output.with_suffix(".execution.json").read_text())
            self.assertTrue(execution["server_identity_before"]["model_revision_matches_client"])
            self.assertFalse(execution["server_identity_after"]["model_revision_matches_client"])
            self.assertFalse(execution["server_identity_verified"])
            self.assertEqual(execution["status"], "failed")

    def test_missing_or_mismatched_server_identity_stops_before_warmup(self):
        for key, value, expected in (
            ("revision", "other-model", False),
            ("revision", None, None),
            ("dtype", "torch.float16", False),
            ("dtype", None, None),
        ):
            with self.subTest(key=key, value=value), tempfile.TemporaryDirectory() as directory:
                payload = copy.deepcopy(self.payload)
                if value is None:
                    del payload["vllm_config"]["model_config"][key]
                else:
                    payload["vllm_config"]["model_config"][key] = value
                output = Path(directory) / "run.json"
                response = io.BytesIO(json.dumps(payload).encode())
                with patch.object(MODULE, "urlopen", return_value=response), \
                        patch.object(MODULE, "run_configured_study", return_value=self.report) as run, \
                        redirect_stdout(io.StringIO()):
                    self.assertEqual(MODULE.main(self.arguments(output)), 2)
                run.assert_not_called()
                execution = json.loads(output.with_suffix(".execution.json").read_text())
                self.assertEqual(execution["server_identity_before"][
                    "model_revision_matches_client" if key == "revision" else "precision_matches_client"
                ], expected)
                self.assertFalse(execution["server_identity_verified"])
                self.assertEqual(execution["status"], "server_identity_unverified")
                self.assertFalse(output.exists())
                self.assertFalse(output.with_name("run.warmup.json").exists())

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

    def test_evidence_endpoints_must_share_the_inference_origin(self):
        for flag, url in (
            ("--server-info-url", "http://127.0.0.1:8001/server_info?config_format=json"),
            ("--metrics-url", "http://127.0.0.1:8001/metrics"),
        ):
            with self.subTest(flag=flag), tempfile.TemporaryDirectory() as directory:
                output = Path(directory) / "run.json"
                args = self.arguments(output)[:-2] + [flag, url]
                with patch.object(MODULE, "urlopen") as get, \
                        patch.object(MODULE, "run_configured_study") as run, \
                        redirect_stderr(io.StringIO()), self.assertRaises(SystemExit) as error:
                    MODULE.main(args)
                self.assertEqual(error.exception.code, 2)
                get.assert_not_called()
                run.assert_not_called()
                self.assertFalse(output.with_suffix(".execution.json").exists())

    def test_server_counter_mismatch_or_reset_fails_the_window(self):
        before = "vllm:request_success_total 10\nvllm:generation_tokens_total 20\n"
        for requests, tokens, expected_code in ((11, 22, 0), (12, 22, 2), (11, 23, 2), (1, 2, 2)):
            with self.subTest(requests=requests, tokens=tokens), tempfile.TemporaryDirectory() as directory:
                output = Path(directory) / "run.json"
                after = f"vllm:request_success_total {requests}\nvllm:generation_tokens_total {tokens}\n"
                args = self.arguments(output)[:-2] + ["--metrics-url", "http://127.0.0.1:8000/metrics"]
                responses = [io.BytesIO(text.encode()) for text in (before, after)]
                with patch.object(MODULE, "urlopen", side_effect=responses), \
                        patch.object(MODULE, "run_configured_study", return_value=self.report), \
                        redirect_stdout(io.StringIO()):
                    self.assertEqual(MODULE.main(args), expected_code)
                execution = json.loads(output.with_suffix(".execution.json").read_text())
                self.assertEqual(execution["server_metrics_consistent"], expected_code == 0)
                self.assertEqual(execution["status"], "succeeded" if expected_code == 0 else "failed")

    def test_missing_client_usage_cannot_match_server_token_total(self):
        self.report.performance_metrics.token_counts = {
            "output_tokens": 2,
            "output_usage_record_count": 0,
        }
        before = "vllm:request_success_total 10\nvllm:generation_tokens_total 20\n"
        after = "vllm:request_success_total 11\nvllm:generation_tokens_total 22\n"
        with tempfile.TemporaryDirectory() as directory:
            output = Path(directory) / "run.json"
            args = self.arguments(output)[:-2] + ["--metrics-url", "http://127.0.0.1:8000/metrics"]
            responses = [io.BytesIO(text.encode()) for text in (before, after)]
            with patch.object(MODULE, "urlopen", side_effect=responses), \
                    patch.object(MODULE, "run_configured_study", return_value=self.report), \
                    redirect_stdout(io.StringIO()):
                self.assertEqual(MODULE.main(args), 2)
            summary = json.loads(output.with_suffix(".server_metrics.json").read_text())
            execution = json.loads(output.with_suffix(".execution.json").read_text())
            self.assertFalse(summary["client_output_usage_complete"])
            self.assertIsNone(summary["output_token_count_matches_client"])
            self.assertFalse(execution["server_metrics_consistent"])

    def test_length_finished_request_fails_benchmark_even_when_counts_match(self):
        before = (
            'vllm:request_success_total{finished_reason="stop"} 10\n'
            'vllm:request_success_total{finished_reason="length"} 0\n'
            'vllm:generation_tokens_total 20\n'
        )
        after = (
            'vllm:request_success_total{finished_reason="stop"} 10\n'
            'vllm:request_success_total{finished_reason="length"} 1\n'
            'vllm:generation_tokens_total 22\n'
        )
        with tempfile.TemporaryDirectory() as directory:
            output = Path(directory) / "run.json"
            args = self.arguments(output)[:-2] + ["--metrics-url", "http://127.0.0.1:8000/metrics"]
            responses = [io.BytesIO(text.encode()) for text in (before, after)]
            with patch.object(MODULE, "urlopen", side_effect=responses), \
                    patch.object(MODULE, "run_configured_study", return_value=self.report), \
                    redirect_stdout(io.StringIO()):
                self.assertEqual(MODULE.main(args), 2)
            summary = json.loads(output.with_suffix(".server_metrics.json").read_text())
            execution = json.loads(output.with_suffix(".execution.json").read_text())
            self.assertTrue(summary["request_count_matches_client"])
            self.assertTrue(summary["output_token_count_matches_client"])
            self.assertEqual(summary["non_stop_request_count"], 1)
            self.assertFalse(execution["server_metrics_consistent"])

    def test_requested_gpu_telemetry_must_produce_samples_from_live_monitor(self):
        header = "timestamp,memory.used,utilization.gpu,power.draw,temperature.gpu,clocks.current.sm,clocks.current.memory,clocks_event_reasons.sw_power_cap,clocks_event_reasons.hw_thermal_slowdown,clocks_event_reasons.hw_power_brake_slowdown\n"
        sample_values = ",100 MiB,10 %,20 W,40 C,1000 MHz,5000 MHz,Not Active,Not Active,Not Active\n"

        def sample(at: datetime):
            return at.strftime("%Y/%m/%d %H:%M:%S.%f") + sample_values

        class FakeMonitor:
            def __init__(self, exit_code):
                self.exit_code = exit_code

            def poll(self):
                return self.exit_code

            def terminate(self):
                pass

            def wait(self, timeout):
                return -15

        for case, exit_code, expected_code in (
            ("none", None, 2),
            ("repeated_header", None, 2),
            ("prewindow", None, 2),
            ("during", 1, 2),
            ("during", None, 0),
        ):
            with self.subTest(case=case, exit_code=exit_code), tempfile.TemporaryDirectory() as directory:
                output = Path(directory) / "run.json"
                responses = [io.BytesIO(json.dumps(self.payload).encode()) for _ in range(2)]
                monitor_stdout = None
                run_count = 0

                def fake_popen(command, *, stdout):
                    nonlocal monitor_stdout
                    monitor_stdout = stdout
                    stdout.write(header)
                    if case == "repeated_header":
                        stdout.write(header)
                    if case == "prewindow":
                        stdout.write(sample(datetime.now() - timedelta(seconds=5)))
                    stdout.flush()
                    return FakeMonitor(exit_code)

                def fake_run(*args, **kwargs):
                    nonlocal run_count
                    run_count += 1
                    if run_count == 2 and case == "during":
                        assert monitor_stdout is not None
                        monitor_stdout.write(sample(datetime.now()))
                        monitor_stdout.flush()
                    return self.report

                with patch.object(MODULE, "urlopen", side_effect=responses), \
                        patch.object(MODULE, "run_configured_study", side_effect=fake_run), \
                        patch.object(MODULE, "select_gpu_telemetry_fields",
                                     return_value=MODULE.GPU_TELEMETRY_EXTENDED_FIELDS), \
                        patch.object(MODULE.subprocess, "Popen", side_effect=fake_popen), \
                        redirect_stdout(io.StringIO()):
                    actual = MODULE.main(self.arguments(output) + ["--gpu-telemetry"])
                self.assertEqual(actual, expected_code)
                execution = json.loads(output.with_suffix(".execution.json").read_text())
                self.assertEqual(execution["gpu_telemetry_valid"], expected_code == 0)
                self.assertFalse(execution["gpu_telemetry_forced_stop"])
                self.assertEqual(execution["gpu_telemetry_window_sample_count"], int(case == "during"))
                self.assertEqual(execution["gpu_telemetry_window_covered"], case == "during")
                self.assertEqual(execution["status"], "succeeded" if expected_code == 0 else "failed")

    def test_gpu_telemetry_probe_falls_back_to_supported_basic_fields(self):
        header = "timestamp, memory.used [MiB], utilization.gpu [%], power.draw [W], temperature.gpu\n"
        sample = "2026/10/08 10:32:14.051, 7807 MiB, 90 %, 56 W, 74\n"

        def fake_run(command, **kwargs):
            if "clocks.current.sm" in command[1]:
                extended_header = header.rstrip() + ", clocks.current.sm, clocks.current.memory, sw cap, thermal, brake\n"
                invalid_sample = sample.rstrip().replace("56 W", "N/A") + ", 1000 MHz, 5000 MHz, Not Active, Not Active, Not Active\n"
                return SimpleNamespace(returncode=0, stdout=extended_header + invalid_sample)
            return SimpleNamespace(returncode=0, stdout=header + sample)

        with patch.object(MODULE.subprocess, "run", side_effect=fake_run) as probe:
            fields = MODULE.select_gpu_telemetry_fields()
        self.assertEqual(fields, MODULE.GPU_TELEMETRY_BASIC_FIELDS)
        self.assertEqual(probe.call_count, 2)

    def test_unavailable_gpu_telemetry_stops_before_formal_study(self):
        with tempfile.TemporaryDirectory() as directory:
            output = Path(directory) / "run.json"
            response = io.BytesIO(json.dumps(self.payload).encode())
            with patch.object(MODULE, "urlopen", return_value=response), \
                    patch.object(MODULE, "run_configured_study", return_value=self.report) as run, \
                    patch.object(MODULE, "select_gpu_telemetry_fields",
                                 side_effect=RuntimeError("no compatible fields")), \
                    patch.object(MODULE.subprocess, "Popen") as monitor, \
                    redirect_stdout(io.StringIO()):
                self.assertEqual(MODULE.main(self.arguments(output) + ["--gpu-telemetry"]), 2)
            self.assertEqual(run.call_count, 1)  # warm-up only
            monitor.assert_not_called()
            execution = json.loads(output.with_suffix(".execution.json").read_text())
            self.assertEqual(execution["status"], "gpu_telemetry_unavailable")

    def test_gpu_telemetry_parser_accepts_historical_five_column_csv(self):
        with tempfile.TemporaryDirectory() as directory:
            csv_path = Path(directory) / "historical.gpu.csv"
            csv_path.write_text(
                "timestamp, memory.used [MiB], utilization.gpu [%], power.draw [W], temperature.gpu\n"
                "2026/10/08 10:31:00.000, 7000 MiB, 10 %, 30 W, 60\n"
                "2026/10/08 10:32:00.000, 7807 MiB, 90 %, N/A, 74\n"
                "2026/10/08 10:32:14.051, 7807 MiB, 90 %, 56 W, 74\n",
                encoding="utf-8",
            )
            start = datetime(2026, 10, 8, 10, 32, tzinfo=timezone(timedelta(hours=8)))
            end = start + timedelta(minutes=1)
            self.assertEqual(
                MODULE.summarize_gpu_telemetry(csv_path, window_start=start, window_end=end),
                (2, 1),
            )

    def test_gpu_telemetry_needs_samples_across_a_long_window(self):
        start = datetime(2026, 10, 8, 10, 32, tzinfo=timezone.utc)
        end = start + timedelta(seconds=30)
        cases = (
            (20, start + timedelta(seconds=1), end - timedelta(seconds=1), True),
            (1, start + timedelta(seconds=15), start + timedelta(seconds=15), False),
            (20, start + timedelta(seconds=1), start + timedelta(seconds=20), False),
            (20, start + timedelta(seconds=10), end - timedelta(seconds=1), False),
        )
        for count, first, last, expected in cases:
            with self.subTest(count=count, first=first, last=last):
                coverage = MODULE.assess_gpu_telemetry_coverage(
                    window_start=start, window_end=end, sample_count=count,
                    first_sample=first, last_sample=last,
                )
                self.assertEqual(coverage["minimum_window_samples"], 10)
                self.assertEqual(coverage["window_covered"], expected)

    def test_stalled_gpu_monitor_is_killed_after_bounded_wait(self):
        class StalledMonitor:
            def __init__(self):
                self.terminated = False
                self.killed = False

            def poll(self):
                return None

            def terminate(self):
                self.terminated = True

            def wait(self, timeout):
                if timeout != 10:
                    raise AssertionError(f"unexpected timeout: {timeout}")
                if not self.killed:
                    raise MODULE.subprocess.TimeoutExpired("nvidia-smi", timeout)
                return -9

            def kill(self):
                self.killed = True

        monitor = StalledMonitor()
        self.assertEqual(MODULE.stop_gpu_monitor(monitor), (None, True))
        self.assertTrue(monitor.terminated)
        self.assertTrue(monitor.killed)
