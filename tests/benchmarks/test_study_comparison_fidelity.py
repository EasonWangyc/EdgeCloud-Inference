from __future__ import annotations

from copy import deepcopy
import json
from pathlib import Path
import tempfile
import unittest
from contextlib import redirect_stderr
from io import StringIO

from scripts.diagnostics.compare_tensorrt_study_reports import compare_records, compare_study_reports, main


def report():
    payload = {
        "study_identity": {"workload_identity": "frozen", "split": "test", "power_mode": "15W",
                           "repetitions": 1, "client_concurrency": 1, "load_pattern": "closed_loop_workers",
                           "runtime_identity": dict.fromkeys(("backend", "backend_revision", "model_id",
                                                              "model_revision", "adapter_revision", "precision"), "fixed")},
        "environment_snapshot": {"hostname": "jetson", "machine": "aarch64", "actual_power_mode": "15W",
                                 "measurement_mode": "serial_native", "input_image_sha256": {"one": "a" * 64}},
        "records": [{"case_id": "one", "workload_identity": "frozen", "input_tokens": 570,
                     "raw_output": "{}", "assessment": {}, "failure": None, "output_tokens": 95,
                     "decode_tokens": 94}],
        "performance_metrics": {"wall_output_tokens_per_second": 7.0, "peak_memory_mb": 4000},
        "quality_metrics": {"json_validity_rate": 0.5},
    }
    payload["records"][0]["runtime_identity"] = deepcopy(
        payload["study_identity"]["runtime_identity"]
    )
    payload["records"][0]["routing_decision"] = None
    return payload


def compare(a, b):
    with tempfile.TemporaryDirectory() as directory:
        paths = [Path(directory) / name for name in ("a.json", "b.json")]
        for path, data in zip(paths, (a, b)):
            path.write_text(json.dumps(data))
        return compare_study_reports(*paths)


class StudyComparisonFidelityTests(unittest.TestCase):
    def test_cli_preserves_existing_evidence(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            source = root / "report.json"
            source.write_text(json.dumps(report()))
            output = root / "comparison.json"
            output.write_text("preserved evidence")
            with redirect_stderr(StringIO()), self.assertRaises(SystemExit) as error:
                main(["--baseline", str(source), "--candidate", str(source), "--output", str(output)])
            self.assertEqual(error.exception.code, 2)
            self.assertEqual(output.read_text(), "preserved evidence")

    def test_throughput_and_quality_use_higher_is_better(self):
        a = report()
        b = deepcopy(a)
        b["performance_metrics"]["wall_output_tokens_per_second"] = 28.0
        b["performance_metrics"]["peak_memory_mb"] = 4200
        b["quality_metrics"]["json_validity_rate"] = 1.0
        result = compare(a, b)
        self.assertEqual(result["schema_version"], "parksight_tensorrt_study_comparison_v3")
        self.assertTrue(result["match"]["comparable"])
        self.assertEqual(result["execution"]["performance"]["wall_output_tokens_per_second"]["speedup"], 4.0)
        self.assertAlmostEqual(result["execution"]["performance"]["peak_memory_mb"]["improvement"], -0.05)
        self.assertEqual(result["quality"]["json_validity_rate"]["improvement"], 1.0)

    def test_same_ids_with_changed_images_or_concurrency_are_not_comparable(self):
        a = report()
        for change in ("image", "concurrency", "missing_environment", "missing_records"):
            b = deepcopy(a)
            if change == "image":
                b["environment_snapshot"]["input_image_sha256"]["one"] = "b" * 64
            elif change == "concurrency":
                b["study_identity"]["client_concurrency"] = 2
            elif change == "missing_environment":
                b.pop("environment_snapshot")
            else:
                b.pop("records")
            with self.subTest(change=change):
                self.assertFalse(compare(a, b)["match"]["comparable"])

    def test_output_change_is_visible_without_invalidating_identical_inputs(self):
        a = report()
        b = deepcopy(a)
        b["records"][0]["raw_output"] = "changed"
        result = compare(a, b)["match"]
        self.assertTrue(result["comparable"])
        self.assertFalse(result["records"]["exact_output_match"])
        self.assertEqual(result["records"]["mismatches"]["raw_output"], ["one"])

    def test_repetition_coverage_and_unknown_tokens_are_not_equal(self):
        rows = report()["records"]
        self.assertFalse(compare_records(rows, rows * 2)["workload_comparable"])
        unknown = deepcopy(rows)
        unknown[0]["decode_tokens"] = None
        result = compare_records(unknown, unknown)
        self.assertFalse(result["exact_output_match"])
        self.assertIn("decode_tokens", result["missing_fields"])

    def test_reordering_is_allowed_but_swapped_field_associations_are_rejected(self):
        rows = [deepcopy(report()["records"][0]) for _ in range(2)]
        rows[1]["raw_output"] = "other"
        rows[1]["assessment"] = {"risk": "other"}
        self.assertTrue(compare_records(rows, list(reversed(rows)))["exact_output_match"])
        candidate = deepcopy(rows)
        candidate[0]["assessment"], candidate[1]["assessment"] = candidate[1]["assessment"], candidate[0]["assessment"]
        self.assertFalse(compare_records(rows, candidate)["exact_output_match"])

    def test_nonfinite_throughput_has_no_speedup(self):
        a = report()
        b = deepcopy(a)
        b["performance_metrics"]["wall_output_tokens_per_second"] = float("inf")
        self.assertIsNone(compare(a, b)["execution"]["performance"]["wall_output_tokens_per_second"]["speedup"])

    def test_same_output_with_different_route_is_not_performance_comparable(self):
        a = report()
        b = deepcopy(a)
        for payload in (a, b):
            payload["study_identity"]["runtime_identity"]["backend"] = "edge_vllm_router"
            payload["records"][0]["runtime_identity"]["backend"] = "vllm"
        a["records"][0]["routing_decision"] = {
            "primary_runtime": "vllm", "selected_runtime": "vllm",
            "attempts": [{"route": "cloud", "succeeded": True}], "fallback_used": False,
            "scene_escalation_attempted": False, "scene_escalation_succeeded": False,
        }
        b["records"][0]["routing_decision"] = {
            "primary_runtime": "tensorrt_edge_llm", "selected_runtime": "vllm",
            "attempts": [
                {"route": "edge", "succeeded": False,
                 "failure": {"category": "timeout"}},
                {"route": "cloud", "succeeded": True},
            ], "fallback_used": True,
            "scene_escalation_attempted": False, "scene_escalation_succeeded": False,
        }
        result = compare(a, b)["match"]
        self.assertTrue(result["records"]["workload_comparable"])
        self.assertTrue(result["records"]["exact_output_match"])
        self.assertFalse(result["records"]["execution_comparable"])
        self.assertEqual(result["records"]["execution_path"]["mismatched_case_ids"], ["one"])
        self.assertFalse(result["comparable"])

    def test_missing_or_changed_per_record_runtime_identity_blocks_comparison(self):
        a = report()
        b = deepcopy(a)
        b["records"][0]["runtime_identity"]["precision"] = "int4"
        result = compare(a, b)["match"]
        self.assertFalse(result["comparable"])
        self.assertEqual(result["records"]["execution_path"]["status"], "mismatch")

        b = deepcopy(a)
        del b["records"][0]["runtime_identity"]
        result = compare(a, b)["match"]
        self.assertFalse(result["comparable"])
        self.assertEqual(result["records"]["execution_path"]["status"], "unverified")

    def test_route_and_input_association_must_survive_repetition_reordering(self):
        rows = [deepcopy(report()["records"][0]) for _ in range(2)]
        rows[1]["input_tokens"] = 600
        rows[1]["routing_decision"] = {
            "primary_runtime": "fixed", "selected_runtime": "fixed",
            "attempts": [{"route": "cloud", "succeeded": True}], "fallback_used": False,
        }
        self.assertTrue(compare_records(rows, list(reversed(rows)))["execution_comparable"])
        candidate = deepcopy(rows)
        candidate[0]["routing_decision"], candidate[1]["routing_decision"] = (
            candidate[1]["routing_decision"], candidate[0]["routing_decision"]
        )
        result = compare_records(rows, candidate)
        self.assertTrue(result["workload_comparable"])
        self.assertFalse(result["execution_comparable"])

    def test_same_fallback_path_with_different_failure_cause_is_not_comparable(self):
        a = report()
        b = deepcopy(a)
        for payload in (a, b):
            payload["study_identity"]["runtime_identity"]["backend"] = "edge_vllm_router"
            payload["records"][0]["runtime_identity"]["backend"] = "vllm"
            payload["records"][0]["routing_decision"] = {
                "primary_runtime": "tensorrt_edge_llm", "selected_runtime": "vllm",
                "attempts": [
                    {"route": "edge", "succeeded": False,
                     "failure": {"category": "timeout"}},
                    {"route": "cloud", "succeeded": True},
                ],
                "fallback_used": True,
            }
        b["records"][0]["routing_decision"]["attempts"][0]["failure"]["category"] = "out_of_memory"
        result = compare(a, b)["match"]
        self.assertTrue(result["records"]["exact_output_match"])
        self.assertFalse(result["records"]["execution_comparable"])
        self.assertFalse(result["comparable"])


if __name__ == "__main__":
    unittest.main()
