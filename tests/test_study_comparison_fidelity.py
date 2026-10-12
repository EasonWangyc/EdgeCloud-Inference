from __future__ import annotations

from copy import deepcopy
import json
from pathlib import Path
import tempfile
import unittest
from contextlib import redirect_stderr
from io import StringIO

from scripts.compare_tensorrt_study_reports import compare_records, compare_study_reports, main


def report():
    return {
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


if __name__ == "__main__":
    unittest.main()
