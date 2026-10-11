from __future__ import annotations

import unittest

from scripts.compare_tensorrt_nsight_summaries import compare_summaries


def _summary(label: str, kernel_ms: float, sampling_ms: float | None) -> dict[str, object]:
    return {
        "schema_version": "parksight_tensorrt_nsight_trace_summary_v1",
        "input": {"label": label, "path": f"{label}.csv"},
        "trace": {
            "trace_duration_ms": kernel_ms + 1.0,
            "kernel_duration_ms": kernel_ms,
            "transfer_duration_ms": 1.0,
        },
        "categories": {
            "sampling": {
                "count": 1 if sampling_ms is not None else 0,
                "total_ms": sampling_ms,
                "mean_ms": sampling_ms,
                "p50_ms": sampling_ms,
                "p90_ms": sampling_ms,
                "max_ms": sampling_ms,
            }
        },
        "top_kernels": [
            {
                "category": "other_kernel",
                "name": "kernel_a",
                "total_ms": kernel_ms,
            }
        ],
    }


class NsightTraceComparisonTests(unittest.TestCase):
    def test_reports_candidate_minus_baseline_and_kernel_delta(self) -> None:
        result = compare_summaries(
            _summary("level0", 100.0, 4.0), _summary("level1", 80.0, 3.0)
        )

        self.assertEqual(result["comparison"]["baseline_label"], "level0")
        self.assertEqual(result["trace"]["kernel_duration_ms"]["absolute"], -20.0)
        self.assertEqual(result["trace"]["kernel_duration_ms"]["relative_percent"], -20.0)
        self.assertEqual(result["categories"]["sampling"]["total_ms"]["absolute"], -1.0)
        self.assertEqual(result["top_kernel_changes"][0]["delta"]["absolute"], -20.0)

    def test_zero_baseline_has_no_relative_percent(self) -> None:
        result = compare_summaries(_summary("a", 0.0, None), _summary("b", 2.0, 1.0))

        self.assertIsNone(result["trace"]["kernel_duration_ms"]["relative_percent"])
        self.assertEqual(result["categories"]["sampling"]["total_ms"]["baseline"], None)

    def test_rejects_wrong_schema(self) -> None:
        with self.assertRaisesRegex(ValueError, "not a parksight"):
            compare_summaries({"schema_version": "wrong"}, _summary("b", 2.0, 1.0))

    def test_changed_or_missing_classification_version_requires_raw_resummary(self):
        baseline = _summary("a", 100.0, 4.0)
        candidate = _summary("b", 80.0, 3.0)
        candidate["classification_version"] = "parksight_cuda_kernel_names_v2"
        with self.assertRaisesRegex(ValueError, "classification versions"):
            compare_summaries(baseline, candidate)
        baseline["classification_version"] = "older"
        with self.assertRaisesRegex(ValueError, "classification versions"):
            compare_summaries(baseline, candidate)
        baseline["classification_version"] = candidate["classification_version"]
        self.assertEqual(compare_summaries(baseline, candidate)["trace"]["kernel_duration_ms"]["absolute"], -20.0)


if __name__ == "__main__":
    unittest.main()
