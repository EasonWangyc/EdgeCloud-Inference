"""区分实际首个请求与未经测量的运行时冷启动。"""

from __future__ import annotations

import unittest
from dataclasses import replace

from parksight_vlm.inference import (
    InferenceRecord,
    ResourceSnapshot,
    RuntimeFailure,
    RuntimeFailureCategory,
    RuntimeIdentity,
    StageTimings,
    StreamTimings,
)
from parksight_vlm.studies.performance import compute_performance_metrics


class PerformanceEvidenceTests(unittest.TestCase):
    def test_failed_first_request_is_recorded_without_claiming_cold_start(self) -> None:
        record = InferenceRecord(
            case_id="case-1",
            runtime_identity=RuntimeIdentity("vllm", "test", "model", "revision", "none", "bf16"),
            workload_identity="workload",
            started_at_utc="2026-10-04T00:00:00+00:00",
            assessment=None,
            failure=RuntimeFailure(RuntimeFailureCategory.TIMEOUT, "timed out", "TimeoutError"),
            raw_output=None,
            stage_timings=StageTimings(end_to_end_ms=12.0),
            resource_snapshot=ResourceSnapshot(),
            output_tokens=None,
        )
        metrics = compute_performance_metrics([record])
        self.assertIsNone(metrics.cold_start_ms)
        self.assertEqual(metrics.first_request_ms, 12.0)
        self.assertEqual(metrics.backend_completed_sample_count, 0)
        self.assertEqual(metrics.stage_latency_ms, {})
        completed = replace(
            record, raw_output="invalid business json", output_tokens=10, input_tokens=20,
            stream_timings=StreamTimings((100.0, 110.0, 130.0)),
        )
        metrics = compute_performance_metrics([record, completed], measurement_wall_seconds=2.0)
        self.assertEqual(metrics.completed_requests_per_second, 0.5)
        self.assertEqual(metrics.wall_output_tokens_per_second, 5.0)
        self.assertEqual(metrics.stream_chunk_interval_ms.p50, 15.0)
        self.assertEqual(metrics.token_counts["input_tokens"], 20)
        missing = replace(completed, output_tokens=None)
        self.assertIsNone(compute_performance_metrics([missing], measurement_wall_seconds=1).wall_output_tokens_per_second)

    def test_empty_study_has_no_first_request_or_cold_start_evidence(self) -> None:
        metrics = compute_performance_metrics([])
        self.assertIsNone(metrics.first_request_ms)
        self.assertIsNone(metrics.cold_start_ms)


if __name__ == "__main__":
    unittest.main()
