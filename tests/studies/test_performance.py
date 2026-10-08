"""区分实际首个请求与未经测量的运行时冷启动。"""

from __future__ import annotations

import unittest
import json
from dataclasses import replace

from parksight_vlm.assessment import ParkingAssessment

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
        self.assertEqual(metrics.attempted_sample_count, 1)
        self.assertEqual(metrics.failed_sample_count, 1)
        self.assertEqual(metrics.all_request_stage_latency_ms["end_to_end_ms"].p99, 12.0)
        self.assertEqual(metrics.failed_request_stage_latency_ms["end_to_end_ms"].count, 1)
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
        self.assertEqual(metrics.attempted_sample_count, 0)
        self.assertEqual(metrics.failed_sample_count, 0)
        self.assertEqual(metrics.all_request_stage_latency_ms, {})
        self.assertEqual(metrics.failed_request_stage_latency_ms, {})
        self.assertIsNone(metrics.attempted_requests_per_second)
        self.assertIsNone(metrics.successful_requests_per_second)

    def test_timeout_tail_and_invalid_json_cost_are_visible_in_separate_populations(self):
        assessment = ParkingAssessment.from_mapping({
            "schema_version": "parking_risk_v1", "risk_level": "low", "events": [],
            "evidence": ["Clear path"], "driver_advice": ["maintain_observation"],
        })
        success = InferenceRecord(
            case_id="success", runtime_identity=RuntimeIdentity("vllm", "test", "model", "rev", "none", "bf16"),
            workload_identity="workload", started_at_utc="2026-10-08T00:00:00Z",
            assessment=assessment, failure=None, raw_output=json.dumps(assessment.to_mapping()),
            stage_timings=StageTimings(end_to_end_ms=20, decode_ms=2),
            resource_snapshot=ResourceSnapshot(peak_memory_mb=1000, average_power_w=10, peak_temperature_c=60), output_tokens=5,
        )
        invalid = replace(success, case_id="invalid", assessment=None, raw_output="not-json",
                          failure=RuntimeFailure(RuntimeFailureCategory.JSON_PARSE_ERROR, "invalid", "ValueError"),
                          stage_timings=StageTimings(end_to_end_ms=100, decode_ms=1), output_tokens=10)
        timeout = replace(success, case_id="timeout", assessment=None, raw_output=None,
                          failure=RuntimeFailure(RuntimeFailureCategory.TIMEOUT, "timeout", "TimeoutError"),
                          stage_timings=StageTimings(end_to_end_ms=2000), output_tokens=None,
                          resource_snapshot=ResourceSnapshot(peak_memory_mb=2000, average_power_w=30, peak_temperature_c=85))
        input_error = replace(timeout, case_id="input", stage_timings=StageTimings(end_to_end_ms=0),
                              failure=RuntimeFailure(RuntimeFailureCategory.INPUT_ERROR, "missing", "OSError"),
                              resource_snapshot=ResourceSnapshot())
        metrics = compute_performance_metrics([success, invalid, timeout, input_error], measurement_wall_seconds=3)
        self.assertEqual(metrics.attempted_sample_count, 4)
        self.assertEqual(metrics.failed_sample_count, 3)
        self.assertEqual(metrics.backend_completed_sample_count, 2)
        self.assertEqual(metrics.successful_sample_count, 1)
        self.assertEqual(metrics.attempted_requests_per_second, 4 / 3)
        self.assertEqual(metrics.successful_requests_per_second, 1 / 3)
        self.assertEqual(metrics.completed_requests_per_second, 2 / 3)
        self.assertEqual(metrics.wall_output_tokens_per_second, 5)
        self.assertEqual(metrics.stage_latency_ms["end_to_end_ms"].count, 2)
        self.assertEqual(metrics.all_request_stage_latency_ms["end_to_end_ms"].count, 4)
        self.assertAlmostEqual(metrics.all_request_stage_latency_ms["end_to_end_ms"].p90, 1430)
        self.assertEqual(metrics.failed_request_stage_latency_ms["end_to_end_ms"].count, 3)
        self.assertEqual(metrics.all_request_stage_latency_ms["decode_ms"].count, 2)
        self.assertEqual(metrics.failed_request_stage_latency_ms["decode_ms"].count, 1)
        mapping = metrics.to_mapping()
        self.assertEqual(mapping["failed_request_stage_latency_ms"]["end_to_end_ms"]["p50"], 100)
        self.assertEqual(mapping["attempted_sample_count"], 4)
        self.assertEqual(metrics.peak_memory_mb, 2000)
        self.assertEqual(metrics.peak_temperature_c, 85)
        self.assertAlmostEqual(metrics.average_power_w, 50 / 3)

    def test_decode_throughput_uses_only_explicit_counts_with_matching_phase_time(self):
        failure = RuntimeFailure(RuntimeFailureCategory.JSON_PARSE_ERROR, "invalid", "ValueError")
        record = InferenceRecord(
            case_id="case", runtime_identity=RuntimeIdentity("transformers", "test", "model", "rev", "none", "fp16"),
            workload_identity="workload", started_at_utc="2026-10-08T00:00:00Z",
            assessment=None, failure=failure, raw_output="not-json", resource_snapshot=ResourceSnapshot(),
            stage_timings=StageTimings(decode_ms=20), output_tokens=10, decode_tokens=9,
        )
        metrics = compute_performance_metrics([record])
        self.assertEqual(metrics.tokens_per_second, 500)  # Historical normalization.
        self.assertEqual(metrics.decode_tokens_per_second, 450)
        self.assertEqual(metrics.token_counts["decode_tokens"], 9)
        self.assertEqual(metrics.token_counts["decode_rate_record_count"], 1)
        unknown = replace(record, decode_tokens=None)
        metrics = compute_performance_metrics([record, unknown])
        self.assertEqual(metrics.decode_tokens_per_second, 450)
        self.assertEqual(metrics.token_counts["decode_usage_record_count"], 1)
        self.assertEqual(metrics.token_counts["decode_rate_record_count"], 1)
        self.assertIsNone(compute_performance_metrics([unknown]).decode_tokens_per_second)
        first_only = replace(record, output_tokens=1, decode_tokens=0, stage_timings=StageTimings(prefill_ms=20))
        self.assertIsNone(compute_performance_metrics([first_only]).decode_tokens_per_second)
        self.assertEqual(compute_performance_metrics([first_only]).token_counts["decode_usage_record_count"], 1)


if __name__ == "__main__":
    unittest.main()
