"""StudyRunner 质量、性能与失败处理集成测试。"""

from __future__ import annotations

import json
import unittest
from pathlib import Path, PurePosixPath

from parksight_vlm.assessment import ParkingAssessment
from parksight_vlm.casebook import DatasetSplit, ParkingCase, ParkingCaseCatalog
from parksight_vlm.inference import (
    ResourceSnapshot,
    RuntimeGeneration,
    StageTimings,
    TransformersRuntime,
)
from parksight_vlm.studies import StudyDefinition, StudyRunner, StudyValidationError
from parksight_vlm.workload import FrozenWorkload


PROJECT_ROOT = Path(__file__).resolve().parents[2]
FIXTURE_ROOT = PROJECT_ROOT / "tests" / "fixtures" / "inference"
WORKLOAD = FrozenWorkload.load(
    PROJECT_ROOT / "configs" / "workloads" / "parking_risk_v1.json"
)


class SequenceBackend:
    def __init__(self, outputs: list[RuntimeGeneration]) -> None:
        self._outputs = iter(outputs)

    def generate(self, *, image_path: Path, workload: FrozenWorkload) -> RuntimeGeneration:
        return next(self._outputs)


class CountingBackend:
    def __init__(self) -> None:
        self.call_count = 0

    def generate(self, *, image_path: Path, workload: FrozenWorkload) -> RuntimeGeneration:
        self.call_count += 1
        return RuntimeGeneration(raw_output="{}")


class StudyRunnerTests(unittest.TestCase):
    def test_runner_aggregates_quality_performance_and_failures(self) -> None:
        catalog = ParkingCaseCatalog(
            cases=(
                self._case(
                    "case-001",
                    {
                        "schema_version": "parking_risk_v1",
                        "risk_level": "medium",
                        "events": ["narrow_passage"],
                        "evidence": ["A narrow corridor is visible."],
                        "driver_advice": ["slow_down"],
                    },
                ),
                self._case(
                    "case-002",
                    {
                        "schema_version": "parking_risk_v1",
                        "risk_level": "high",
                        "events": ["vru_near_maneuver_path"],
                        "evidence": ["A pedestrian is beside the reversing path."],
                        "driver_advice": ["yield", "prepare_to_stop"],
                    },
                ),
            ),
            manifest_path=Path("manifest.jsonl"),
            annotations_path=Path("annotations.jsonl"),
        )
        catalog.validate()
        backend = SequenceBackend(
            [
                RuntimeGeneration(
                    raw_output=json.dumps(catalog.cases[0].reference_assessment.to_mapping()),
                    stage_timings=StageTimings(decode_ms=20.0),
                    resource_snapshot=ResourceSnapshot(
                        peak_memory_mb=2048.0,
                        average_power_w=12.0,
                        peak_temperature_c=55.0,
                    ),
                    output_tokens=10,
                ),
                RuntimeGeneration(raw_output="invalid-json"),
            ]
        )
        runtime = TransformersRuntime(
            data_root=FIXTURE_ROOT,
            backend=backend,
            backend_revision="test",
            model_id="Qwen/Qwen3-VL-2B-Instruct",
            model_revision="test-revision",
        )
        study = StudyDefinition(
            study_id="quality-test",
            workload=WORKLOAD,
            split=DatasetSplit.TEST,
            repetitions=1,
            power_mode="test-mode",
        )

        report = StudyRunner(
            environment_provider=lambda: {"platform": "unit-test"}
        ).run(catalog, runtime, study)

        self.assertEqual(len(report.records), 2)
        self.assertEqual(report.quality_metrics.json_validity_rate, 0.5)
        self.assertEqual(report.quality_metrics.risk_level_accuracy, 0.5)
        self.assertAlmostEqual(report.quality_metrics.event_micro_f1, 2.0 / 3.0)
        self.assertAlmostEqual(report.quality_metrics.event_macro_f1, 1.0 / 6.0)
        self.assertEqual(
            report.quality_metrics.event_metrics["narrow_passage"]["support"],
            1,
        )
        self.assertEqual(
            report.quality_metrics.event_metrics["narrow_passage"]["f1"],
            1.0,
        )
        self.assertEqual(
            report.quality_metrics.event_metrics["vru_near_maneuver_path"]["false_negative"],
            1,
        )
        self.assertEqual(report.quality_metrics.unsafe_advice_rate, 0.5)
        self.assertEqual(report.quality_metrics.semantic_consistency_rate, 1.0)
        self.assertEqual(report.quality_metrics.semantic_issue_counts, {})
        self.assertEqual(report.failure_summary, {"json_parse_error": 1})
        self.assertEqual(report.performance_metrics.successful_sample_count, 1)
        self.assertEqual(report.performance_metrics.backend_completed_sample_count, 2)
        self.assertIsNone(report.performance_metrics.cold_start_ms)
        self.assertEqual(
            report.performance_metrics.first_request_ms,
            report.records[0].stage_timings.end_to_end_ms,
        )
        self.assertEqual(report.performance_metrics.tokens_per_second, 500.0)
        self.assertAlmostEqual(
            report.performance_metrics.aggregate_output_tokens_per_end_to_end_second,
            10.0 / (report.records[0].stage_timings.end_to_end_ms / 1000.0),
        )
        self.assertEqual(report.performance_metrics.peak_memory_mb, 2048.0)
        self.assertEqual(report.environment_snapshot, {"platform": "unit-test"})
        self.assertEqual(len(report.to_mapping()["records"]), 2)

    def test_runner_rejects_unlabeled_cases_before_inference(self) -> None:
        catalog = ParkingCaseCatalog(
            cases=(
                ParkingCase(
                    case_id="unlabeled",
                    image_ref=PurePosixPath("scene.jpg"),
                    source_group_id="source-unlabeled",
                    split=DatasetSplit.TEST,
                    reference_assessment=None,
                ),
            ),
            manifest_path=Path("manifest.jsonl"),
            annotations_path=Path("annotations.jsonl"),
        )
        backend = CountingBackend()
        runtime = TransformersRuntime(
            data_root=FIXTURE_ROOT,
            backend=backend,
            backend_revision="test",
            model_id="Qwen/Qwen3-VL-2B-Instruct",
            model_revision="test-revision",
        )
        study = StudyDefinition(
            study_id="unlabeled-test",
            workload=WORKLOAD,
            split=DatasetSplit.TEST,
            repetitions=1,
            power_mode="test-mode",
        )

        with self.assertRaisesRegex(
            StudyValidationError, "require reference assessments"
        ):
            StudyRunner().run(catalog, runtime, study)

        self.assertEqual(backend.call_count, 0)

    def test_runner_reports_semantic_issue_without_rewriting_prediction(self) -> None:
        reference = {
            "schema_version": "parking_risk_v1",
            "risk_level": "low",
            "events": [],
            "evidence": ["The maneuver path is visible and clear."],
            "driver_advice": ["maintain_observation"],
        }
        predicted = dict(reference)
        predicted["driver_advice"] = ["prepare_to_stop"]
        catalog = ParkingCaseCatalog(
            cases=(self._case("semantic-case", reference),),
            manifest_path=Path("manifest.jsonl"),
            annotations_path=Path("annotations.jsonl"),
        )
        runtime = TransformersRuntime(
            data_root=FIXTURE_ROOT,
            backend=SequenceBackend(
                [RuntimeGeneration(raw_output=json.dumps(predicted))]
            ),
            backend_revision="test",
            model_id="Qwen/Qwen3-VL-2B-Instruct",
            model_revision="test-revision",
        )
        study = StudyDefinition(
            study_id="semantic-audit-test",
            workload=WORKLOAD,
            split=DatasetSplit.TEST,
            repetitions=1,
            power_mode="test-mode",
        )

        report = StudyRunner().run(catalog, runtime, study)

        self.assertEqual(report.quality_metrics.semantic_consistency_rate, 0.0)
        self.assertEqual(
            report.quality_metrics.semantic_issue_counts,
            {"low_risk_with_prepare_to_stop": 1},
        )
        self.assertEqual(
            report.records[0].assessment.driver_advice[0].value,
            "prepare_to_stop",
        )

    @staticmethod
    def _case(case_id: str, assessment: dict[str, object]) -> ParkingCase:
        return ParkingCase(
            case_id=case_id,
            image_ref=PurePosixPath("scene.jpg"),
            source_group_id=f"source-{case_id}",
            split=DatasetSplit.TEST,
            reference_assessment=ParkingAssessment.from_mapping(assessment),
        )


if __name__ == "__main__":
    unittest.main()
