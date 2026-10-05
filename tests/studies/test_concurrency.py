"""并发研究使用独立运行时，保存提交顺序，并关闭连接资源。"""

from __future__ import annotations

import unittest
import json
from dataclasses import replace
from pathlib import Path, PurePosixPath
from threading import Barrier, Lock

from parksight_vlm.assessment import ParkingAssessment
from parksight_vlm.casebook import DatasetSplit, ParkingCase, ParkingCaseCatalog
from parksight_vlm.inference import (
    InferenceRecord, ResourceSnapshot, RiskRuntime, RuntimeGeneration,
    RuntimeIdentity, StageTimings,
)
from parksight_vlm.studies import StudyDefinition, StudyRunner, StudyValidationError
from parksight_vlm.workload import FrozenWorkload


ROOT = Path(__file__).resolve().parents[2]
WORKLOAD = FrozenWorkload.load(ROOT / "configs/workloads/parking_risk_v1.json")
IDENTITY = RuntimeIdentity("vllm", "test", "model", "revision", "none", "bf16")
ASSESSMENT = ParkingAssessment.from_mapping({
    "schema_version": "parking_risk_v1", "risk_level": "low", "events": [],
    "evidence": ["Clear path"], "driver_advice": ["maintain_observation"],
})


class WorkerRuntime(RiskRuntime):
    def __init__(self, barrier: Barrier | None = None) -> None:
        super().__init__(data_root=ROOT, identity=IDENTITY)
        self.barrier = barrier
        self.calls = 0
        self.closed = False
        self.guard = Lock()

    def analyze(self, case: ParkingCase, workload: FrozenWorkload) -> InferenceRecord:
        if not self.guard.acquire(blocking=False):
            raise AssertionError("same runtime used simultaneously")
        try:
            self.calls += 1
            if self.calls == 1 and self.barrier is not None:
                self.barrier.wait(timeout=5)
            return InferenceRecord(
                case_id=case.case_id, runtime_identity=self.identity, workload_identity=workload.identity,
                started_at_utc="2026-10-05T00:00:00+00:00", assessment=ASSESSMENT,
                failure=None, raw_output=json.dumps(ASSESSMENT.to_mapping()),
                stage_timings=StageTimings(end_to_end_ms=10),
                resource_snapshot=ResourceSnapshot(), output_tokens=5,
            )
        finally:
            self.guard.release()

    def close(self) -> None:
        self.closed = True

    def _generate(self, *, image_path: Path, workload: FrozenWorkload) -> RuntimeGeneration:
        raise AssertionError("not used")


class ConcurrentStudyTests(unittest.TestCase):
    def setUp(self) -> None:
        self.catalog = ParkingCaseCatalog(manifest_path=None, annotations_path=None, cases=tuple(
            ParkingCase(case_id=f"case-{i}", image_ref=PurePosixPath("scene.jpg"),
                        source_group_id=f"source-{i}", split=DatasetSplit.TEST,
                        reference_assessment=ASSESSMENT)
            for i in range(4)
        ))
        self.study = StudyDefinition("concurrent-test", WORKLOAD, DatasetSplit.TEST, 1, "test")

    def test_workers_overlap_without_sharing_runtime_and_preserve_order(self) -> None:
        for concurrency in (2, 4):
            with self.subTest(concurrency=concurrency):
                barrier = Barrier(concurrency)
                workers = []

                def factory():
                    worker = WorkerRuntime(barrier)
                    workers.append(worker)
                    return worker

                report = StudyRunner().run(
                    self.catalog, WorkerRuntime(), self.study,
                    concurrency=concurrency, runtime_factory=factory,
                )
                self.assertEqual(len(workers), concurrency)
                self.assertTrue(all(worker.closed for worker in workers))
                self.assertEqual(sum(worker.calls for worker in workers), 4)
                self.assertEqual([r.case_id for r in report.records], [c.case_id for c in self.catalog.cases])
                self.assertEqual(report.study_identity["client_concurrency"], concurrency)
                self.assertEqual(report.performance_metrics.backend_completed_sample_count, 4)
                self.assertIsNotNone(report.performance_metrics.wall_output_tokens_per_second)

    def test_concurrency_requires_independent_factory(self) -> None:
        runtime = WorkerRuntime()
        with self.assertRaises(StudyValidationError):
            StudyRunner().run(self.catalog, runtime, self.study, concurrency=2)
        with self.assertRaises(StudyValidationError):
            StudyRunner().run(self.catalog, runtime, self.study, concurrency=2, runtime_factory=lambda: runtime)

    def test_worker_identity_failure_still_closes_created_runtime(self) -> None:
        worker = WorkerRuntime()
        worker._identity = replace(IDENTITY, model_revision="different")
        with self.assertRaises(StudyValidationError):
            StudyRunner().run(self.catalog, WorkerRuntime(), self.study, concurrency=2, runtime_factory=lambda: worker)
        self.assertTrue(worker.closed)

    def test_invalid_concurrency_is_rejected(self) -> None:
        for value in (0, -1, True, 1.5):
            with self.subTest(value=value), self.assertRaises(StudyValidationError):
                StudyRunner().run(self.catalog, WorkerRuntime(), self.study, concurrency=value)
